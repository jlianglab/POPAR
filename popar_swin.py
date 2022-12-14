import torch.nn as nn

import torch
import argparse
import torch.backends.cudnn as cudnn
from utils import AverageMeter,save_model
import time
import os
import config

from datasets import Popar_chestxray, build_md_transform
import wandb
import sys
import numpy as np
from timm.utils import NativeScaler, get_state_dict, ModelEma
import math
from torch import optim as optim
from swin_transformer import SwinTransformer
from einops import rearrange





parser = argparse.ArgumentParser('argument for training')
parser.add_argument('--batch_size', type=int, default=256,  help='batch_size')
parser.add_argument('--num_workers', type=int, default=10, help='num of workers to use')
parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
parser.add_argument('--epochs', type=int, default=2000, help='number of training epochs')
parser.add_argument('--gpu', dest='gpu', default="0,1,2,3", type=str, help="gpu index")
parser.add_argument('--task', dest='task', default="POPAR_swin", type=str)
parser.add_argument('--dataset', dest='dataset', default="nih14", type=str)
parser.add_argument('--weight', dest='weight', default=None)
parser.add_argument('--test', dest='test', default=False, action="store_true")
parser.add_argument('--depth', dest='depth', type=str, default="2,2,18,2")
parser.add_argument('--heads', dest='heads', type=str, default="4,8,16,32")
parser.add_argument('--in_channel', dest='in_channel', default=3, type=int, help="input color channel")

args = parser.parse_args()

def step_decay(step, conf,warmup_epochs = 5):
    lr = conf.lr
    progress = (step - warmup_epochs) / float(conf.epochs - warmup_epochs)
    progress = np.clip(progress, 0.0, 1.0)
    #decay_type == 'cosine':
    lr = lr * 0.5 * (1. + np.cos(np.pi * progress))
    if warmup_epochs:
      lr = lr * np.minimum(1., step / warmup_epochs)
    return lr


class _SwinTransformer(SwinTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert self.num_classes == 0

    def forward(self, x):
        x = self.patch_embed(x)

        B, L, _ = x.shape
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)

        # x = x.transpose(1, 2)
        # B, C, L = x.shape
        # H = W = int(L ** 0.5)
        # x = x.reshape(B, C, H, W)

        return x

    @torch.jit.ignore
    def no_weight_decay(self):
        return super().no_weight_decay() | {'mask_token'}


class SwinModel(nn.Module):
    def __init__(self,hidden_size = 128, num_classes = 196, depth=[ 2, 2, 18, 2 ],heads=[ 4, 8, 16, 32 ], device = "gpu"):
        super(SwinModel, self).__init__()

        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.device = device
        self.depth = depth
        self.heads = heads
        self.swin_model = _SwinTransformer(img_size=448,patch_size=4,in_chans=3,num_classes=0,embed_dim=self.hidden_size,depths=self.depth,num_heads= self.heads,
                                          window_size=7,mlp_ratio=4.,qkv_bias=True,qk_scale=None,drop_rate=0,drop_path_rate=0.1,ape=False,patch_norm=True,use_checkpoint=False)

        self.head = nn.Linear(1024 , self.num_classes,bias=False)
        self.bias = nn.Parameter(torch.zeros(self.num_classes))
        self.head.bias = self.bias
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.decoder = nn.Sequential(
            nn.Conv2d(
                in_channels=1024,
                out_channels=32 ** 2 * 3, kernel_size=1),
            nn.PixelShuffle(32),
            # nn.Conv2d(3, 3, kernel_size=1),
            # nn.BatchNorm2d(3),
            # nn.GELU(),
            # nn.Conv2d(3, 3, kernel_size=1),
            # nn.BatchNorm2d(3),
            # nn.GELU(),
        )

    def forward(self, img_x,perm):
        B,C,H,W = img_x.shape

        img_x = rearrange(img_x, 'b c (h p1) (w p2)-> b (h w) c p1 p2', p1=32, p2=32, w=14,h=14)
        for i in range(B):
            img_x[i] = img_x[i,perm[i],:,:,:]
        img_x = rearrange(img_x, 'b (h w) c p1 p2 -> b c (h p1) (w p2)', p1=32, p2=32, w=14,h=14)
        out = self.swin_model(img_x)
        B, L, C = out.shape

        cls_feature = out.reshape(-1, 1024)
        restor_feature = out.transpose(1, 2)
        H = W = int(L ** 0.5)
        restor_feature = restor_feature.reshape(B, C, H, W)

        decoder_out = self.decoder(restor_feature)

        return decoder_out, self.head(cls_feature)

def build_model(conf):
    start_epoch = 1

    if conf.weight is None and not conf.test:
        model = SwinModel()
    else:
        print("Loading pretrained weights", file=conf.log_writter)
        if conf.weight is None:
            model = SwinModel()
            checkpoint = torch.load(os.path.join(conf.model_path, "last.pth"), map_location='cpu')
            state_dict = {k.replace("module.", ""): v for k, v in checkpoint['model'].items()}
            model.load_state_dict(state_dict)
        else:
            model = SwinModel()
            checkpoint = torch.load(conf.weight, map_location='cpu')
            state_dict = {k.replace("module.", ""): v for k, v in checkpoint['model'].items()}
            model.load_state_dict(state_dict)
            start_epoch = checkpoint['epoch'] + 1

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    #optimizer = AdamW(optimizer_grouped_parameters, lr=conf.lr)
    optimizer = optim.SGD(model.parameters(), lr=conf.lr, weight_decay=0, momentum=0.9, nesterov=False)
    model = model.double()
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model, device_ids=[i for i in range(torch.cuda.device_count())])
        model = model.cuda()
        cudnn.benchmark = True
    loss_scaler = NativeScaler()

    return model, optimizer,loss_scaler,start_epoch


def train(train_loader, model, optimizer, epoch,loss_scaler, conf):
    """one epoch training"""
    model.train(True)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    order_losses = AverageMeter()
    restor_losses = AverageMeter()

    end = time.time()
    ce_loss = nn.CrossEntropyLoss()
    mse_loss =nn.MSELoss()
    for idx, (randperm,gt_whole,aug_whole) in enumerate(train_loader):

        data_time.update(time.time() - end)
        bsz = aug_whole.shape[0]

        if torch.cuda.is_available():
            aug_whole = aug_whole.double().cuda(non_blocking=True)
            gt_whole = gt_whole.double().cuda(non_blocking=True)
            randperm = randperm.long().cuda(non_blocking=True)



        pred_restor,pred_order = model(aug_whole, randperm)

        randperm = randperm.reshape(-1)
        gt_whole = gt_whole.reshape(pred_restor.shape)

        order_loss = ce_loss(pred_order, randperm)
        restor_loss = mse_loss(pred_restor,gt_whole)
        loss = (order_loss + restor_loss)/2

        if not math.isfinite( loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), file=conf.log_writter)
            sys.exit(1)
        # update metric
        order_losses.update(order_loss.item(),bsz)
        restor_losses.update(restor_loss.item(),bsz)
        losses.update(loss.item(), bsz)
        print("in train", randperm.shape[0])

        optimizer.zero_grad()
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=None,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % 10 == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'lr {lr}\t'
                  'Order loss {orderloss.val:.3f} ({orderloss.avg:.3f})\t'
                  'Restor loss {restorloss.val:.3f} ({restorloss.avg:.3f})\t'
                  'Total loss {ttloss.val:.3f} ({ttloss.avg:.3f})'.format(
                epoch, idx + 1, len(train_loader), batch_time=batch_time,
                data_time=data_time, ttloss=losses, restorloss=restor_losses, orderloss=order_losses,
                lr=optimizer.param_groups[0]['lr']), file=conf.log_writter)
            conf.log_writter.flush()
            if conf.debug_mode:
                break

    wandb.log({"order loss": order_losses.avg,
               "restor loss": restor_losses.avg})

    return losses.avg






def test(test_loader, model,conf):
    """one epoch training"""
    model.eval()

    matches = 0
    total = 0
    mse_loss =nn.MSELoss()
    restor_losses = AverageMeter()

    with torch.no_grad():
        for idx,  (randperm,gt_whole,aug_whole) in enumerate(test_loader):
            bsz = aug_whole.shape[0]

            if torch.cuda.is_available():
                aug_whole = aug_whole.double().cuda(non_blocking=True)
                gt_whole = gt_whole.double().cuda(non_blocking=True)
                randperm = randperm.long().cuda(non_blocking=True)

            pred_restor, pred_order = model(aug_whole, randperm)
            gt_whole = gt_whole.reshape(pred_restor.shape)
            restor_losses.update(mse_loss(pred_restor, gt_whole).item(), bsz)

            tp1 = pred_order.argmax(dim=1)
            randperm = randperm.reshape(-1)

            print("predicted order: ", tp1, file=conf.log_writter)
            print("gt order: ", randperm, file=conf.log_writter)

            matches += (tp1 == randperm).sum()
            total += randperm.shape[0]
            print("in test", randperm.shape[0])

            if conf.debug_mode:
                break

    #matches = matches.item()
    accuracy = matches / total

    wandb.log({"test restor loss": restor_losses.avg,
               "test accuracy": accuracy})

    return accuracy, restor_losses.avg




def main(conf):

    wandb.login()
    with wandb.init(project=conf.method, config = conf):

        model, optimizer,loss_scaler,start_epoch = build_model(conf)

        train_dataset = Popar_chestxray(conf.train_image_path_file, build_md_transform(mode="train", dataset="chexray"),image_size=448,patch_size=32 )
        test_dataset = Popar_chestxray(conf.test_image_path_file, build_md_transform(mode="validation", dataset="chexray"), image_size=448,patch_size=32)

        sampler_train = torch.utils.data.RandomSampler(train_dataset)
        sampler_test = torch.utils.data.SequentialSampler(test_dataset)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=conf.batch_size, num_workers=conf.num_workers,pin_memory=True,sampler=sampler_train)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=conf.batch_size, num_workers=conf.num_workers, pin_memory=True,sampler=sampler_test)

        print(model, file=conf.log_writter)
        print("train size: {} test size:{}".format(len(train_loader), len(test_dataset)))
        wandb.watch(model, criterion=None, log="all", log_freq=5, log_graph=True)

        for epoch in range(start_epoch, conf.epochs + 1):
            time1 = time.time()

            lr_ = step_decay(epoch,conf)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_
            loss = train(train_loader, model, optimizer , epoch,loss_scaler, conf)
            time2 = time.time()
            print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1),file = conf.log_writter)

            # tensorboard logger
            print('loss: {}@Epoch: {}'.format(loss,epoch),file = conf.log_writter)
            print('learning_rate: {},{}'.format(optimizer.param_groups[0]['lr'],epoch),file = conf.log_writter)
            conf.log_writter.flush()
            if epoch % 10 == 0 or epoch == 1 or conf.dataset == "imagenet" or conf.dataset=="allxrays":
                save_file = os.path.join(conf.model_path, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
                save_model(model, optimizer, conf, epoch, save_file)

                accuracy, restor_losses = test(test_loader, model,conf)
                print('Accuracy: {}. Restoration loss: {}'.format(accuracy, restor_losses), file=conf.log_writter)
                conf.log_writter.flush()
                if conf.debug_mode:
                    break


        # save the last model
        save_file = os.path.join(conf.model_path, 'last.pth')
        save_model(model, optimizer, conf, conf.epochs, save_file)

        accuracy, restor_losses = test(test_loader, model,conf)
        print('Accuracy: {}. Restoration loss: {}'.format(accuracy, restor_losses), file=conf.log_writter)
        conf.log_writter.flush()



if __name__ == '__main__':


    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


    if args.dataset == "nih14":
        conf = config.ChestX_ray14(args)
    elif args.dataset == "imagenet":
        conf = config.ImageNet(args)
    elif args.dataset == "allxrays":
        conf = config.AllXray_config(args)
    else:
        print("Dataset doest not exist. Exit!")
        exit(-1)

    conf.display()
    main(conf)