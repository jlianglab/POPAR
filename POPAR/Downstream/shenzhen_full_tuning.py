import torch.nn as nn

import torch
import argparse
import torch.backends.cudnn as cudnn
from utils import AverageMeter, computeAUROC, save_checkpoint, load_pretrained,load_swin_pretrained
import time
import os
import config

from datasets import build_classfication_dataset
import sys
import numpy as np
from timm.utils import NativeScaler
import math
from torch import optim as optim
from swin_transformer import SwinTransformer
from swin_transformer_v2 import SwinTransformerV2


import copy





parser = argparse.ArgumentParser('argument for training')
parser.add_argument('--batch_size', type=int, default=60,  help='batch_size')
parser.add_argument('--num_workers', type=int, default=10, help='num of workers to use')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--epochs', type=int, default=200, help='number of training epochs')
parser.add_argument('--gpu', dest='gpu', default="0,1", type=str, help="gpu index")
parser.add_argument('--run', dest='run', default="shenzhen_full_tuning", type=str)
parser.add_argument('--weight', dest='weight', default=None)
parser.add_argument('--anno_percent', type=int, default=100, help='percent of traning samples')
parser.add_argument('--patience', type=int, default=20, help='percent of traning samples')
parser.add_argument('--optm', dest='optm', default="sgd", type=str, help="Optimizer")
parser.add_argument('--img_size', type=int, default=448, help='Input resolution')
parser.add_argument('--test_only', action='store_true', default=False)
parser.add_argument('--ape', dest='ape', action="store_true")

args = parser.parse_args()

def step_decay(step, conf,warmup_epochs = 0):
    lr = conf.lr
    progress = (step - warmup_epochs) / float(conf.epochs - warmup_epochs)
    progress = np.clip(progress, 0.0, 1.0)
    #decay_type == 'cosine':
    lr = lr * 0.5 * (1. + np.cos(np.pi * progress))
    if warmup_epochs:
      lr = lr * np.minimum(1., step / warmup_epochs)
    return lr



class ModelWrapper(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.cls_head = torch.nn.Sequential(torch.nn.Linear(1024, 512),
                                    torch.nn.BatchNorm1d(512, affine=False, eps=1e-6),
                                    torch.nn.ReLU(inplace=True),
                                    torch.nn.Linear(512, 256),
                                    torch.nn.BatchNorm1d(256, affine=False, eps=1e-6),
                                    torch.nn.ReLU(inplace=True),
                                    torch.nn.Linear(256, 1))

    def forward(self, image):
        embedding = self.backbone(image)
        return self.cls_head(embedding)



def build_model(conf):
    start_epoch = 1

    if conf.weight == "popar_swin_allxrays_448":
        model = SwinTransformer(img_size=448, num_classes=14)
        checkpoint = torch.load("popar_swin_allxrays_448.pth", map_location='cpu')
        checkpoint = {k.replace("module.", ""): v for k, v in checkpoint.items()}
        checkpoint = {k.replace("backbone.", ""): v for k, v in checkpoint.items()}
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in checkpoint:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint[k]
        result = model.load_state_dict(checkpoint, strict=False)

        print(result, file=conf.log_writter)
        model = ModelWrapper(model)
    elif conf.weight == "popar_swin_nih14_448":
        model = SwinTransformer(img_size=448, num_classes=14)
        checkpoint = torch.load("popar_swin_nih14_448.pth", map_location='cpu')
        checkpoint = {k.replace("module.", ""): v for k, v in checkpoint.items()}
        checkpoint = {k.replace("backbone.", ""): v for k, v in checkpoint.items()}
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in checkpoint:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint[k]
        result = model.load_state_dict(checkpoint, strict=False)

        print(result, file=conf.log_writter)
        model = ModelWrapper(model)
    elif conf.weight == "popar_swinv2_allxrays_512":
        model = SwinTransformerV2(img_size=512, num_classes=14, in_chans=3, embed_dim=128, depths=[2, 2, 18, 2],
                                  num_heads=[4, 8, 16, 32], window_size=16, use_ape=False, )
        checkpoint = torch.load("popar_swinv2_allxrays_512.pth", map_location='cpu')

        checkpoint = {k.replace("module.", ""): v for k, v in checkpoint.items()}
        checkpoint = {k.replace("backbone.", ""): v for k, v in checkpoint.items()}
        checkpoint = {k.replace("swin_model.", ""): v for k, v in checkpoint.items()}

        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in checkpoint:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint[k]
        result = model.load_state_dict(checkpoint, strict=False)

        print(result, file=conf.log_writter)
        model = ModelWrapper(model)
    elif conf.weight == "popar_swinv2_nih14_448":
        model = SwinTransformerV2(img_size=512, num_classes=14, in_chans=3, embed_dim=128, depths=[2, 2, 18, 2],
                                  num_heads=[4, 8, 16, 32], window_size=16, use_ape=False, )
        checkpoint = torch.load("popar_swinv2_nih14_448.pth", map_location='cpu')

        checkpoint = {k.replace("module.", ""): v for k, v in checkpoint.items()}
        checkpoint = {k.replace("backbone.", ""): v for k, v in checkpoint.items()}
        checkpoint = {k.replace("swin_model.", ""): v for k, v in checkpoint.items()}

        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in checkpoint:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint[k]
        result = model.load_state_dict(checkpoint, strict=False)

        print(result, file=conf.log_writter)
        model = ModelWrapper(model)


    if conf.optm == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=conf.lr)
    else:
        optimizer = optim.SGD(model.parameters(), lr=conf.lr, weight_decay=0, momentum=0.9, nesterov=False)

    loss_scaler = NativeScaler()



    model = model.cuda()
    model = torch.nn.DataParallel(model, device_ids=[i for i in range(torch.cuda.device_count())])

    cudnn.benchmark = True


    return model, optimizer,loss_scaler, start_epoch


def train(train_loader, model, optimizer, epoch,loss_scaler, conf):
    """one epoch training"""
    model.train(True)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()
    bce_loss = nn.BCEWithLogitsLoss()
    for idx, (image,lbl ) in enumerate(train_loader):

        data_time.update(time.time() - end)
        image = image.float().cuda(non_blocking=True)
        lbl = lbl.cuda(non_blocking=True)
        pred = model(image)
        loss = bce_loss(pred, lbl)

        losses.update(loss.item(),image.shape[0])
        if not math.isfinite( loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), file=conf.log_writter)
            sys.exit(1)

        optimizer.zero_grad()
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=None, parameters=model.parameters(), create_graph=is_second_order)
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
                  'loss {ttloss.val:.3f} ({ttloss.avg:.3f})'.format(
                epoch, idx + 1, len(train_loader), batch_time=batch_time,
                data_time=data_time, ttloss=losses, lr=optimizer.param_groups[0]['lr']), file=conf.log_writter)
            conf.log_writter.flush()
            if conf.debug_mode:
                break


    return losses.avg



def evaluate(valid_loader, model, epoch):
    model.eval()
    bce_loss = nn.BCEWithLogitsLoss()
    losses = AverageMeter()
    with torch.no_grad():
        for idx, (image,lbl) in enumerate(valid_loader):
            image = image.float().cuda(non_blocking=True)
            lbl = lbl.cuda(non_blocking=True)
            pred = model(image)
            loss = bce_loss(pred, lbl)

            losses.update(loss.item(),image.shape[0])
            if (idx + 1) % 50 == 0:
                print('Valid: [{0}][{1}/{2}]\t'
                      'loss {ttloss.val:.3f} ({ttloss.avg:.3f})'.format(epoch, idx + 1, len(valid_loader), ttloss=losses), file=conf.log_writter)
                conf.log_writter.flush()
                if conf.debug_mode:
                    break

    return losses.avg


def test(model_path, test_loader,conf):
    if conf.weight == "popar_swin_allxrays_448":
        model = SwinTransformer(img_size=448, num_classes=14)
        model = ModelWrapper(model)
    elif conf.weight == "popar_swin_nih14_448":
        model = SwinTransformer(img_size=448, num_classes=14)
        model = ModelWrapper(model)
    elif conf.weight == "popar_swinv2_allxrays_512":
        model = SwinTransformerV2(img_size=512, num_classes=14, in_chans=3, embed_dim=128, depths=[2, 2, 18, 2],
                                  num_heads=[4, 8, 16, 32], window_size=16, use_ape=False, )
        model = ModelWrapper(model)
    elif conf.weight == "popar_swinv2_nih14_448":
        model = SwinTransformerV2(img_size=512, num_classes=14, in_chans=3, embed_dim=128, depths=[2, 2, 18, 2],
                                  num_heads=[4, 8, 16, 32], window_size=16, use_ape=False, )
        model = ModelWrapper(model)

    checkpoint = torch.load(model_path, map_location='cpu')
    state_dict = {k.replace("module.", ""): v for k, v in checkpoint['model'].items()}
    model.load_state_dict(state_dict)
    model = model.cuda()
    model = torch.nn.DataParallel(model, device_ids=[i for i in range(torch.cuda.device_count())])




    model.eval()
    y_test = torch.FloatTensor().cuda()
    p_test = torch.FloatTensor().cuda()

    with torch.no_grad():
        for idx, (images,lbl) in enumerate(test_loader):
            images = images.float().cuda(non_blocking=True)
            lbl = lbl.cuda(non_blocking=True)

            if len(images.size()) == 4:
                bs, c, h, w = images.size()
                n_crops = 1
            elif len(images.size()) == 5:
                bs, n_crops, c, h, w = images.size()
            with torch.no_grad():
                varInput = torch.autograd.Variable(images.view(-1, c, h, w).cuda())

                out = model(varInput)
                out = torch.sigmoid(out)
                outMean = out.view(bs, n_crops, -1).mean(1)
                p_test = torch.cat((p_test, outMean.data), 0)

                lbl = lbl.type_as(out)
                y_test = torch.cat((y_test, lbl), 0)

            if (idx + 1) % 50 == 0:
                print('Testing: [{0}/{1}]'.format(idx + 1, len(test_loader)), file=conf.log_writter)
                conf.log_writter.flush()



            if conf.debug_mode and idx == 100:
                break



    aurocIndividual = computeAUROC(y_test, p_test, 1)
    print("Individual Diseases:",file = conf.log_writter)
    print(">> AUC = {}".format(np.array2string(np.array(aurocIndividual), precision=4, separator=',')),file = conf.log_writter)
    aurocMean = np.array(aurocIndividual).mean()
    print(">>Mean AUC = {:.4f}".format(aurocMean),file = conf.log_writter)
    conf.log_writter.flush()

def main(conf):

    train_dataset = build_classfication_dataset("ShenzhenCXR", conf.train_image_path_file, conf.input_size, conf.anno_percent, mode="train")
    valid_dataset = build_classfication_dataset("ShenzhenCXR", conf.valid_image_path_file, conf.input_size, anno_percent = 100, mode="validation")
    test_dataset = build_classfication_dataset("ShenzhenCXR", conf.test_image_path_file, conf.input_size, anno_percent = 100, mode="test")
    test_dataset_noTenC = build_classfication_dataset("ShenzhenCXR", conf.test_image_path_file, conf.input_size, anno_percent = 100, mode="validation")



    sampler_train = torch.utils.data.RandomSampler(train_dataset)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=conf.batch_size, num_workers=conf.num_workers,pin_memory=True,sampler=sampler_train)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=conf.batch_size, num_workers=conf.num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=conf.batch_size//2 + 1, num_workers=conf.num_workers, pin_memory=True)
    test_loader_noTenC = torch.utils.data.DataLoader(test_dataset_noTenC, batch_size=conf.batch_size//2 + 1, num_workers=conf.num_workers, pin_memory=True)
    print("train size: ", len(train_dataset), file=conf.log_writter)
    print("val size: ", len(valid_dataset), file=conf.log_writter)
    print("test size: ", len(test_dataset), file=conf.log_writter)
    if args.test_only:
        test(os.path.join(conf.model_path, 'ckpt.pth'), test_loader, conf)
        conf.log_writter.flush()

        exit(0)

    else:
        model, optimizer, loss_scaler, start_epoch = build_model(conf)

    lossMIN =10000
    patience_cnt =0

    for epoch in range(start_epoch, conf.epochs + 1):
        time1 = time.time()

        lr_ = step_decay(epoch,conf)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_
        train_loss = train(train_loader, model, optimizer , epoch,loss_scaler, conf)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1),file = conf.log_writter)
        print('training loss: {}@Epoch: {}'.format(train_loss,epoch),file = conf.log_writter)
        print('learning_rate: {},{}'.format(optimizer.param_groups[0]['lr'],epoch),file = conf.log_writter)

        valid_loss = evaluate(valid_loader,model, epoch)
        conf.log_writter.flush()
        print("Epoch {:04d}: val_loss:{:.5f} best loss {:.5f}".format(epoch, valid_loss,lossMIN),file=conf.log_writter)

        if valid_loss < lossMIN:
            save_file = os.path.join(conf.model_path, 'ckpt.pth')
            print("Epoch {:04d}: val_loss improved from {:.5f} to {:.5f}, saving model to {}".format(epoch, lossMIN, valid_loss, save_file),file = conf.log_writter)
            save_checkpoint({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'scaler': loss_scaler.state_dict(),
            }, filename=save_file)
            lossMIN = valid_loss
            patience_cnt = 0
            conf.log_writter.flush()
        else:
            patience_cnt += 1

        if patience_cnt > conf.patience or conf.debug_mode:
            break

    print("Testing:",file = conf.log_writter)
    test(save_file, test_loader,conf)



    conf.log_writter.flush()



if __name__ == '__main__':


    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    conf = config.ShenzhenRCXR(args)

    conf.display()
    main(conf)