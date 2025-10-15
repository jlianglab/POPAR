from upernet import UperNet_swin, UperNet_swinv2
import torch
import argparse
import os
from segmentationConfig import Segmentation as config
from utils import AverageMeter, save_model, dice_score, mean_dice_coef, torch_dice_coef_loss, step_decay, load_swin_pretrained
import torch.backends.cudnn as cudnn
from torch import optim as optim
from timm.utils import NativeScaler, ModelEma
import numpy as np
import math
import sys
from segmentation_datasets import MontgomeryDataset, JSRTClavicleDataset, JSRTHeartDataset,JSRTLungDataset, VinDrRibCXRDataset
import time

import torch.nn.functional as F



parser = argparse.ArgumentParser('argument for training')
parser.add_argument('--batch_size', type=int, default=60,  help='batch_size')
parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
parser.add_argument('--epochs', type=int, default=350, help='number of training epochs')
parser.add_argument('--patience', type=int, default=50, help='number of training patience')
parser.add_argument('--gpu', dest='gpu', default="0,1", type=str, help="gpu index")
parser.add_argument('--dataset', dest='dataset', default="vindrribcxr", type=str)
parser.add_argument('--runs', dest='runs', default="output_runs", type=str)
parser.add_argument('--img_size', type=int, default=448, help='image size')
parser.add_argument('--anno_percent', type=int, default=100, help='percent of traning samples')
parser.add_argument('--num_classes', dest='num_classes', type=int,default=20)
parser.add_argument('--test_only', dest='test_only', action="store_true")
parser.add_argument('--ape', dest='ape', action="store_true")

parser.add_argument('--weight', dest='weight', default=None)
args = parser.parse_args()



def build_model(conf):
    start_epoch = 1

    if conf.weight == "popar_swin_allxrays_448":
        model = UperNet_swin(img_size=448, num_classes=conf.num_classes)
        checkpoint = torch.load("popar_swin_allxrays_448.pth", map_location='cpu')
        checkpoint = {k.replace("module.", ""): v for k, v in checkpoint.items()}
        checkpoint = {k.replace("backbone.", ""): v for k, v in checkpoint.items()}
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in checkpoint:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint[k]
        result = model.backbone.load_state_dict(checkpoint, strict=False)
        print(result, file=conf.log_writter)

    elif conf.weight == "popar_swin_nih14_448":
        model = UperNet_swin(img_size=448, num_classes=conf.num_classes)
        checkpoint = torch.load("popar_swin_nih14_448.pth", map_location='cpu')
        checkpoint = {k.replace("module.", ""): v for k, v in checkpoint.items()}
        checkpoint = {k.replace("backbone.", ""): v for k, v in checkpoint.items()}
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in checkpoint:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint[k]
        result = model.backbone.load_state_dict(checkpoint, strict=False)
        print(result, file=conf.log_writter)

    elif conf.weight == "popar_swinv2_allxrays_512":
        model = UperNet_swinv2(img_size=512, num_classes=1)
        checkpoint = torch.load("popar_swinv2_allxrays_512.pth", map_location='cpu')
        checkpoint = {k.replace("module.", ""): v for k, v in checkpoint.items()}
        checkpoint = {k.replace("backbone.", ""): v for k, v in checkpoint.items()}
        checkpoint = {k.replace("swin_model.", ""): v for k, v in checkpoint.items()}

        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in checkpoint:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint[k]
        result = model.backbone.load_state_dict(checkpoint, strict=False)

        print(result, file=conf.log_writter)
    elif conf.weight == "popar_swinv2_nih14_448":
        model = UperNet_swinv2(img_size=512, num_classes=1)
        checkpoint = torch.load("popar_swinv2_nih14_448.pth", map_location='cpu')

        checkpoint = {k.replace("module.", ""): v for k, v in checkpoint.items()}
        checkpoint = {k.replace("backbone.", ""): v for k, v in checkpoint.items()}
        checkpoint = {k.replace("swin_model.", ""): v for k, v in checkpoint.items()}

        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in checkpoint:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint[k]
        result = model.backbone.load_state_dict(checkpoint, strict=False)

        print(result, file=conf.log_writter)



    optimizer = optim.AdamW(model.parameters(), lr=conf.lr)
    loss_scaler = NativeScaler()

    model = model.cuda()
    model = torch.nn.DataParallel(model, device_ids=[i for i in range(torch.cuda.device_count())])

    cudnn.benchmark = True

    return model, optimizer,loss_scaler, start_epoch


def save_image(input,idx):
    from PIL import Image

    def disparity_normalization(disp):  # disp is an array in uint8 data type
        _min = np.amin(disp)
        _max = np.amax(disp)
        disp_norm = (disp - _min) * 255.0 / (_max - _min)
        return np.uint8(disp_norm)

    im = disparity_normalization(input)
    im = Image.fromarray(im)
    im.save("{}.jpeg".format(idx))


def train_one_epoch(model,train_loader, optimizer, loss_scaler, epoch ):
    model.train(True)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    criterion = torch_dice_coef_loss
    end = time.time()

    for idx, (img,mask) in enumerate(train_loader):
        data_time.update(time.time() - end)
        bsz = img.shape[0]




        img = img.float().cuda(non_blocking=True)
        mask = mask.float().cuda(non_blocking=True)

        outputs = torch.sigmoid(model(img))

        loss = criterion(mask, outputs)
        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), file=conf.log_writter)
            sys.exit(1)
            # update metric
        losses.update(loss.item(), bsz)

        optimizer.zero_grad()
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=None,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()




        if (idx + 1) % 10 == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'lr {lr}\t'
                  'Total loss {ttloss.val:.5f} ({ttloss.avg:.5f})'.format(
                epoch, idx + 1, len(train_loader), batch_time=batch_time,
                data_time=data_time, lr=optimizer.param_groups[0]['lr'], ttloss=losses), file=conf.log_writter)
            conf.log_writter.flush()
            if conf.debug_mode:
                break
    return losses.avg


def evaluation(model, val_loader, epoch):
    model.eval()
    losses = AverageMeter()
    criterion = torch_dice_coef_loss

    with torch.no_grad():
        for idx, (img, mask) in enumerate(val_loader):
            bsz = img.shape[0]

            img = img.float().cuda(non_blocking=True)
            mask = mask.float().cuda(non_blocking=True)

            outputs = torch.sigmoid(model(img))

            loss = criterion(mask, outputs)

            if not math.isfinite(loss.item()):
                print("Loss is {}, stopping training".format(loss.item()), file=conf.log_writter)
                sys.exit(1)
                # update metric
            losses.update(loss.item(), bsz)


            torch.cuda.synchronize()


            if (idx + 1) % 10 == 0:
                print('Evaluation: [{0}][{1}/{2}]\t'
                      'Total loss {ttloss.val:.5f} ({ttloss.avg:.5f})'.format(
                    epoch, idx + 1, len(val_loader), ttloss=losses), file=conf.log_writter)
                conf.log_writter.flush()
                if conf.debug_mode:
                    break
    return losses.avg

def test(test_loader, conf):
    if conf.weight == "popar_swin_allxrays_448":
        model = UperNet_swin(img_size=448, num_classes=conf.num_classes)
        checkpoint = torch.load("popar_swin_allxrays_448.pth", map_location='cpu')
        checkpoint = {k.replace("module.", ""): v for k, v in checkpoint.items()}
        checkpoint = {k.replace("backbone.", ""): v for k, v in checkpoint.items()}
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in checkpoint:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint[k]
        result = model.backbone.load_state_dict(checkpoint, strict=False)
        print(result, file=conf.log_writter)

    elif conf.weight == "popar_swin_nih14_448":
        model = UperNet_swin(img_size=448, num_classes=conf.num_classes)
        checkpoint = torch.load("popar_swin_nih14_448.pth", map_location='cpu')
        checkpoint = {k.replace("module.", ""): v for k, v in checkpoint.items()}
        checkpoint = {k.replace("backbone.", ""): v for k, v in checkpoint.items()}
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in checkpoint:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint[k]
        result = model.backbone.load_state_dict(checkpoint, strict=False)
        print(result, file=conf.log_writter)

    elif conf.weight == "popar_swinv2_allxrays_512":
        model = UperNet_swinv2(img_size=512, num_classes=1)
        checkpoint = torch.load("popar_swinv2_allxrays_512.pth", map_location='cpu')
        checkpoint = {k.replace("module.", ""): v for k, v in checkpoint.items()}
        checkpoint = {k.replace("backbone.", ""): v for k, v in checkpoint.items()}
        checkpoint = {k.replace("swin_model.", ""): v for k, v in checkpoint.items()}

        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in checkpoint:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint[k]
        result = model.backbone.load_state_dict(checkpoint, strict=False)

        print(result, file=conf.log_writter)
    elif conf.weight == "popar_swinv2_nih14_448":
        model = UperNet_swinv2(img_size=512, num_classes=1)
        checkpoint = torch.load("popar_swinv2_nih14_448.pth", map_location='cpu')

        checkpoint = {k.replace("module.", ""): v for k, v in checkpoint.items()}
        checkpoint = {k.replace("backbone.", ""): v for k, v in checkpoint.items()}
        checkpoint = {k.replace("swin_model.", ""): v for k, v in checkpoint.items()}

        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in checkpoint:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint[k]
        result = model.backbone.load_state_dict(checkpoint, strict=False)

        print(result, file=conf.log_writter)
    checkpoint = torch.load(os.path.join(conf.model_path, 'ckpt.pth'), map_location='cpu')
    checkpoint_model = {k.replace("module.", ""): v for k, v in checkpoint['model'].items()}
    message = model.load_state_dict(checkpoint_model)
    print(message, file=conf.log_writter)


    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model, device_ids=[i for i in range(torch.cuda.device_count())])
        model = model.cuda()
        cudnn.benchmark = True


    model.eval()
    with torch.no_grad():
        test_p = None
        test_y = None
        for idx, (img, mask) in enumerate(test_loader):
            bsz = img.shape[0]
            with torch.cuda.amp.autocast():
                img = img.float().cuda(non_blocking=True)
                outputs = torch.sigmoid(model(img))

                outputs = outputs.cpu().detach()
                mask = mask.cpu().detach()

                dice = 100.0 * dice_score(outputs, mask)
                mean_dice = mean_dice_coef(outputs > 0.5, mask > 0.5)

                # print(outputs.shape, mask.shape)
                # print("[INFO] Dice = {:.2f}%".format(100.0 * dice_score(outputs, mask)))
                # print("Mean Dice = {:.4f}".format(mean_dice_coef(outputs > 0.5, mask > 0.5)))


                # for b in range(bsz):
                #     save_image(img[b].cpu().numpy().transpose(1, 2, 0), conf.model_path + "/{}_{}_{}_input".format("test", idx, b))
                #
                #     save_image(np.sum(mask.cpu().numpy(), axis=1)[b], conf.model_path + "/{}_iter_{}_batch_{}_mask_{}_{}".format("test", idx, b, dice,mean_dice))
                #     save_image(np.sum(outputs.cpu().numpy(), axis=1)[b], conf.model_path + "/{}_iter_{}_batch_{}_pred_{}_{}".format("test", idx, b, dice, mean_dice))
                    # for i in range(conf.num_classes):
                    #     save_image(mask[b].cpu().numpy()[i],
                    #                conf.model_path + "/{}_iter_{}_batch_{}_mask_{}_{}_{}".format("test", idx, b, i, 100.0 * dice_score(outputs, mask), mean_dice_coef(outputs > 0.5, mask > 0.5)))
                    #     save_image(outputs[b].cpu().numpy()[i],
                    #                conf.model_path + "/{}_iter_{}_batch_{}_pred_{}_{}_{}".format("test", idx, b, i, 100.0 * dice_score(outputs, mask), mean_dice_coef(outputs > 0.5, mask > 0.5)))

                # if conf.dataset =="vindrribcxr":
                #     for b in range(bsz):
                #         save_image(img[b].cpu().numpy().transpose(1, 2, 0), conf.model_path+"/{}_{}_input".format("test", b))
                #         for i in range(conf.num_classes):
                #             save_image(mask[b].cpu().numpy()[i],
                #                        conf.model_path + "/{}_iter_{}_batch_{}_mask_{}".format("test",idx, b,i))
                #             save_image(outputs[b].cpu().numpy()[i],
                #                        conf.model_path + "/{}_iter_{}_batch_{}_pred_{}".format("test",idx, b,i))
                #
                # else:
                #     for b in range(bsz):
                #         save_image(img[b].cpu().numpy().transpose(1, 2, 0), conf.model_path+"/{}_{}_input".format("test", b))
                #         save_image(mask[b].cpu().squeeze(0).numpy(), conf.model_path+"/{}_{}_mask".format("test", b))
                #         save_image(outputs[b].cpu().squeeze(0).numpy(), conf.model_path+"/{}_{}_pred".format("test", b))
                #

                if test_p is None and test_y is None:
                    test_p = outputs
                    test_y = mask
                else:
                    test_p = torch.cat((test_p, outputs), 0)
                    test_y = torch.cat((test_y, mask), 0)
                torch.cuda.empty_cache()
                if (idx + 1) % 20 == 0:
                    print("Testing Step[{}/{}] ".format(idx + 1, len(test_loader)), file=conf.log_writter)
                    conf.log_writter.flush()
                    if conf.debug_mode:
                        break



        print("Done testing iteration!", file=conf.log_writter)
        conf.log_writter.flush()

    test_p = test_p.numpy()
    test_y = test_y.numpy()
    test_y = test_y.reshape(test_p.shape)
    return test_y, test_p




def main(conf):
    train_dataset = MontgomeryDataset(conf.train_image_path_file, image_size=(conf.img_size, conf.img_size))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=conf.batch_size,
                                               num_workers=conf.num_workers, pin_memory=True, shuffle=True,
                                               drop_last=True)

    val_dataset = MontgomeryDataset(conf.val_image_path_file, image_size=(conf.img_size, conf.img_size), mode="val")
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=conf.batch_size, num_workers=conf.num_workers,
                                             pin_memory=True, shuffle=True, drop_last=False)

    test_dataset = MontgomeryDataset(conf.test_image_path_file, image_size=(conf.img_size, conf.img_size), mode="val")

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=conf.batch_size,
                                              num_workers=conf.num_workers, pin_memory=True, drop_last=False)
    print("train size: ", len(train_dataset), file=conf.log_writter)
    print("val size: ", len(val_dataset), file=conf.log_writter)
    print("test size: ", len(test_dataset), file=conf.log_writter)


    if args.test_only:
        test_y, test_p = test(test_loader, conf)
        print("[INFO] Dice = {:.2f}%".format(100.0 * dice_score(test_p, test_y)), file=conf.log_writter)
        print("Mean Dice = {:.4f}".format(mean_dice_coef(test_y > 0.5, test_p > 0.5)), file=conf.log_writter)

        conf.log_writter.flush()

        exit(0)

    else:
        model, optimizer,loss_scaler, start_epoch = build_model(conf)

    best_val_loss = 100000
    patience_counter = 0


    for epoch in range(start_epoch, conf.epochs):

        lr_ = step_decay(epoch, conf)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_
        print('learning_rate: {},{}'.format(optimizer.param_groups[0]['lr'], epoch), file=conf.log_writter)

        loss_avg = train_one_epoch(model,train_loader, optimizer, loss_scaler, epoch)
        print('Training loss: {}@Epoch: {}'.format(loss_avg, epoch), file=conf.log_writter)
        conf.log_writter.flush()


        val_avg = evaluation(model,val_loader,epoch)


        if val_avg < best_val_loss:
            save_file = os.path.join(conf.model_path, 'ckpt.pth')
            save_model(model, optimizer, conf, epoch+1, save_file)


            print( "Epoch {:04d}: val_loss improved from {:.5f} to {:.5f}, saving model to {}".format(epoch, best_val_loss,val_avg, save_file), file=conf.log_writter)
            best_val_loss = val_avg
            patience_counter = 0
        else:
            print("Epoch {:04d}: val_loss did not improve from {:.5f} ".format(epoch, best_val_loss), file=conf.log_writter)
            patience_counter += 1
        if patience_counter > conf.patience:
            print("Early Stopping", file=conf.log_writter)
            break

        conf.log_writter.flush()
        if conf.debug_mode:
            break


    test_y, test_p = test(test_loader, conf)

    print("[INFO] Dice = {:.2f}%".format(100.0 * dice_score(test_p, test_y)), file=conf.log_writter)
    print("Mean Dice = {:.4f}".format(mean_dice_coef(test_y > 0.5, test_p > 0.5)), file=conf.log_writter)
    conf.log_writter.flush()


if __name__ == '__main__':


    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    conf = config(args)
    conf.display()
    main(conf)