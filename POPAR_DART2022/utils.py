import torch
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import interp
from sklearn.metrics import roc_curve, auc

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()

        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        #print(target, pred, batch_size, correct, res)
        return res

def save_model(model, optimizer, conf, epoch, save_file):
    print('==> Saving...',file=conf.log_writter)
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state

def unNormalize(tensor, mean=[0.449], std=[0.226]):
    """
    Args:
        tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
    Returns:
        Tensor: Normalized image.
    """
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
        # The normalize code -> t.sub_(m).div_(s)
    return tensor

def norm01(tensor):

    return (tensor - torch.min(tensor)) / (torch.max(tensor) - torch.min(tensor))



def draw_roc(fpr,tpr, roc_auc,name, n_classes = 8):
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    lw = 2

    color_dict = {
        0: ["darkorange","Four"],
        1: ["aqua", "Three"],
        2: ["cornflowerblue", "One"],
        3: ["deeppink", "Zero"]
    }
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], color=color_dict[i][0], lw=lw, label='ROC curve for '+ color_dict[i][1] +'(area = %0.2f)' % roc_auc[i])

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROR for Supervised Multi-lablel Classfication')
    plt.legend(loc="lower right")
    plt.savefig("{}.png".format(name))