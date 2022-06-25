from smooth_loss import get_saliency_smoothness
import pytorch_iou
import torch.nn as nn
iou_loss = pytorch_iou.IOU(size_average=True)
bce_loss = nn.BCELoss(size_average=True)


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def muti_loss_fusion(s1, s2, s3, s4, s5, labels_v, aerf=1):
    loss1 = bce_loss(s1, labels_v)
    iou1 = iou_loss(s1, labels_v)
    sml1 = get_saliency_smoothness(s1, labels_v)
    loss2 = bce_loss(s2, labels_v)
    iou2 = iou_loss(s2, labels_v)
    sml2 = get_saliency_smoothness(s2, labels_v)
    loss3 = bce_loss(s3, labels_v)
    iou3 = iou_loss(s3, labels_v)
    sml3 = get_saliency_smoothness(s3, labels_v)
    loss4 = bce_loss(s4, labels_v)
    iou4 = iou_loss(s4, labels_v)
    sml4 = get_saliency_smoothness(s4, labels_v)
    loss5 = bce_loss(s5, labels_v)
    iou5 = iou_loss(s5, labels_v)
    sml5 = get_saliency_smoothness(s5, labels_v)


    loss = loss1 + aerf * iou1 + aerf * sml1 + loss2 + aerf * iou2 + aerf * sml2 \
           + loss3 + aerf * iou3 + aerf * sml3 + loss4 + aerf * iou4 + aerf * sml4 + loss5 + aerf * iou5 + aerf * sml5


    return loss

def adjust_learning_rate_poly(optimizer, epoch, num_epochs, base_lr, power):
    lr = base_lr * (1-epoch/num_epochs)**power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr