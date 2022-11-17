# -*- coding: utf-8 -*-
"""
# @file name  : cutmix.py
# @author     : DJH
# @date       : 2022/9/24 11:17
# @brief      :
"""
import numpy as np
import torch


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def CutMix(args, img, anno, heatmap, boundary):
    '''

    :param args:
    :param img: BCHW
    :param anno: BHW
    :param heatmap: BHW
    :param boundary: BHW
    :return:
    '''
    r = np.random.rand(1)
    if args.beta > 0 and r < args.cutmix_prob:  # cutmix
        # generate mixed sample
        lam = np.random.beta(args.beta, args.beta)
        rand_index = torch.randperm(img.size()[0]).cuda()
        bbx1, bby1, bbx2, bby2 = rand_bbox(img.size(), lam)
        print(bbx1, bby1, bbx2, bby2)
        img[:, :, bbx1:bbx2, bby1:bby2] = img[rand_index, :, bbx1:bbx2, bby1:bby2]
        anno[:, bbx1:bbx2, bby1:bby2] = anno[rand_index, bbx1:bbx2, bby1:bby2]
        heatmap[:, bbx1:bbx2, bby1:bby2] = heatmap[rand_index, bbx1:bbx2, bby1:bby2]
        boundary[:, bbx1:bbx2, bby1:bby2] = boundary[rand_index, bbx1:bbx2, bby1:bby2]
        return img, anno, heatmap, boundary

    else:
        return img, anno, heatmap, boundary


def CutMix_unlabeled(args, un_pred_T, un_pred_T_pseudo_label, unlabeled_img_s):

    r = np.random.rand(1)
    if args.beta > 0 and r < args.cutmix_prob:
        # generate mixed sample
        lam = np.random.beta(args.beta, args.beta)
        rand_index = torch.randperm(un_pred_T.size()[0]).cuda()
        bbx1, bby1, bbx2, bby2 = rand_bbox(unlabeled_img_s.size(), lam)
        un_pred_T[:, :, bbx1:bbx2, bby1:bby2] = un_pred_T[rand_index, :, bbx1:bbx2, bby1:bby2]
        un_pred_T_pseudo_label[:, bbx1:bbx2, bby1:bby2] = un_pred_T_pseudo_label[rand_index, bbx1:bbx2, bby1:bby2]
        unlabeled_img_s[:, :, bbx1:bbx2, bby1:bby2] = unlabeled_img_s[rand_index, :, bbx1:bbx2, bby1:bby2]
        return un_pred_T, un_pred_T_pseudo_label, unlabeled_img_s

    else:
        return un_pred_T, un_pred_T_pseudo_label, unlabeled_img_s
