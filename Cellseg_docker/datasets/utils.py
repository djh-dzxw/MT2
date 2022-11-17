import cv2
from cv2 import ROTATE_90_CLOCKWISE
from cv2 import ROTATE_180
from cv2 import ROTATE_90_COUNTERCLOCKWISE
import numpy as np
import random


def random_crop(img, anno, heatmap, boundary, crop_size=(256, 256)):
    H, W, C = img.shape
    H_range = H - crop_size[0]
    W_range = W - crop_size[1]
    h_start = random.randint(0, H_range)
    w_start = random.randint(0, W_range)
    img = img[h_start:h_start + crop_size[0], w_start:w_start + crop_size[1], :]
    anno = anno[h_start:h_start + crop_size[0], w_start:w_start + crop_size[1]]
    heatmap = heatmap[h_start:h_start + crop_size[0], w_start:w_start + crop_size[1]]
    boundary = boundary[h_start:h_start + crop_size[0], w_start:w_start + crop_size[1]]
    return img, anno, heatmap, boundary


def random_scale(img, anno, heatmap, boundary, scale_range=(0.5, 2.0), crop_size=(256, 256)):
    H, W, C = img.shape
    # gen_scale_ratio
    scale = random.random()
    scale = scale * (scale_range[1] - scale_range[0]) + scale_range[0]
    # calculate the min scale_ratio
    min_h_scale = crop_size[0] / H
    min_w_scale = crop_size[1] / W
    scale = max(scale, min_h_scale, min_w_scale)

    h_dest = round(H * scale)
    w_dest = round(W * scale)
    img = cv2.resize(img, (w_dest, h_dest), interpolation=cv2.INTER_LINEAR)
    anno = cv2.resize(anno, (w_dest, h_dest), interpolation=cv2.INTER_NEAREST)
    heatmap = cv2.resize(heatmap, (w_dest, h_dest), interpolation=cv2.INTER_LINEAR)
    boundary = cv2.resize(boundary, (w_dest, h_dest), interpolation=cv2.INTER_LINEAR)
    return img, anno, heatmap, boundary


def random_crop_v2(img, anno, heatmap, boundary, crop_size=(256, 256)):
    H, W, C = img.shape
    H_range = H - crop_size[0]
    W_range = W - crop_size[1]
    h_start = random.randint(0, H_range)
    w_start = random.randint(0, W_range)
    img = img[h_start:h_start + crop_size[0], w_start:w_start + crop_size[1], :]
    anno = anno[h_start:h_start + crop_size[0], w_start:w_start + crop_size[1]]
    heatmap = heatmap[h_start:h_start + crop_size[0], w_start:w_start + crop_size[1]]
    boundary = boundary[h_start:h_start + crop_size[0], w_start:w_start + crop_size[1]]
    return img, anno, heatmap, boundary


def random_scale_v2(img, anno, heatmap, boundary, scale_range=(0.5, 2.0), crop_size=(256, 256)):
    H, W, C = img.shape
    # gen_scale_ratio
    scale = random.random()
    scale = scale * (scale_range[1] - scale_range[0]) + scale_range[0]
    # calculate the min scale_ratio
    h_dest = round(H * scale)
    w_dest = round(W * scale)
    img_temp = cv2.resize(img, (w_dest, h_dest), interpolation=cv2.INTER_LINEAR)
    anno_temp = cv2.resize(anno, (w_dest, h_dest), interpolation=cv2.INTER_NEAREST)
    heatmap_temp = cv2.resize(heatmap, (w_dest, h_dest), interpolation=cv2.INTER_LINEAR)
    boundary_temp = cv2.resize(boundary, (w_dest, h_dest), interpolation=cv2.INTER_LINEAR)
    h_temp, w_temp, _ = img_temp.shape
    h_pad = max(0, crop_size[0] - h_temp)
    w_pad = max(0, crop_size[1] - w_temp)
    img = cv2.copyMakeBorder(img_temp, 0, h_pad, 0, w_pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    anno = cv2.copyMakeBorder(anno_temp, 0, h_pad, 0, w_pad, cv2.BORDER_CONSTANT, value=0)
    heatmap = cv2.copyMakeBorder(heatmap_temp, 0, h_pad, 0, w_pad, cv2.BORDER_CONSTANT, value=0)
    boundary = cv2.copyMakeBorder(boundary_temp, 0, h_pad, 0, w_pad, cv2.BORDER_CONSTANT, value=0)

    return img, anno, heatmap, boundary


def random_flip(img, anno, heatmap, boundary, flip_ratio=0.5):
    h_flip = random.random()
    w_flip = random.random()
    if h_flip < flip_ratio:
        img = cv2.flip(img, 1)
        anno = cv2.flip(anno, 1)
        heatmap = cv2.flip(heatmap, 1)
        boundary = cv2.flip(boundary, 1)
    if w_flip < flip_ratio:
        img = cv2.flip(img, 0)
        anno = cv2.flip(anno, 0)
        heatmap = cv2.flip(heatmap, 0)
        boundary = cv2.flip(boundary, 0)

    return img, anno, heatmap, boundary


def slide_window(img, anno, heatmap, boundary, window_size):
    H, W, C = img.shape
    H_stride = round(window_size[0] * 3 / 4)
    W_stride = round(window_size[1] * 3 / 4)
    H_num = int((H - window_size[0]) / H_stride) + 1
    W_num = int((W - window_size[1]) / W_stride) + 1

    imgs = []
    annos = []
    heatmaps = []
    boundaries = []
    pos = []
    for h_num in range(H_num):
        for w_num in range(W_num):
            if h_num == H_num - 1:
                h_start = H - window_size[0]
            else:
                h_start = h_num * H_stride
            if w_num == W_num - 1:
                w_start = W - window_size[1]
            else:
                w_start = w_num * W_stride

            crop_img = img[h_start:h_start + window_size[0], w_start:w_start + window_size[1], :]
            crop_anno = anno[h_start:h_start + window_size[0], w_start:w_start + window_size[1]]
            crop_heatmap = heatmap[h_start:h_start + window_size[0], w_start:w_start + window_size[1]]
            crop_boundary = boundary[h_start:h_start + window_size[0], w_start:w_start + window_size[1]]
            imgs.append(crop_img)
            annos.append(crop_anno)
            heatmaps.append(heatmap)
            boundaries.append(boundary)
            pos.append([h_start, w_start])

    return imgs, annos, heatmaps, boundaries, pos


def rescale_to_min(img, anno, heatmap, boundary, min_size):
    H, W, C = img.shape
    min_H, min_W = min_size
    if H > min_H and W > min_W:
        return img, anno, heatmap, boundary
    else:
        ratio_h = min_H / H
        ratio_w = min_W / W
        ratio = max(ratio_h, ratio_w)
        img = cv2.resize(img, (int(W * ratio), int(H * ratio)), interpolation=cv2.INTER_LINEAR)
        anno = cv2.resize(anno, (int(W * ratio), int(H * ratio)), interpolation=cv2.INTER_NEAREST)
        heatmap = cv2.resize(heatmap, (int(W * ratio), int(H * ratio)), interpolation=cv2.INTER_LINEAR)
        boundary = cv2.resize(boundary, (int(W * ratio), int(H * ratio)), interpolation=cv2.INTER_LINEAR)
        return img, anno, heatmap, boundary


def random_rotate(img, anno, heatmap, boundary):
    rot_id = random.randint(0, 3)
    if rot_id == 0:
        pass
    elif rot_id == 1:
        img = cv2.rotate(img, ROTATE_90_CLOCKWISE)
        anno = cv2.rotate(anno, ROTATE_90_CLOCKWISE)
        heatmap = cv2.rotate(heatmap, ROTATE_90_CLOCKWISE)
        boundary = cv2.rotate(boundary, ROTATE_90_CLOCKWISE)
    elif rot_id == 2:
        img = cv2.rotate(img, ROTATE_180)
        anno = cv2.rotate(anno, ROTATE_180)
        heatmap = cv2.rotate(heatmap, ROTATE_180)
        boundary = cv2.rotate(boundary, ROTATE_180)
    elif rot_id == 3:
        img = cv2.rotate(img, ROTATE_90_COUNTERCLOCKWISE)
        anno = cv2.rotate(anno, ROTATE_90_COUNTERCLOCKWISE)
        heatmap = cv2.rotate(heatmap, ROTATE_90_COUNTERCLOCKWISE)
        boundary = cv2.rotate(boundary, ROTATE_90_COUNTERCLOCKWISE)
    return img, anno, heatmap, boundary


def multi_scale_test(img, anno, heatmap, boundary, scale=[1.0, 1.1, 1.2]):
    img_list = []
    anno_list = []
    heatmap_list = []
    boundary_list = []
    H, W, C = img.shape
    for s in scale:
        img_list.append(cv2.resize(img, (int(W * s), int(H * s)), interpolation=cv2.INTER_LINEAR))
        anno_list.append(cv2.resize(anno, (int(W * s), int(H * s)), interpolation=cv2.INTER_NEAREST))
        heatmap_list.append(cv2.resize(heatmap, (int(W * s), int(H * s)), interpolation=cv2.INTER_LINEAR))
        boundary_list.append(cv2.resize(boundary, (int(W * s), int(H * s)), interpolation=cv2.INTER_LINEAR))
    return img_list, anno_list, heatmap_list, boundary_list


def multi_scale_test_v2(img, scale=[0.5, 1.0, 1.5], crop_size=(512, 512)):
    img_list = []
    valid_list = []
    H, W, C = img.shape
    for s in scale:
        img_temp = cv2.resize(img, (int(W * s), int(H * s)), interpolation=cv2.INTER_LINEAR)
        h_temp, w_temp, _ = img_temp.shape
        h_pad = max(0, crop_size[0] - h_temp)
        w_pad = max(0, crop_size[1] - w_temp)
        valid_region = (h_temp, w_temp)
        img_temp = cv2.copyMakeBorder(img_temp, 0, h_pad, 0, w_pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        img_list.append(img_temp)
        valid_list.append(valid_region)
    return img_list, valid_list
