#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 13:12:04 2022
convert instance labels to three class labels:
0: background
1: interior
2: boundary
@author: jma
"""

import os

join = os.path.join
import argparse
import cv2
from skimage import io, segmentation, morphology, exposure
from skimage.color import rgb2hsv, hsv2rgb
import numpy as np
import tifffile as tif
from tqdm import tqdm
from PIL import ImageFile
import json
import shutil
import math
from multiprocessing import Pool

# ImageFile.LOAD_TRUNCATED_IMAGES = True

parser = argparse.ArgumentParser('Preprocessing for microscopy image segmentation', add_help=False)
parser.add_argument('-i', '--input_path', default='/raid/zby/NeurISP2022-CellSeg/Train-Unlabeled', type=str,
                    help='training data path; subfolders: images, labels')
parser.add_argument("-o", '--output_path', default='/raid/zby/data/official_unlabeled_1', type=str,
                    help='preprocessing data path')
parser.add_argument("-f", '--info_path', default='/raid/zby/data/split_info.json', type=str,
                    help='preprocessing data path')
parser.add_argument("-m", '--mode', default='test', type=str, help='mode to preprocess data path')
args = parser.parse_args()

source_path = args.input_path
output_path = args.output_path
# split_info = json.load(open(args.info_path))

img_path = join(source_path, 'images')
# gt_path = join(source_path, 'labels')
img_names = sorted(os.listdir(img_path))
gt_names = [img_name.split('.')[0] + '_label.tiff' for img_name in img_names]

train_img_path = join(output_path, 'images')
# train_label_path = join(output_path, 'labels')
# train_gt_path = join(output_path, 'gts')
os.makedirs(train_img_path, exist_ok=True)


# os.makedirs(train_label_path, exist_ok=True)
# os.makedirs(train_gt_path, exist_ok=True)


def normalize_channel(img, lower=1, upper=99):
    non_zero_vals = img[np.nonzero(img)]
    percentiles = np.percentile(non_zero_vals, [lower, upper])
    if percentiles[1] - percentiles[0] > 0.001:
        img_norm = exposure.rescale_intensity(img, in_range=(percentiles[0], percentiles[1]), out_range='uint8')
    else:
        img_norm = img
    return img_norm.astype(np.uint8)


# def normalize_image(img, lower=1, upper=99):
#     hsv = rgb2hsv(img)
#     v = hsv[:, :, 2]
#     non_zero_vals = v[np.nonzero(v)]
#     percentiles = np.percentile(v, [lower, upper])
#     if percentiles[1] - percentiles[0] > 0.001:
#         v_norm = exposure.rescale_intensity(v, in_range=(percentiles[0], percentiles[1]), out_range='uint8')
#     else:
#         v_norm = v
#     hsv[:, :, 2] = v_norm
#     img_norm = hsv2rgb(hsv)
#     return img_norm.astype(np.uint8)




def create_interior_map(inst_map):
    """
    Parameters
    ----------
    inst_map : (H,W), np.int16
        DESCRIPTION.
    Returns
    -------
    interior : (H,W), np.uint8 
        three-class map, values: 0,1,2
        0: background
        1: interior
        2: boundary
    """
    # inst_num = np.max(inst_map)
    # seg_map = np.zeros(inst_map.shape)
    # for inst_idx in range(1, inst_num + 1):
    #     inst_mask = inst_map == inst_idx
    #     inst_mask = inst_mask * 1
    #     inst_size = np.sum(inst_mask)
    #     if inst_size <= 16:
    #         continue
    #     inst_boundary = segmentation.find_boundaries(inst_mask, mode='inner')
    #     if inst_size <= 4000:  # 2000
    #         radius = 1
    #     elif inst_size <= 20000:  # 10000
    #         radius = 1
    #     elif inst_size <= 100000:  # 50000
    #         radius = 2
    #     else:
    #         radius = 3
    #     inst_mask = morphology.binary_dilation(inst_mask, morphology.disk(radius))
    #     inst_boundary = morphology.binary_dilation(inst_boundary, morphology.disk(radius))
    #     seg_map[inst_mask] = 1
    #     seg_map[inst_boundary] = 2
    #     pass

    # # create interior-edge map
    boundary = segmentation.find_boundaries(inst_map, mode='inner')
    boundary = morphology.binary_dilation(boundary, morphology.disk(1))

    interior_temp = np.logical_and(~boundary, inst_map > 0)
    # interior_temp[boundary] = 0
    interior_temp = morphology.remove_small_objects(interior_temp, min_size=16)
    interior = np.zeros_like(inst_map, dtype=np.uint8)
    interior[interior_temp] = 1
    interior[boundary] = 2
    return interior


def preprocess(tt_list):
    # for (img_name, label_name) in tt_list:
    img_name, label_name = tt_list
    if os.path.exists(join(output_path, 'images_new', img_name.split('.')[0] + '.png')):
        return
    if img_name.endswith('.tif') or img_name.endswith('.tiff'):
        img_data = tif.imread(join(img_path, img_name))
    else:
        img_data = io.imread(join(img_path, img_name))

    # normalize image data
    # if len(img_data.shape) == 2:
    #     img_data = np.repeat(np.expand_dims(img_data, axis=-1), 3, axis=-1)
    # elif len(img_data.shape) == 3 and img_data.shape[-1] > 3:
    #     img_data = img_data[:,:, :3]
    # else:
    #     pass
    # # original normalize channel
    # pre_img_data = np.zeros(img_data.shape, dtype=np.uint8)
    # for i in range(3):
    #     img_channel_i = img_data[:,:,i]
    #     if len(img_channel_i[np.nonzero(img_channel_i)])>0:
    #         pre_img_data[:,:,i] = normalize_channel(img_channel_i, lower=1, upper=99)

    # Easy normalize
    # max_value = np.max(img_data)
    # img_data = (img_data/max_value)*255
    # pre_img_data = img_data.astype(np.uint8)

    # Gamma
    # pre_img_data = gamma_trans(pre_img_data, 1.0)

    pre_img_data = normalize_channel(img_data, lower=1, upper=99)

    # conver instance bask to three-class mask: interior, boundary
    # if args.mode == 'train':
    #     label_data = tif.imread(join(gt_path, label_name))
    #     interior_map = create_interior_map(label_data.astype(np.int16))
    # else:
    #     interior_map = np.zeros_like(pre_img_data)

    # if img_name.split('.')[0] in split_info['train'] + split_info['val']:
    os.makedirs(join(output_path, 'images_new'), exist_ok=True)
    io.imsave(join(output_path, 'images_new', img_name.split('.')[0] + '.png'), pre_img_data.astype(np.uint8),
              check_contrast=True)
    # io.imsave(join(output_path, 'labels', label_name.split('.')[0] + '.png'), interior_map.astype(np.uint8),
    #           check_contrast=False)
    # shutil.copyfile(join(gt_path, label_name), join(train_gt_path, label_name))
    # else:
    #     print("Image {} is not in the split!".format(img_name))
    # return 0
    # print(img_name)

#######################################################
# For the unlabeled images
# img_path = join(source_path, 'release-part1')
# img_names = sorted(os.listdir(img_path))
# pre_img_path = join(target_path, 'images')
# os.makedirs(pre_img_path, exist_ok=True)
# for img_name in tqdm(img_names):
#     if img_name.endswith('.tif') or img_name.endswith('.tiff'):
#         img_data = tif.imread(join(img_path, img_name))
#     else:
#         img_data = io.imread(join(img_path, img_name))

#     # normalize image data
#     if len(img_data.shape) == 2:
#         img_data = np.repeat(np.expand_dims(img_data, axis=-1), 3, axis=-1)
#     elif len(img_data.shape) == 3 and img_data.shape[-1] > 3:
#         img_data = img_data[:,:, :3]
#     else:
#         pass

#     pre_img_data = np.zeros(img_data.shape, dtype=np.uint8)
#     for i in range(3):
#         img_channel_i = img_data[:,:,i]
#         if len(img_channel_i[np.nonzero(img_channel_i)])>0:
#             pre_img_data[:,:,i] = normalize_channel(img_channel_i, lower=1, upper=99)

#     io.imsave(join(target_path, 'images_new', img_name.split('.')[0]+'.png'), pre_img_data.astype(np.uint8), check_contrast=False)


def gen_split_list():
    train_list = os.listdir('/home/zby/Cellseg/data/Train_Labeled_3class/labels')
    train_list = [train_name.replace('_label.png', '') for train_name in train_list]
    val_list = os.listdir('/home/zby/Cellseg/data/Val_Labeled_3class/labels')
    val_list = [val_name.replace('_label.png', '') for val_name in val_list]
    dataset_dict = {'train': train_list, 'val': val_list}
    with open('/home/zby/Cellseg/data/split_info.json', 'w') as f:
        json.dump(dataset_dict, f, indent=2)


tt_list = zip(img_names, gt_names)
# pool = Pool(processes=16)
# pool.imap_unordered(preprocess, tt_list)
# for i in tqdm(pool.map(preprocess, tt_list)):
#     pass
with tqdm(tt_list) as t, Pool() as p:
    for example in p.imap_unordered(preprocess, tt_list):
        t.update()
# if __name__ == "__main__":
#     # gen_split_list()
#     main()
