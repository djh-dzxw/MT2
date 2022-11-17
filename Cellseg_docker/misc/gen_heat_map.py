import os
import numpy as np
import cv2
import tifffile as tif
from skimage import io, segmentation, morphology, exposure
from distancemap import distance_map_from_binary_matrix
import multiprocessing


def create_heat_map(inst_map):
    inst_num = np.max(inst_map)
    seg_map = np.zeros(inst_map.shape).astype(np.float32)  # H W
    for inst_idx in range(1, inst_num + 1):
        inst_mask = inst_map == inst_idx
        inst_mask = inst_mask * 1
        if np.sum(inst_mask) <= 16:
            continue
        inst_boundary = segmentation.find_boundaries(inst_mask, mode='inner')

        temp_map = np.zeros(inst_map.shape)
        temp_map[inst_boundary] = 1
        temp_map = distance_map_from_binary_matrix(temp_map)
        temp_map = temp_map * inst_mask
        max_range = np.max(temp_map)
        radius = int(max_range / 10)
        radius = radius if radius % 2 == 1 else radius + 1
        temp_map[temp_map < radius] = 0
        temp_map[temp_map >= radius] = 1
        temp_map = temp_map.astype(np.float32)
        sigma = 0.3 * ((radius - 1) * 0.5 - 1) + 0.8
        temp_map = cv2.GaussianBlur(temp_map, (radius, radius), sigmaX=sigma)
        temp_map = np.clip(temp_map, 0, 1)
        # temp_map = temp_map/np.max(temp_map)
        seg_map += temp_map
    return seg_map


def gen_heat_map(gt_paths, output_dir):
    # gt_names = os.listdir(gt_dir)
    # gt_paths = [os.path.join(gt_dir,gt_name) for gt_name in gt_names]
    ii = 0
    for gt_path in gt_paths:
        ii += 1
        print("Generating the {}/{} dist_maps...".format(ii, len(gt_paths)), end='\r')
        label = tif.imread(gt_path)
        seg_map = create_heat_map(label) * 255
        seg_map = seg_map.astype(np.uint8)
        save_path = gt_path.replace(gt_dir, output_dir).replace('.tiff', '.png')
        cv2.imwrite(save_path, seg_map)
    pass


def create_boundary_heat_map(inst_map):
    inst_num = np.max(inst_map)
    seg_map = np.zeros(inst_map.shape).astype(np.float32)
    for inst_idx in range(1, inst_num + 1):
        inst_mask = inst_map == inst_idx
        inst_mask = inst_mask * 1
        inst_boundary = segmentation.find_boundaries(inst_mask, mode='inner')

        temp_map = np.zeros(inst_map.shape)
        temp_map[inst_boundary] = 1
        # Get Radius
        dist_map = distance_map_from_binary_matrix(temp_map)
        dist_map = dist_map * inst_mask
        max_range = np.max(dist_map)
        radius = int(max_range / 10)
        radius = radius if radius % 2 == 1 else radius + 1
        # GaussianBlur the boundary
        sigma = 0.3 * ((radius - 1) * 0.5 - 1) + 0.8
        temp_map = cv2.GaussianBlur(temp_map, (radius, radius), sigmaX=sigma)
        temp_map = temp_map / np.max(temp_map)
        # temp_map = temp_map/np.max(temp_map)
        seg_map += temp_map
    return seg_map


def gen_boundary_heat_map(gt_paths, output_dir):
    ii = 0
    for gt_path in gt_paths:
        ii += 1
        print("Generating the {}/{} dist_maps...".format(ii, len(gt_paths)), end='\r')
        label = tif.imread(gt_path)
        seg_map = create_boundary_heat_map(label) * 255
        seg_map = seg_map.astype(np.uint8)
        save_path = gt_path.replace(gt_dir, output_dir).replace('.tiff', '.png')
        seg_map = np.clip(seg_map, 0, 255)
        cv2.imwrite(save_path, seg_map)
    pass


if __name__ == "__main__":
    process_num = 10
    gt_dir = '/raid/zby/data/official/gts'
    output_dir = '/raid/zby/data/official/boundary_maps'
    os.makedirs(output_dir, exist_ok=True)
    gt_names = os.listdir(gt_dir)
    gt_paths = [os.path.join(gt_dir, gt_name) for gt_name in gt_names]
    databins = [[] for i in range(process_num)]
    for ii, gt_path in enumerate(gt_paths):
        bin_num = ii % process_num
        databins[bin_num].append(gt_path)
    print(len(databins[0]))
    pool = multiprocessing.Pool()
    for i in range(process_num):
        # pool.apply_async(gen_heat_map, (databins[i], output_dir))
        pool.apply_async(gen_boundary_heat_map,(databins[i], output_dir))
    pool.close()
    pool.join()
    # gen_heat_map(databins[0], output_dir)
