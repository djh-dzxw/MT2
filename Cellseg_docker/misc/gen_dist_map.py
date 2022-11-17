import os
import numpy as np
import cv2
import tifffile as tif
from skimage import io, segmentation, morphology, exposure
from distancemap import distance_map_from_binary_matrix


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
    inst_num = np.max(inst_map)
    seg_map = np.zeros(inst_map.shape)
    for inst_idx in range(1, inst_num + 1):
        inst_mask = inst_map == inst_idx
        inst_mask = inst_mask * 1
        inst_size = np.sum(inst_mask)
        if inst_size <= 16:
            continue
        inst_boundary = segmentation.find_boundaries(inst_mask, mode='inner')
        radius = 1
        inst_mask = morphology.binary_dilation(inst_mask, morphology.disk(radius))
        inst_boundary = morphology.binary_dilation(inst_boundary, morphology.disk(radius))
        temp_map = np.zeros(inst_map.shape)
        temp_map[inst_boundary] = 1
        temp_map = distance_map_from_binary_matrix(temp_map)
        temp_map = temp_map * inst_mask
        seg_map[inst_mask] = temp_map[inst_mask]
    return seg_map


def gen_dist_map(gt_dir, output_dir):
    gt_names = os.listdir(gt_dir)
    gt_paths = [os.path.join(gt_dir, gt_name) for gt_name in gt_names]
    ii = 0
    for gt_path in gt_paths:
        ii += 1
        print("Generating the {}/{} dist_maps...".format(ii, len(gt_paths)), end='\r')
        label = tif.imread(gt_path)
        seg_map = create_interior_map(label)
        save_path = gt_path.replace(gt_dir, output_dir).replace('.tiff', '.png')
        cv2.imwrite(save_path, seg_map)
    pass


def create_heat_map(inst_map):
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
    inst_num = np.max(inst_map)
    seg_map = np.zeros(inst_map.shape)
    for inst_idx in range(1, inst_num + 1):
        inst_mask = inst_map == inst_idx
        inst_mask = inst_mask * 1
        inst_size = np.sum(inst_mask)
        if inst_size <= 16:
            continue
        inst_boundary = segmentation.find_boundaries(inst_mask, mode='inner')
        radius = 1
        inst_mask = morphology.binary_dilation(inst_mask, morphology.disk(radius))
        inst_boundary = morphology.binary_dilation(inst_boundary, morphology.disk(radius))
        temp_map = np.zeros(inst_map.shape)
        temp_map[inst_boundary] = 1
        temp_map = distance_map_from_binary_matrix(temp_map)
        temp_map = temp_map * inst_mask
        temp_map = (temp_map / np.max(temp_map)) * 255
        seg_map[inst_mask] = temp_map[inst_mask]
    return seg_map


def gen_heat_map(gt_dir, output_dir):
    gt_names = os.listdir(gt_dir)
    gt_paths = [os.path.join(gt_dir, gt_name) for gt_name in gt_names]
    ii = 0
    for gt_path in gt_paths:
        ii += 1
        print("Generating the {}/{} dist_maps...".format(ii, len(gt_paths)), end='\r')
        label = tif.imread(gt_path)
        seg_map = create_heat_map(label)
        save_path = gt_path.replace(gt_dir, output_dir).replace('.tiff', '.png')
        cv2.imwrite(save_path, seg_map)
    pass


if __name__ == "__main__":
    gt_dir = '/home/zby/Cellseg/data/fix_boundary/gts'
    output_dir = '/home/zby/Cellseg/data/fix_boundary/heatmaps'
    # gen_dist_map(gt_dir, output_dir)
    gen_heat_map(gt_dir, output_dir)
