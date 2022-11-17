import os
import cv2
import numpy as np
import tifffile as tif
from skimage import measure, color, morphology, segmentation


def postprocess(img_dir, heat_dir, output_dir):
    threshold = 0.5
    high_threshold = 0.7
    heat_list = sorted(os.listdir(heat_dir))
    heat_paths = [os.path.join(heat_dir, heat_name) for heat_name in heat_list]
    for ii, heat_path in enumerate(heat_paths):
        print("Processing the {}/{} images....".format(ii, len(heat_paths)), end='\r')
        img_path = heat_path.replace(heat_dir, img_dir)
        heatmap = cv2.imread(heat_path, flags=0) / 255
        H, W = heatmap.shape
        heatmap_temp = (heatmap >= threshold) * 1
        heatmap_temp = heatmap_temp.astype('uint8')
        kernel = np.ones((3, 3), np.uint8)
        heatmap_temp = cv2.morphologyEx(heatmap_temp, cv2.MORPH_ERODE, kernel=kernel, iterations=1)
        nLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(heatmap_temp.astype(np.int8),
                                                                             connectivity=4)

        new_labels = np.zeros((H, W))
        for i in range(1, nLabels):
            size = stats[i, cv2.CC_STAT_AREA]
            if size < 25:
                continue
            if np.max(heatmap[labels == i]) < high_threshold:
                continue
            # make segmentation map
            new_labels[labels == i] = i
        new_labels = cv2.dilate(heatmap_temp.astype(np.uint16), kernel, iterations=1)
        tif.imwrite(os.path.join(output_dir, heat_path.replace(heat_dir, output_dir).replace('.png', '_label.tiff')),
                    new_labels, compression='zlib')
    pass


def postprocess_cls(img_dir, heat_dir, output_dir):
    # threshold = 0.3
    heat_list = sorted(os.listdir(heat_dir))
    heat_paths = [os.path.join(heat_dir, heat_name) for heat_name in heat_list]
    for ii, heat_path in enumerate(heat_paths):
        print("Processing the {}/{} images....".format(ii, len(heat_paths)), end='\r')
        img_path = heat_path.replace(heat_dir, img_dir)
        heatmap = cv2.imread(heat_path, flags=0)
        heatmap[heatmap == 128] = 1
        heatmap[heatmap == 255] = 0
        kernel = np.ones((3, 3), np.uint8)
        heatmap = cv2.morphologyEx(heatmap, cv2.MORPH_OPEN, kernel, iterations=1)
        ret, heatmap = cv2.connectedComponents(heatmap.astype(np.int8))
        new_inst_map = cv2.dilate(heatmap.astype(np.uint16), kernel, iterations=1)
        tif.imwrite(os.path.join(output_dir, heat_path.replace(heat_dir, output_dir).replace('.png', '_label.tiff')),
                    new_inst_map, compression='zlib')
    pass


if __name__ == "__main__":
    img_dir = '/home/zby/Cellseg/data/fix_boundary/images_new'
    heat_dir = '/home/zby/Cellseg/workspace/extend_unet50_ep500_b16_crp512_reg_boundary_dice_cutmix_largescalerange/results_test_multi_max/heat_pred'
    output_dir = heat_dir.replace('heat_pred', 'postprocessed_reg_v4')
    os.makedirs(output_dir, exist_ok=True)
    postprocess(img_dir, heat_dir, output_dir)

    # img_dir = '/home/zby/Cellseg/data/fix_boundary/images_new'
    # heat_dir = '/home/zby/Cellseg/workspace/extend_beit_ep500_b18_crp512_reg_boundary/results_test/pred'
    # output_dir = heat_dir.replace('pred', 'postprocessed_cls_v2')
    # os.makedirs(output_dir,exist_ok=True)
    # postprocess_cls(img_dir, heat_dir, output_dir)
