import os
import cv2
import numpy as np
import tifffile as tif
from skimage import measure, color, morphology, segmentation


def postprocess(img_dir, heat_dir, output_dir):
    low_threshold = 0.5
    high_threshold = 0.7
    heat_list = sorted(os.listdir(heat_dir))
    heat_paths = [os.path.join(heat_dir, heat_name) for heat_name in heat_list]
    for ii, heat_path in enumerate(heat_paths):
        print("Processing the {}/{} images....".format(ii, len(heat_paths)), end='\r')
        img_path = heat_path.replace(heat_dir, img_dir)
        img = cv2.imread(img_path)
        heatmap_ori = cv2.imread(heat_path, flags=0)
        heatmap = heatmap_ori / 255
        heatmap[heatmap >= high_threshold] = 255
        heatmap[heatmap < high_threshold] = 0
        heatmap = heatmap.astype(np.uint8)
        unknown = np.zeros(heatmap.shape)
        unknown[heatmap >= low_threshold] = 1
        unknown[heatmap >= high_threshold] = 0

        ret, markers = cv2.connectedComponents(heatmap)
        markers = markers + 1
        markers[unknown == 1] = 0
        markers = cv2.watershed(img, markers)

        # Add it if needed
        # kernel = np.ones((3,3),np.uint8)
        # markers = cv2.morphologyEx(markers, cv2.MORPH_OPEN, kernel, iterations = 1)

        # cv2.imwrite(os.path.join(heat_path.replace(heat_dir, output_dir).replace('.png','_label.png')), markers)
        tif.imwrite(os.path.join(heat_path.replace(heat_dir, output_dir).replace('.png', '_label.tiff')), markers,
                    compression='zlib')
    pass


def postprocess_cls(img_dir, heat_dir, output_dir):
    threshold = 0.3
    heat_list = sorted(os.listdir(heat_dir))
    heat_paths = [os.path.join(heat_dir, heat_name) for heat_name in heat_list]
    for ii, heat_path in enumerate(heat_paths):
        print("Processing the {}/{} images....".format(ii, len(heat_paths)), end='\r')
        img_path = heat_path.replace(heat_dir, img_dir)
        heatmap = cv2.imread(heat_path, flags=0)
        heatmap[heatmap == 128] = 1
        heatmap[heatmap == 255] = 0
        heatmap = measure.label(heatmap, background=0)
        H, W = heatmap.shape
        if H > 5000 or W > 5000:
            new_inst_map = heatmap
        else:
            inst_num = np.max(heatmap)
            new_inst_map = np.zeros_like(heatmap)
            new_inst_num = 0
            for inst_idx in range(1, inst_num):
                print("Cls: Processing the {}/{} images... {}/{} instances...".format(ii, len(heat_paths), inst_idx,
                                                                                      inst_num), end='\r')
                if np.sum(heatmap == inst_idx) < 16:
                    continue
                new_inst_num += 1
                temp_map = heatmap == inst_idx
                temp_map = temp_map * 1
                temp_map = morphology.binary_dilation(temp_map, morphology.disk(1)) * 1
                new_inst_map[temp_map == 1] = new_inst_num
        tif.imwrite(os.path.join(output_dir, heat_path.replace(heat_dir, output_dir).replace('.png', '_label.tiff')),
                    new_inst_map, compression='zlib')
    pass


if __name__ == "__main__":
    img_dir = './data/Val_1_3class/images_new'
    heat_dir = './workspace/extend_split_unet50_ep500_b8_crp512_reg_boundary/heat_pred_new_1Channel'
    output_dir = heat_dir.replace('heat_pred', 'postprocessed_reg_v3')
    os.makedirs(output_dir, exist_ok=True)
    postprocess(img_dir, heat_dir, output_dir)

    # img_dir = '/home/zby/Cellseg/data/Val_1_3class/images_new'
    # heat_dir = '/home/zby/Cellseg/workspace/extend_beit_ep500_b18_crp512_reg_boundary/results_test/pred'
    # output_dir = heat_dir.replace('pred', 'postprocessed_cls_v2')
    # os.makedirs(output_dir,exist_ok=True)
    # postprocess_cls(img_dir, heat_dir, output_dir)
