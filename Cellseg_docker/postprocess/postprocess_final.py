import os
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.segmentation import watershed
from skimage import measure
from skimage.transform import rescale
import tifffile as tif
from skimage.color import label2rgb


def postprocess_v6(heatmap, th_cell, th_seed, output_path):
    mask, seeds, new_inst_map = mc_distance_postprocessing(heatmap, th_cell, th_seed, downsample=False)

    # color labels
    # output_dir = os.path.dirname(output_path)

    # clr_labels = label2rgb(new_inst_map, bg_label=0)
    # clr_labels *= 255
    # clr_labels = clr_labels.astype('uint8')
    # os.makedirs(os.path.join(output_dir, 'vis'), exist_ok=True)
    # cv2.imwrite(os.path.join(output_dir, 'vis', os.path.basename(output_path)),clr_labels)

    # mask = mask.astype(np.uint8)
    # seeds = (seeds > 0).astype(np.uint8)
    # bbd = mask - seeds
    # hc_img = np.concatenate((mask, seeds, bbd), axis=1)  # H W
    # hc_img = (hc_img * 255).astype(np.uint8)
    # os.makedirs(os.path.join(output_dir, 'vis_bbd'), exist_ok=True)
    # cv2.imwrite(os.path.join(output_dir, 'vis_bbd', os.path.basename(output_path)),hc_img)

    return new_inst_map


def mc_distance_postprocessing(cell_prediction, th_cell, th_seed, downsample):
    """ Post-processing for distance label (cell + neighbor) prediction.

    :param cell_prediction: heatmap，0-1
    :type cell_prediction:
    :param th_cell:
    :type th_cell: float
    :param th_seed:
    :type th_seed: float
    :param downsample:
    :type downsample: bool

    :return: Instance segmentation mask.
    """

    min_area = 64  # keep only seeds larger than threshold area

    # Instance segmentation (use summed up channel 0)
    mask = cell_prediction > th_cell  # get binary mask by thresholding distance prediction

    seeds = measure.label(cell_prediction > th_seed, background=0)
    props = measure.regionprops(seeds)  # Remove very small seeds

    for idx, prop in enumerate(props):
        if prop.area < min_area:
            seeds[seeds == prop.label] = 0

    seeds = measure.label(seeds, background=0)
    prediction_instance = watershed(image=-cell_prediction, markers=seeds, mask=mask, watershed_line=False)

    # # Semantic segmentation / classification
    # prediction_class = np.zeros_like(prediction_instance)
    # for idx in range(1, prediction_instance.max()+1):
    #     # Get sum of distance prediction of selected cell for each class (class 0 is sum of the other classes)
    #     pix_vals = cell_prediction[1:][:, prediction_instance == idx]
    #     cell_layer = np.sum(pix_vals, axis=1).argmax() + 1  # +1 since class 0 needs to be considered for argmax
    #     prediction_class[prediction_instance == idx] = cell_layer

    if downsample:
        # Downsample instance segmentation
        prediction_instance = rescale(prediction_instance,
                                      scale=0.8,
                                      order=0,
                                      preserve_range=True,
                                      anti_aliasing=False).astype(np.int32)

        # Downsample semantic segmentation
        # prediction_class = rescale(prediction_class,
        #                            scale=0.8,
        #                            order=0,
        #                            preserve_range=True,
        #                            anti_aliasing=False).astype(np.uint16)

    # Combine instance segmentation and semantic segmentation results
    # prediction = np.concatenate((prediction_instance[np.newaxis, ...], prediction_class[np.newaxis, ...]), axis=0)
    prediction = prediction_instance
    return mask, seeds, prediction.astype(np.int32)


def postprocess_v2(heatmap, threshold):
    heatmap[heatmap >= threshold] = 1
    heatmap[heatmap < threshold] = 0
    kernel = np.ones((5, 5), np.uint8)
    heatmap = cv2.morphologyEx(heatmap, cv2.MORPH_OPEN, kernel, iterations=1)
    ret, heatmap = cv2.connectedComponents(heatmap.astype(np.int8), ltype=cv2.CV_32S)
    if np.max(heatmap) > 65000:
        heatmap = heatmap.astype(np.float32)
    else:
        heatmap = heatmap.astype(np.uint16)
    kernel = np.ones((3, 3), np.uint8)
    new_inst_map = cv2.dilate(heatmap, kernel, iterations=1)
    if np.max(new_inst_map) > 65000:
        new_inst_map = new_inst_map.astype(np.int32)
    else:
        new_inst_map = new_inst_map.astype(np.int16)
    return new_inst_map


def postprocess(img_dir, heat_dir, output_dir):
    heat_list = sorted(os.listdir(heat_dir))
    heat_paths = [os.path.join(heat_dir, heat_name) for heat_name in heat_list]
    thres_v2 = 0.6
    thres_cell_v6 = 0.3
    thres_seed_v6 = 0.7
    for ii, heat_path in enumerate(heat_paths):
        print("Processing the {}/{} images....".format(ii, len(heat_paths)), end='\r')
        heatmap = cv2.imread(heat_path, flags=0) / 255
        H, W = heatmap.shape
        if H > 5000 or W > 5000:
            new_inst_map = postprocess_v2(heatmap, thres_v2)
        else:
            new_inst_map = postprocess_v6(heatmap, thres_cell_v6, thres_seed_v6)
        tif.imwrite(os.path.join(heat_path.replace(heat_dir, output_dir).replace('.png', '_label.tiff')), new_inst_map,
                    compression='zlib')


def postprocess_final(heatmap, output_path):
    thres_v2 = 0.6
    thres_cell_v6 = 0.3
    thres_seed_v6 = 0.8
    H, W = heatmap.shape
    if H > 50000 or W > 50000:
        new_inst_map = postprocess_v2(heatmap, thres_v2)
    else:
        new_inst_map = postprocess_v6(heatmap, thres_cell_v6, thres_seed_v6, output_path)
    tif.imwrite(output_path, new_inst_map, compression='zlib')


if __name__ == "__main__":
    img_dir = '/home/zby/Cellseg/data/fix_boundary/images_new'
    heat_dir = '/home/dmt218/Cellseg/workspace/extendv2_unetpp_ep500_b24_crp512_reg_bnd_dice_cm_s0.4_4.0_cj_b5_c5_s0/results_test/heat_pred'
    output_dir = heat_dir.replace('heat_pred', 'postprocessed_final')
    os.makedirs(output_dir, exist_ok=True)
    postprocess(img_dir, heat_dir, output_dir)
