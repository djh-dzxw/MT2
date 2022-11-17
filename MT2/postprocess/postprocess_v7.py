import os
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.segmentation import watershed
from skimage import measure
from skimage.transform import rescale
import tifffile as tif
from skimage.color import label2rgb
from skimage.morphology import erosion, disk, dilation, erosion
from skimage.morphology import square
from skimage.morphology.greyreconstruct import reconstruction


def postprocess(img_dir, heat_dir, output_dir):
    heat_list = sorted(os.listdir(heat_dir))
    heat_paths = [os.path.join(heat_dir, heat_name) for heat_name in heat_list]
    for ii, heat_path in enumerate(heat_paths):
        print("Processing the {}/{} images....".format(ii, len(heat_paths)), end='\r')
        img_path = heat_path.replace(heat_dir, img_dir)
        heatmap = cv2.imread(heat_path, flags=0) / 255
        th_cell = 0.5
        th_seed = 0.7
        new_inst_map = mc_distance_postprocessing(heatmap, th_cell, th_seed, downsample=False)
        tif.imwrite(os.path.join(output_dir, heat_path.replace(heat_dir, output_dir).replace('.png', '_label.tiff')),
                    new_inst_map, compression='zlib')

        # color labels
        # clr_labels = label2rgb(new_inst_map, bg_label=0)
        # clr_labels *= 255
        # clr_labels = clr_labels.astype('uint8')
        # os.makedirs(os.path.join(output_dir, 'vis'), exist_ok=True)
        # cv2.imwrite(
        #     os.path.join(output_dir, 'vis', os.path.basename(heat_path)),
        #     clr_labels)


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

    # min_area = 50  # keep only seeds larger than threshold area

    # Instance segmentation (use summed up channel 0)
    # sigma_cell = 0.5
    # cell_prediction[0] = gaussian_filter(cell_prediction[0], sigma=sigma_cell)  # slight smoothing

    mask_ori = cell_prediction > th_cell  # get binary mask by thresholding distance prediction
    # mask_ero = erosion(mask_ori, square(7))
    # mask = reconstruction(mask_ero, mask_ori, 'dilation')

    # ret, heatmap = cv2.connectedComponents(heatmap.astype(np.int8))
    # new_inst_map = cv2.dilate(heatmap.astype(np.uint16), kernel_2, iterations=1)
    ori_inst_map = measure.label(mask_ori, background=0)
    props = measure.regionprops(ori_inst_map)

    region_miu = np.mean([prop.area for prop in props if prop.area > 50])
    region_sigma = np.sqrt(np.var([prop.area for prop in props if prop.area > 50]))
    region_range = [region_miu - 2 * region_sigma, region_miu + 2 * region_sigma]
    for idx, prop in enumerate(props):
        if prop.area < region_range[0] or prop.area <= 50:
            ori_inst_map[ori_inst_map == prop.label] = 0
    inst_map = measure.label(ori_inst_map, background=0)

    prediction_instance = watershed(image=-cell_prediction, markers=inst_map, mask=mask_ori * inst_map,
                                    watershed_line=False)

    # seeds = measure.label(cell_prediction > th_seed, background=0)  # get seeds
    # props = measure.regionprops(seeds)  # Remove very small seeds
    # for idx, prop in enumerate(props):
    #     if prop.area < min_area:
    #         seeds[seeds == prop.label] = 0
    # seeds = measure.label(seeds, background=0)
    # prediction_instance = watershed(image=-cell_prediction, markers=seeds, mask=mask,
    #                                 watershed_line=False)

    # # Semantic segmentation / classification
    # prediction_class = np.zeros_like(prediction_instance)
    # for idx in range(1, prediction_instance.max()+1):
    #     # Get sum of distance prediction of selected cell for each class (class 0 is sum of the other classes)
    #     pix_vals = cell_prediction[1:][:, prediction_instance == idx]
    #     cell_layer = np.sum(pix_vals, axis=1).argmax() + 1  # +1 since class 0 needs to be considered for argmax
    #     prediction_class[prediction_instance == idx] = cell_layer

    if downsample:
        # Downsample instance segmentation
        prediction_instance = rescale(inst_map,
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
    return prediction.astype(np.int32)


if __name__ == "__main__":
    img_dir = '/home/zby/Cellseg/data/fix_boundary/images_new'
    heat_dir = '/raid/zby/Cellseg_zby/workspace/unetpp_78/heat_pred/'
    output_dir = heat_dir.replace('heat_pred', 'postprocessed_reg_v7')
    os.makedirs(output_dir, exist_ok=True)
    postprocess(img_dir, heat_dir, output_dir)
