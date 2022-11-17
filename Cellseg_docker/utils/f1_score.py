# Taken from https://github.com/carpenterlab/unet4nuclei/blob/master/unet4nuclei/utils/evaluation.py
# Aug 06 / 2018

import numpy as np
import pandas as pd
import os
import cv2
import numpy as np
from skimage import measure, color, morphology
from PIL import Image
import tifffile as tif
from utils.compute_metric import compute_metric


def intersection_over_union(ground_truth, prediction):
    # Count objects
    true_objects = len(np.unique(ground_truth))
    pred_objects = len(np.unique(prediction))

    # Compute intersection
    h = np.histogram2d(ground_truth.flatten(), prediction.flatten(), bins=(true_objects, pred_objects))
    intersection = h[0]

    # Area of objects
    area_true = np.histogram(ground_truth, bins=true_objects)[0]
    area_pred = np.histogram(prediction, bins=pred_objects)[0]

    # Calculate union
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)
    union = area_true + area_pred - intersection

    # Exclude background from the analysis
    intersection = intersection[1:, 1:]
    union = union[1:, 1:]

    # Compute Intersection over Union
    union[union == 0] = 1e-9
    IOU = intersection / union

    return IOU


def measures_at(threshold, IOU):
    matches = IOU > threshold

    true_positives = np.sum(matches, axis=1) == 1  # Correct objects
    false_positives = np.sum(matches, axis=0) == 0  # Extra objects
    false_negatives = np.sum(matches, axis=1) == 0  # Missed objects

    assert np.all(np.less_equal(true_positives, 1))
    assert np.all(np.less_equal(false_positives, 1))
    assert np.all(np.less_equal(false_negatives, 1))

    TP, FP, FN = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)

    f1 = 2 * TP / (2 * TP + FP + FN + 1e-9)
    official_score = TP / (TP + FP + FN + 1e-9)

    precision = TP / (TP + FP + 1e-9)
    recall = TP / (TP + FN + 1e-9)

    return f1, TP, FP, FN, official_score, precision, recall


# Compute Average Precision for all IoU thresholds
def compute_af1_results(ground_truth, prediction, results, image_name):
    # Compute IoU
    IOU = intersection_over_union(ground_truth, prediction)
    if IOU.shape[0] > 0:
        jaccard = np.max(IOU, axis=0).mean()
    else:
        jaccard = 0.0

    # Calculate F1 score at all thresholds
    f1_list = []
    for t in np.arange(0.5, 1.0, 0.05):
        f1, tp, fp, fn, os, prec, rec = measures_at(t, IOU)
        res = {"Image": image_name, "Threshold": t, "F1": f1, "Jaccard": jaccard,
               "TP": tp, "FP": fp, "FN": fn, "Official_Score": os, "Precision": prec, "Recall": rec}
        f1_list.append(f1)
        # row = len(results)
        # results.loc[row] = res
    # return results
    return f1_list


# Count number of False Negatives at 0.7 IoU
def get_false_negatives(ground_truth, prediction, results, image_name, threshold=0.7):
    # Compute IoU
    IOU = intersection_over_union(ground_truth, prediction)
    true_objects = len(np.unique(ground_truth))
    if true_objects <= 1:
        return results

    area_true = np.histogram(ground_truth, bins=true_objects)[0][1:]
    true_objects -= 1

    # Identify False Negatives
    matches = IOU > threshold
    false_negatives = np.sum(matches, axis=1) == 0

    data = np.asarray([
        area_true.copy(),
        np.array(false_negatives, dtype=np.int32)
    ])

    results = pd.concat([results, pd.DataFrame(data=data.T, columns=["Area", "False_Negative"])])

    return results


# Count the number of splits and merges
def get_splits_and_merges(ground_truth, prediction, results, image_name):
    # Compute IoU
    IOU = intersection_over_union(ground_truth, prediction)

    matches = IOU > 0.1
    merges = np.sum(matches, axis=0) > 1
    splits = np.sum(matches, axis=1) > 1
    r = {"Image_Name": image_name, "Merges": np.sum(merges), "Splits": np.sum(splits)}
    results.loc[len(results) + 1] = r
    return results


def get_f1_score(args):
    show_dir = os.path.join(args.workspace, args.results_val)
    all_f1_results = []
    pred_dir = os.path.join(show_dir, 'pred')
    upload_dir = os.path.join(show_dir, 'BUPT_MCPRL')
    os.makedirs(upload_dir, exist_ok=True)
    pred_names = os.listdir(pred_dir)
    for ii, pred_name in enumerate(pred_names):
        print("Calculating the {}/{} samples....".format(ii, len(pred_names)), end='\r')
        pred_path = os.path.join(pred_dir, pred_name)
        seg_path = os.path.join(args.gt_dir, pred_name.replace('.png', '_label.tiff'))
        pred = cv2.imread(pred_path, flags=0)
        seg = np.array(Image.open(seg_path))
        pred[pred == 255] = 0
        pred[pred == 128] = 1
        pred = measure.label(pred, background=0)
        tif.imwrite(os.path.join(upload_dir, pred_name.split('.')[0] + '_label.tiff'), pred,
                    compression='zlib')
        f1_list = np.array(compute_af1_results(seg, pred, 0, 0))
        if all_f1_results == []:
            all_f1_results = f1_list
        else:
            all_f1_results += f1_list
    all_f1_results = all_f1_results / len(pred_names)
    results = {
        'F1@0.5': all_f1_results[0],
        'F1@0.75': all_f1_results[5],
        'F1@0.9': all_f1_results[8],
        'F1@0.5:1.0:0.05': np.mean(all_f1_results),
    }
    F1_score = compute_metric(args.gt_dir, upload_dir, args.workspace)
    return results, F1_score


def get_f1_score_with_heatmap(args):
    threshold = args.infer_threshold
    show_dir = os.path.join(args.workspace, args.results_val)
    all_f1_results = []
    pred_dir = os.path.join(show_dir, 'heat_pred')
    upload_dir = os.path.join(show_dir, 'BUPT_MCPRL_heatmap')
    os.makedirs(upload_dir, exist_ok=True)
    pred_names = os.listdir(pred_dir)
    for ii, pred_name in enumerate(pred_names):
        print("Calculating the {}/{} samples....".format(ii, len(pred_names)), end='\r')
        pred_path = os.path.join(pred_dir, pred_name)
        seg_path = os.path.join(args.gt_dir, pred_name.replace('.png', '_label.tiff'))
        pred = cv2.imread(pred_path, flags=0) / 255
        seg = np.array(Image.open(seg_path))
        pred[pred >= threshold] = 1
        pred[pred < threshold] = 0
        # pred = morphology.diameter_closing(pred)
        pred = measure.label(pred, background=0)
        tif.imwrite(os.path.join(upload_dir, pred_name.split('.')[0] + '_label.tiff'), pred, compression='zlib')
        f1_list = np.array(compute_af1_results(seg, pred, 0, 0))
        if all_f1_results == []:
            all_f1_results = f1_list
        else:
            all_f1_results += f1_list
    all_f1_results = all_f1_results / len(pred_names)
    results = {
        'F1@0.5': all_f1_results[0],
        'F1@0.75': all_f1_results[5],
        'F1@0.9': all_f1_results[8],
        'F1@0.5:1.0:0.05': np.mean(all_f1_results),
    }

    F1_score = compute_metric(args.gt_dir, upload_dir, args.workspace)
    return results, F1_score


def gen_upload_tiff(args):
    show_dir = os.path.join(args.workspace, args.results_test)
    pred_dir = os.path.join(show_dir, 'pred')
    upload_dir = os.path.join(show_dir, 'BUPT_MCPRL')
    heat_pred_dir = os.path.join(show_dir, 'heat_pred')
    heat_upload_dir = os.path.join(show_dir, 'BUPT_MCPRL_heatmap')
    os.makedirs(upload_dir, exist_ok=True)
    os.makedirs(heat_upload_dir, exist_ok=True)
    pred_names = os.listdir(pred_dir)
    for ii, pred_name in enumerate(pred_names):
        print("Calculating the {}/{} samples....".format(ii, len(pred_names)), end='\r')
        pred_path = os.path.join(pred_dir, pred_name)
        pred = cv2.imread(pred_path, flags=0)
        pred[pred == 255] = 0
        pred[pred == 128] = 1
        pred = measure.label(pred, background=0)
        tif.imwrite(os.path.join(upload_dir, pred_name.split('.')[0] + '_label.tiff'), pred, compression='zlib')

    heat_pred_names = os.listdir(heat_pred_dir)
    threshold = args.test_threshold
    for ii, heat_pred_name in enumerate(heat_pred_names):
        print("Calculating the {}/{} samples....".format(ii, len(heat_pred_names)), end='\r')
        heat_pred_path = os.path.join(heat_pred_dir, heat_pred_name)
        print("Loading from: {}".format(heat_pred_path))
        heat_pred = cv2.imread(heat_pred_path, flags=0) / 255
        heat_pred[heat_pred >= threshold] = 1
        heat_pred[heat_pred < threshold] = 0
        heat_pred = measure.label(heat_pred, background=0)
        print("Save to: {}".format(os.path.join(heat_upload_dir, heat_pred_name.split('.')[0] + '_label.tiff')))
        tif.imwrite(os.path.join(heat_upload_dir, heat_pred_name.split('.')[0] + '_label.tiff'), heat_pred,
                    compression='zlib')
