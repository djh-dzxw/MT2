import os
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets.cellseg import CellSeg
from models.resnet50 import UNet

from utils.slide_infer import slide_inference
from utils.f1_score import gen_upload_tiff, get_f1_score, get_f1_score_with_heatmap


def infer(args, logger):
    test_dataset = CellSeg(args, mode='val')  #
    test_dataloader = DataLoader(test_dataset, batch_size=1, num_workers=1)

    model = UNet(args)
    logger.info("Loading checkpoint from: {}".format(os.path.join(args.workspace, args.checkpoint)))
    model.load_state_dict(torch.load(os.path.join(args.workspace, args.checkpoint)), strict=True)
    model = model.cuda()
    # model = torch.nn.DataParallel(model.cuda())
    model.eval()

    logger.info("============== Testing ==============")
    for ii, item in enumerate(test_dataloader):
        if ii % 20 == 0:
            logger.info("Validating the {}/{} images...".format(ii, len(test_dataloader)))
        img, anno, img_meta = item
        img = img.cuda()
        anno = anno.cuda()
        preds, heat_preds, boundaries = slide_inference(model, img, img_meta, rescale=True, args=args)
        preds = torch.argmax(preds.detach().cpu(), dim=1).squeeze().numpy()
        preds[preds == 1] = 128
        preds[preds == 2] = 255
        heat_preds = heat_preds - boundaries
        heat_preds = heat_preds.squeeze()
        heat_preds = heat_preds.detach().cpu().numpy() * 255
        heat_preds = np.clip(heat_preds, 0, 255)
        # Visualization
        img_path = img_meta['img_path'][0]
        save_path = os.path.join(args.workspace, args.results_val, 'pred', os.path.basename(img_path))
        heat_save_path = os.path.join(args.workspace, args.results_val, 'heat_pred', os.path.basename(img_path))
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        os.makedirs(os.path.dirname(heat_save_path), exist_ok=True)
        cv2.imwrite(save_path, preds)
        cv2.imwrite(heat_save_path, heat_preds)
    logger.info("Test Complete!!!")

    logger.info("============== Calculating Metrics ==============")
    results, F1_score = get_f1_score(args)
    logger.info("Results:")
    logger.info("F1@0.5: {} ".format(results['F1@0.5']))
    logger.info("F1@0.75: {} ".format(results['F1@0.75']))
    logger.info("F1@0.9: {} ".format(results['F1@0.9']))
    logger.info("F1@0.5:1.0:0.05: {} ".format(results['F1@0.5:1.0:0.05']))

    heat_results, heat_F1_score = get_f1_score_with_heatmap(args)
    logger.info("Heatmap Results:")
    logger.info("F1@0.5: {} ".format(heat_results['F1@0.5']))
    logger.info("F1@0.75: {} ".format(heat_results['F1@0.75']))
    logger.info("F1@0.9: {} ".format(heat_results['F1@0.9']))
    logger.info("F1@0.5:1.0:0.05: {} ".format(heat_results['F1@0.5:1.0:0.05']))
    logger.info("============== Test Complete! ==============")
    # gen_upload_tiff(args)


if __name__ == "__main__":
    infer()
