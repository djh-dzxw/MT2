import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from datasets.cellseg import CellSeg,normalize_channel
from datasets.utils import multi_scale_test_v2
from models.unetplusplus import NestedUNet

from utils.slide_infer import slide_inference
from postprocess.postprocess_final import postprocess_final
import tifffile as tif
from skimage import io, exposure

import time
import json

def test_v2(args):
    image_names = sorted(os.listdir(args.test_image_dir))
    img_list = [os.path.join(args.test_image_dir, image_name) for image_name in image_names]

    model = NestedUNet(args)

    model_checkpoint = torch.load("./final.pth")
    model.load_state_dict(model_checkpoint, strict=True)
    model = model.half()
    model = model.cuda()
    model.eval()
    time_dict = {}
    last_stop_time = time.time()
    # Test process
    print("============== Testing ==============")
    for ii, item in enumerate(img_list):
        print("Validating the {}/{} images...".format(ii,len(img_list)),end='\r')
        # Preprocessing
        img_path = item

        if img_path.endswith('.tif') or img_path.endswith('.tiff'):
            img_data = tif.imread(img_path)
        else:
            img_data = io.imread(img_path)
        pre_img_data = normalize_channel(img_data, lower=1, upper=99)

        if len(pre_img_data.shape) == 2:
            pre_img_data = np.repeat(np.expand_dims(pre_img_data, axis=-1), 3, axis=-1)
        elif len(pre_img_data.shape) == 3 and pre_img_data.shape[-1] > 3:
            pre_img_data = pre_img_data[:,:, :3]
        else:
            pass
        img = pre_img_data.astype(np.uint8)  # HW3
        img = img[:, :, [2, 1, 0]]  # BGR

        img_flag = False
        if np.sum(img[:, :, 2]) == 0 and img.shape[0] < 5000 and img.shape[1] < 5000:
            img_flag = True

        transform = ToTensor()
        img_meta = {'img_path': img_path, 'ori_shape':img.shape, 'img_flag': img_flag}
        img, valid_region = multi_scale_test_v2(img, scale=[1.5] if img_flag else args.test_multi_scale, crop_size=args.crop_size)
        img_meta['valid_region'] = valid_region
        if isinstance(img, list):
            imgs = [transform(i).unsqueeze(0) for i in img]
        else:
            imgs = [transform(img).unsqueeze(0)]

        preprocess_time = time.time()
        
        heat_preds_list = []
        for idx, img in enumerate(imgs):
            img = img.half().cuda()
            valid_region = img_meta['valid_region'][idx]
            with torch.no_grad():
                heat_preds, boundaries = slide_inference(model, img, img_meta, rescale=True, args=args, valid_region=valid_region)
            heat_preds = torch.sigmoid(heat_preds) - torch.sigmoid(boundaries)
            heat_preds_list.append(heat_preds.detach().cpu())
        # Fusion
        if args.test_fusion =='mean':
            fused_heat_preds = torch.mean(torch.stack(heat_preds_list, dim=0), dim=0)
        if args.test_fusion == 'max':
            fused_heat_preds,_ = torch.max(torch.stack(heat_preds_list, dim=0), dim=0)
        infer_time = time.time()

        heat_preds = fused_heat_preds.squeeze()  # HW
        heat_preds = heat_preds.numpy()
        output_path = os.path.join(args.output_dir,os.path.basename(img_meta['img_path']).split('.')[0]+'_label.tiff')
        postprocess_final(heat_preds,output_path)
        postprocess_time = time.time()
        H,W,C = img_meta['ori_shape']
        if H*W < 1e6:
            tolerance_time = 10
        else:
            tolerance_time = (H*W/1e6)*10
        time_dict[os.path.basename(img_meta['img_path']).split('.')[0]] = {
            'preprocess': float(preprocess_time - last_stop_time),
            'infer': float(infer_time - preprocess_time),
            'postprocess': float(postprocess_time - infer_time),
            'total': float(postprocess_time - last_stop_time),
            'tolerance': float(tolerance_time),
            'exceed': max(0, float(postprocess_time-last_stop_time-tolerance_time))}
        last_stop_time = time.time()
    print("Test Complete!!!")
    # json.dump(time_dict,open('./times.json','w'), indent=2)

    import pandas as pd
    time_file = pd.DataFrame(columns=['img_name', 'preprocess', 'infer', 'postprocess', 'total', 'tolerance', 'exceed'])
    for ii, item in enumerate(time_dict.items()):
        img_name, item_time = item
        time_file.loc[ii+1] = [img_name, item_time['preprocess'], item_time['infer'], item_time['postprocess'], item_time['total'], item_time['tolerance'], item_time['exceed']]
    time_file.to_excel("./times_for_infer_scale0.8_1.0_1.2_stride256.xlsx")

