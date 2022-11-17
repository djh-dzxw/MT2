import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets.cellseg import CellSeg
from models.unetplusplus import NestedUNet

from utils.slide_infer import slide_inference
from postprocess.postprocess_final import postprocess_final
import tifffile as tif
# from torchsummary import summary
# from thop import profile
# from torchstat import stat
# import time
# from ptflops import get_model_complexity_info

def test(args):
    test_dataset = CellSeg(args, mode='test', preprocess_flag=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, num_workers=0)
    model = NestedUNet(args)

    model_checkpoint = torch.load("./final.pth")
    model.load_state_dict(model_checkpoint, strict=True)
    model = model.half()
    model = model.cuda()
    model.eval()
    # time_dict = {}
    # last_stop_time = time.time()
    # Test process
    print("============== Testing ==============")
    for ii, item in enumerate(test_dataloader):
        # preprocess_time = time.time()
        print("Validating the {}/{} images...".format(ii,len(test_dataloader)),end='\r')
        imgs, img_meta = item
        heat_preds_list = []
        for idx, img in enumerate(imgs):
            img = img.half().cuda()
            valid_region = img_meta['valid_region'][idx]
            with torch.no_grad():
                # preds, heat_preds, boundaries = slide_inference(model, img, img_meta, rescale=True, args=args, valid_region=valid_region)
                heat_preds, boundaries = slide_inference(model, img, img_meta, rescale=True, args=args, valid_region=valid_region)
            heat_preds = torch.sigmoid(heat_preds) - torch.sigmoid(boundaries)
            heat_preds_list.append(heat_preds.detach().cpu())
        # Fusion
        if args.test_fusion =='mean':
            fused_heat_preds = torch.mean(torch.stack(heat_preds_list, dim=0), dim=0)
        if args.test_fusion == 'max':
            fused_heat_preds,_ = torch.max(torch.stack(heat_preds_list, dim=0), dim=0)
        # infer_time = time.time()

        heat_preds = fused_heat_preds.squeeze()  # HW
        heat_preds = heat_preds.numpy()
        output_path = os.path.join(args.output_dir,os.path.basename(img_meta['img_path'][0]).split('.')[0]+'_label.tiff')
        postprocess_final(heat_preds,output_path)
        # postprocess_time = time.time()
        # H,W,C = img_meta['ori_shape']
        # if H*W < 1e6:
        #     tolerance_time = 10
        # else:
        #     tolerance_time = (H*W/1e6)*10
        # time_dict[os.path.basename(img_meta['img_path'][0]).split('.')[0]] = {
        #     'preprocess': float(preprocess_time - last_stop_time),
        #     'infer': float(infer_time - preprocess_time),
        #     'postprocess': float(postprocess_time - infer_time),
        #     'total': float(postprocess_time - last_stop_time),
        #     'tolerance': float(tolerance_time),
        #     'exceed': max(0, float(postprocess_time-last_stop_time-tolerance_time))}
        # last_stop_time = time.time()
    print("Test Complete!!!")
    # json.dump(time_dict,open('./times.json','w'), indent=2)

    # import pandas as pd
    # time_file = pd.DataFrame(columns=['img_name', 'preprocess', 'infer', 'postprocess', 'total', 'tolerance', 'exceed'])
    # for ii, item in enumerate(time_dict.items()):
    #     img_name, item_time = item
    #     time_file.loc[ii+1] = [img_name, item_time['preprocess'], item_time['infer'], item_time['postprocess'], item_time['total'], item_time['tolerance'], item_time['exceed']]
    # excel_path = args.output_dir.replace('outputs','times')+'.xlsx'
    # time_file.to_excel(excel_path)

if __name__=="__main__":
    test()