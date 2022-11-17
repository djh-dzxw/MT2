import os
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models

from datasets.cellseg import CellSeg
from models.resnet50 import UNet
from models.deeplabv3plus.modeling import *
from models.unetplusplus import NestedUNet
from models.CENet import CE_Net_, CE_Net_backbone_DAC_with_inception, CE_Net_backbone_DAC_without_atrous, \
    CE_Net_backbone_inception_blocks, CE_Net_OCT
from models.BEIT import BEiT
from models.swinunetr import SwinUNETR
# from models.SegNeXt.SegNeXt import SegNeXt
# from models.unetformer import UNetFormer
from models.loss import DiceLoss, FocalLoss

from utils.slide_infer import slide_inference
from utils.f1_score import gen_upload_tiff, get_f1_score, get_f1_score_with_heatmap


def test(args, logger):
    # if os.path.isdir(os.path.join(args.workspace, args.results_test, "heat_pred")):
    #     gen_upload_tiff(args)
    #     return
    test_dataset = CellSeg(args, mode='test')  #
    test_dataloader = DataLoader(test_dataset, batch_size=1, num_workers=1)

    if args.net_name.lower() == 'unet':
        print("Using Model: unet")
        model = UNet(args)
    elif args.net_name.lower() == 'deeplabv3plus_xception':
        print("Using Model: deeplabv3plus_xception")
        model = deeplabv3plus_xception(num_classes=args.net_num_classes, output_stride=8)
    elif args.net_name.lower() == 'deeplabv3plus_r50':
        print("Using Model: deeplabv3plus_r50")
        model = deeplabv3plus_resnet50(num_classes=args.net_num_classes, output_stride=8)
    elif args.net_name.lower() == 'deeplabv3plus_r101':
        print("Using Model: deeplabv3plus_r101")
        model = deeplabv3plus_resnet101(num_classes=args.net_num_classes, output_stride=8)
    elif args.net_name.lower() == 'deeplabv3plus_hr32':
        print("Using Model: deeplabv3plus_hr32")
        model = deeplabv3plus_hrnetv2_32(num_classes=args.net_num_classes, output_stride=4)
    elif args.net_name.lower() == 'deeplabv3plus_hr48':
        print("Using Model: deeplabv3plus_hr48")
        model = deeplabv3plus_hrnetv2_48(num_classes=args.net_num_classes, output_stride=4)
    elif args.net_name.lower() == 'deeplabv3plus_mobile':
        print("Using Model: deeplabv3plus_mobile")
        model = deeplabv3plus_mobilenet(num_classes=args.net_num_classes, output_stride=8)
    elif args.net_name.lower() == 'unetplusplus':
        print("Using Model: unetplusplus")
        model = NestedUNet(args)
    elif args.net_name.lower() == 'cenet':
        print("Using Model: cenet")
        model = CE_Net_(args)
    elif args.net_name.lower() == 'beit':
        print("Using Model: beit")
        # model = BEiT(img_size=args.crop_size, patch_size=8, num_classes=args.net_num_classes, embed_dim=384, drop_rate=0.1, num_heads=4)
        model = BEiT(img_size=args.crop_size, patch_size=16, num_classes=args.net_num_classes, embed_dim=768,
                     drop_rate=0.1, attn_drop_rate=0.)
    elif args.net_name.lower() == 'swinunetr':
        print("Using Model: swinunetr")
        model = SwinUNETR(args.crop_size, in_channels=3, out_channels=3, use_checkpoint=True, spatial_dims=2)
    elif args.net_name.lower() == 'segnext':
        print("Using Model: segnext_small")
        model = SegNeXt()
    elif args.net_name.lower() == 'unetformer':
        print("Using Model: unetformer")
        model = UNetFormer()
    else:
        raise NotImplementedError("Model {} is not implemented!".format(args.net_name.lower()))

    if args.net_preclass == True:
        precls_net = models.resnet50(num_classes=11).cuda()
        precls_net.load_state_dict(torch.load(args.net_preclass_checkpoint), strict=True)
        precls_net = torch.nn.DataParallel(precls_net.cuda())
        precls_net.eval()

    logger.info("Loading checkpoint from: {}".format(os.path.join(args.workspace, args.checkpoint)))
    model.load_state_dict(torch.load(os.path.join(args.workspace, args.checkpoint)), strict=True)
    model = model.cuda()
    # model = torch.nn.DataParallel(model.cuda())
    model.eval()

    logger.info("============== Testing ==============")
    for ii, item in enumerate(test_dataloader):
        if ii % 10 == 0:
            logger.info("Validating the {}/{} images...".format(ii, len(test_dataloader)))
        imgs, annos, img_meta = item
        preds_list = []
        heat_preds_list = []
        boundaries_list = []
        for idx, img in enumerate(imgs):
            img = img.cuda()
            valid_region = img_meta['valid_region'][idx]
            with torch.no_grad():
                if args.net_preclass == True:
                    thumbnail = img_meta['thumbnail'].cuda()
                    pred_cls = precls_net(thumbnail)
                    pred_cls = torch.argmax(pred_cls, dim=1)
                    preds, heat_preds, boundaries = slide_inference(model, img, img_meta, rescale=True, args=args,
                                                                    valid_region=valid_region, pred_cls=pred_cls)
                else:
                    preds, heat_preds, boundaries = slide_inference(model, img, img_meta, rescale=True, args=args,
                                                                    valid_region=valid_region)
            # Classification
            preds = torch.softmax(preds, dim=1)
            # preds = torch.argmax(preds.detach().cpu(),dim=1).squeeze().numpy()
            # preds[preds==1] = 128
            # preds[preds==2] = 255
            # Regression
            heat_preds = torch.sigmoid(heat_preds) - torch.sigmoid(boundaries)
            # heat_preds = heat_preds.squeeze()
            # heat_preds = heat_preds.detach().cpu().numpy()*255
            # heat_preds = np.clip(heat_preds,0,255)
            # boundaries = boundaries.squeeze().detach().cpu().numpy()*255

            preds_list.append(preds.detach().cpu())
            heat_preds_list.append(heat_preds.detach().cpu())
            boundaries_list.append(boundaries.detach().cpu())
        # Fusion
        if args.test_fusion == 'mean':
            fused_preds = torch.mean(torch.stack(preds_list, dim=0), dim=0)
            fused_heat_preds = torch.mean(torch.stack(heat_preds_list, dim=0), dim=0)
            fused_boundaries = torch.mean(torch.stack(boundaries_list, dim=0), dim=0)
        if args.test_fusion == 'max':
            fused_preds, _ = torch.max(torch.stack(preds_list, dim=0), dim=0)
            fused_heat_preds, _ = torch.max(torch.stack(heat_preds_list, dim=0), dim=0)
            fused_boundaries, _ = torch.max(torch.stack(boundaries_list, dim=0), dim=0)

        preds = torch.argmax(fused_preds, dim=1).squeeze().numpy()
        preds[preds == 1] = 128
        preds[preds == 2] = 255
        heat_preds = fused_heat_preds.squeeze()
        heat_preds = heat_preds.numpy() * 255
        heat_preds = np.clip(heat_preds, 0, 255)
        boundaries = fused_boundaries.squeeze().numpy() * 255

        # Visualization
        img_path = img_meta['img_path'][0]
        save_path = os.path.join(args.workspace, args.results_test, 'pred', os.path.basename(img_path))
        heat_save_path = os.path.join(args.workspace, args.results_test, 'heat_pred', os.path.basename(img_path))
        boundary_save_path = os.path.join(args.workspace, args.results_test, 'boundary_pred',
                                          os.path.basename(img_path))
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        os.makedirs(os.path.dirname(heat_save_path), exist_ok=True)
        os.makedirs(os.path.dirname(boundary_save_path), exist_ok=True)
        cv2.imwrite(save_path, preds)
        cv2.imwrite(heat_save_path, heat_preds)
        cv2.imwrite(boundary_save_path, boundaries)
    logger.info("Test Complete!!!")

    # gen_upload_tiff(args)


if __name__ == "__main__":
    test()
