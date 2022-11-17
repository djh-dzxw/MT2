import os
from random import sample
import cv2
import numpy as np
import copy
# from sklearn.metrics import f1_score
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR, StepLR
from torch.utils.data import DataLoader, WeightedRandomSampler
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
from models.loss import DiceLoss, FocalLoss, SSIM, ssimloss

from utils.slide_infer import slide_inference
from utils.f1_score import get_f1_score, get_f1_score_with_heatmap
from utils.ema_update import ema_update
from utils.cutmix import CutMix, CutMix_unlabeled


def cycle(iterable):
    iterator = iter(iterable)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(iterable)


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def validate(args, logger, model, dataloader):
    logger.info("============== Validating ==============")
    for ii, item in enumerate(dataloader):
        if ii % 50 == 0:
            logger.info("Validating the {}/{} images...".format(ii, len(dataloader)))
        img, anno, img_meta = item
        img = img.cuda()
        anno = anno.cuda()
        valid_region = img_meta['valid_region']
        preds, heat_preds, boundary_preds = slide_inference(model, img, img_meta, rescale=True, args=args,
                                                            valid_region=valid_region)
        preds = torch.argmax(preds.detach().cpu(), dim=1).squeeze().numpy()
        preds[preds == 1] = 128
        preds[preds == 2] = 255
        heat_preds = torch.sigmoid(heat_preds) - torch.sigmoid(boundary_preds)
        heat_preds = heat_preds.squeeze()
        heat_preds = heat_preds.detach().cpu().numpy() * 255
        heat_preds = np.clip(heat_preds, 0, 255)
        boundary_preds = boundary_preds.squeeze().detach().cpu().numpy() * 255
        # print(heat_preds.shape)
        # Visualization
        img_path = img_meta['img_path'][0]
        save_path = os.path.join(args.workspace, args.results_val, 'pred', os.path.basename(img_path))
        heat_save_path = os.path.join(args.workspace, args.results_val, 'heat_pred', os.path.basename(img_path))
        boundary_save_path = os.path.join(args.workspace, args.results_val, 'boundary_pred', os.path.basename(img_path))
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        os.makedirs(os.path.dirname(heat_save_path), exist_ok=True)
        os.makedirs(os.path.dirname(boundary_save_path), exist_ok=True)
        cv2.imwrite(save_path, preds)
        cv2.imwrite(heat_save_path, heat_preds)
        cv2.imwrite(boundary_save_path, boundary_preds)
    logger.info("Validation Complete!!!")
    logger.info("============== Calculating Metrics ==============")
    results, F1_score = get_f1_score(args)
    logger.info("Results:")
    logger.info("F1@0.5: {} ".format(results['F1@0.5']))
    logger.info("F1@0.75: {} ".format(results['F1@0.75']))
    logger.info("F1@0.9: {} ".format(results['F1@0.9']))
    logger.info("F1@0.5:1.0:0.05: {} ".format(results['F1@0.5:1.0:0.05']))
    logger.info("F1@0.5_official: {} ".format(F1_score))

    heat_results, heat_F1_score = get_f1_score_with_heatmap(args)
    logger.info("Heatmap Results:")
    logger.info("F1@0.5: {} ".format(heat_results['F1@0.5']))
    logger.info("F1@0.75: {} ".format(heat_results['F1@0.75']))
    logger.info("F1@0.9: {} ".format(heat_results['F1@0.9']))
    logger.info("F1@0.5:1.0:0.05: {} ".format(heat_results['F1@0.5:1.0:0.05']))
    logger.info("F1@0.5_official: {} ".format(heat_F1_score))

    return results['F1@0.5'], heat_results['F1@0.5']


def train(args, logger):
    seed = 512
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.enabled = True

    train_dataset = CellSeg(args, mode=args.train_mode)
    val_dataset = CellSeg(args, mode='val')
    # sampler = WeightedRandomSampler(train_dataset.weights, num_samples=round(len(train_dataset)/args.batch_size))
    # train_dataloader = DataLoader(train_dataloader,batch_size=args.batch_size, sampler=sampler, num_workers=args.num_worker)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_worker)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)

    if args.semi:
        unlabeled_dataset = CellSeg(args, mode='unlabeled')
        unlabeled_dataloader = DataLoader(unlabeled_dataset, batch_size=args.unlabeled_batch_size, shuffle=True,
                                          num_workers=args.num_worker)

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
                     drop_rate=0.1)
    elif args.net_name.lower() == 'swinunetr':
        print("Using Model: swinunetr")
        model = SwinUNETR(args.crop_size, in_channels=3, out_channels=3, depths=(2, 2, 2, 2), feature_size=24,
                          use_checkpoint=True, spatial_dims=2)
    elif args.net_name.lower() == 'segnext':
        print("Using Model: segnext_small")
        model = SegNeXt()
    elif args.net_name.lower() == 'unetformer':
        print("Using Model: unetformer")
        model = UNetFormer()
    else:
        raise NotImplementedError("Model {} is not implemented!".format(args.net_name.lower()))

    if args.train_resume is not None:
        model.load_state_dict(torch.load(args.train_resume), strict=True)
        print("Load model successfully!")

    # Using the mean-teacher
    if args.net_mean_teacher == True:
        teacher_model = copy.deepcopy(model)
        teacher_model.eval()

    if args.semi:
        model_T = copy.deepcopy(model)
        # if args.train_resume is not None:
        #     model_T.load_state_dict(torch.load(args.train_resume_T), strict=True)

    model = torch.nn.DataParallel(model.cuda())

    if args.semi:
        model_T = torch.nn.DataParallel(model_T.cuda())

    if args.net_mean_teacher == True:
        teacher_model = torch.nn.DataParallel(teacher_model.cuda())

    if args.net_preclass == True:
        precls_net = models.resnet50(num_classes=11).cuda()
        precls_net.load_state_dict(torch.load(args.net_preclass_checkpoint), strict=True)
        precls_net = torch.nn.DataParallel(precls_net.cuda())
        precls_net.eval()

    ceLoss = nn.CrossEntropyLoss(ignore_index=255)
    diceloss = DiceLoss()
    focalloss = FocalLoss()
    mseloss = nn.MSELoss()
    kldivloss = nn.KLDivLoss(reduction='mean')
    ssimloss = SSIM(window_size=11)

    # optimizer = AdamW(model.parameters(),lr=args.net_learning_rate)
    optimizer = AdamW([{'params': list(model.module.pretrained.parameters()), 'lr': args.net_learning_rate / 10},
                       {'params': list(model.module.new_added.parameters()), 'lr': args.net_learning_rate}])

    scheduler = CosineAnnealingLR(optimizer, T_max=args.net_num_epoches * len(train_dataloader), eta_min=1e-8)
    # scheduler = ExponentialLR(optimizer, gamma=0.9998)
    # scheduler = StepLR(optimizer, step_size=round(args.net_num_epoches*len(train_dataloader)/100),gamma=0.9)
    best_score = 0
    best_epoch = 0
    best_mode = 'None'

    if args.semi:
        unlabeled_dataloader = iter(cycle(unlabeled_dataloader))

    for ep in range(args.net_num_epoches):
        logger.info("==============Training {}/{} Epoches==============".format(ep, args.net_num_epoches))
        for ii, item in enumerate(train_dataloader):
            optimizer.zero_grad()
            img, anno, img_meta = item
            heatmap = img_meta['heatmaps'].cuda()
            boundary = img_meta['boundaries'].cuda()
            img = img.cuda()
            anno = anno.cuda()

            if args.semi:
                unlabeled_img_w, unlabeled_img_s = next(unlabeled_dataloader)
                unlabeled_img_w = unlabeled_img_w.cuda()
                unlabeled_img_s = unlabeled_img_s.cuda()

            if args.cutmix:
                img, anno, heatmap, boundary = CutMix(args, img, anno, heatmap, boundary)

            if args.net_preclass == True:
                thumbnail = img_meta['thumbnail'].cuda()
                with torch.no_grad():
                    pred_cls = precls_net(thumbnail)
                    pred_cls = torch.argmax(pred_cls, dim=1)
                pred, pred_heat, pred_boundary = model(img, pred_cls)
            else:
                pred, pred_heat, pred_boundary = model(img)

            loss = 0
            loss_dict = {}
            if isinstance(pred, list):
                pred = [F.interpolate(p, img.shape[-2:], mode='bilinear') for p in
                        pred]  # [torch.softmax(F.interpolate(p, img.shape[-2:], mode='bilinear'), dim=1) for p in pred]
                pred_heat = [torch.sigmoid(F.interpolate(p, img.shape[-2:], mode='bilinear')) for p in pred_heat]
                pred_boundary = [torch.sigmoid(F.interpolate(p, img.shape[-2:], mode='bilinear')) for p in
                                 pred_boundary]
                for iii in range(len(pred)):
                    if args.net_celoss:
                        loss_ce = ceLoss(pred[iii], anno.long())
                        loss += loss_ce
                        loss_dict["loss_ce"] = loss_ce
                    if args.net_diceloss:
                        loss_dice = diceloss(pred[iii], anno.long())
                        loss += loss_dice
                        loss_dict["loss_dice"] = loss_dice
                    if args.net_focalloss:
                        loss_focal = focalloss(pred[iii], anno.long())
                        loss += loss_focal
                        loss_dict["loss_focal"] = loss_focal
                    if args.net_regression:
                        loss_mse_heat = mseloss(pred_heat[iii], heatmap.unsqueeze(1)) * 10
                        loss_dict['loss_mse_heat'] = loss_mse_heat
                        loss_mse_boundary = mseloss(pred_boundary[iii], boundary.unsqueeze(1)) * 10
                        loss_dict['loss_mse_boundary'] = loss_mse_boundary
                        loss_ssim_heat = ssimloss(pred_heat[iii], heatmap.unsqueeze(1))
                        loss_dict['loss_ssim_heat'] = loss_ssim_heat
                        loss_ssim_boundary = ssimloss(pred_boundary[iii], boundary.unsqueeze(1))
                        loss_dict['loss_ssim_boundary'] = loss_ssim_boundary
                        loss += loss_mse_heat + loss_mse_boundary + loss_ssim_heat + loss_ssim_boundary
            else:
                pred = F.interpolate(pred, img.shape[-2:],
                                     mode='bilinear')  # torch.softmax(F.interpolate(pred, img.shape[-2:], mode='bilinear'), dim=1)
                pred_heat = torch.sigmoid(F.interpolate(pred_heat, img.shape[-2:], mode='bilinear'))
                pred_boundary = torch.sigmoid(F.interpolate(pred_boundary, img.shape[-2:], mode='bilinear'))
                if args.net_celoss:
                    loss_ce = ceLoss(pred, anno.long())
                    loss += loss_ce
                    loss_dict["loss_ce"] = loss_ce
                if args.net_diceloss:
                    loss_dice = diceloss(pred, anno.long())
                    loss += loss_dice
                    loss_dict["loss_dice"] = loss_dice
                if args.net_focalloss:
                    loss_focal = focalloss(pred, anno.long())
                    loss += loss_focal
                    loss_dict["loss_focal"] = loss_focal
                if args.net_regression:
                    loss_mse_heat = mseloss(pred_heat, heatmap.unsqueeze(1)) * 10
                    loss_dict['loss_mse_heat'] = loss_mse_heat
                    loss_mse_boundary = mseloss(pred_boundary, boundary.unsqueeze(1)) * 10
                    loss_dict['loss_mse_boundary'] = loss_mse_boundary
                    loss_ssim_heat = ssimloss(pred_heat, heatmap.unsqueeze(1))
                    loss_dict['loss_ssim_heat'] = loss_ssim_heat
                    loss_ssim_boundary = ssimloss(pred_boundary, boundary.unsqueeze(1))
                    loss_dict['loss_ssim_boundary'] = loss_ssim_boundary
                    loss += loss_mse_heat + loss_mse_boundary + loss_ssim_heat + loss_ssim_boundary

            if args.semi and ep >= args.warm_up:

                if ep == args.warm_up and ii == 0:
                    model_T.load_state_dict(model.state_dict())
                    print("Create model_T success!")

                with torch.no_grad():
                    un_pred_T, un_pred_heat_T, un_pred_boundary_T = model_T(unlabeled_img_w)
                    un_pred_T_prob = torch.softmax(un_pred_T, dim=1)
                    un_pred_T_pseudo_label = torch.argmax(un_pred_T_prob, dim=1)

                    un_pred_heat_T = torch.sigmoid(
                        F.interpolate(un_pred_heat_T, img.shape[-2:], mode='bilinear'))  # B 1 H W

                    un_pred_boundary_T = torch.sigmoid(
                        F.interpolate(un_pred_boundary_T, img.shape[-2:], mode='bilinear'))  # B1HW

                if args.unlabeled_cutmix:
                    un_pred_T_prob, un_pred_T_pseudo_label, un_pred_heat_T, un_pred_boundary_T, unlabeled_img_s = CutMix_unlabeled(
                        args, un_pred_T_prob,
                        un_pred_T_pseudo_label, un_pred_heat_T, un_pred_boundary_T,
                        unlabeled_img_s)

                un_pred_S, un_pred_heat_S, un_pred_boundary_S = model(unlabeled_img_s)
                un_pred_S_prob = torch.softmax(un_pred_S, dim=1)
                un_pred_heat_S = torch.sigmoid(F.interpolate(un_pred_heat_S, img.shape[-2:], mode='bilinear'))
                un_pred_boundary_S = torch.sigmoid(
                    F.interpolate(un_pred_boundary_S, img.shape[-2:], mode='bilinear'))

                if args.un_pseudo_label:
                    '''
                    total_iter = (args.net_num_epoches - args.warm_up) * len(train_dataloader)
                    cur_iter = (ep - args.warm_up) * len(train_dataloader) + ii
                    prob_threshold = update_prob_threshold(init_thr=args.init_prob_threshold,
                                                           end_thr=args.end_prob_threshold, cur_iter=cur_iter,
                                                           total_iter=total_iter, power=args.power)
                    
                    ceLoss_unsup = nn.CrossEntropyLoss(ignore_index=255, reduction='none')

                    thr_mask = torch.max(un_pred_T_prob, dim=1)[0] > prob_threshold

                    loss_unsup = (1 / (torch.sum(thr_mask) + 1e-5)) * torch.sum(ceLoss_unsup(
                        un_pred_S, un_pred_T_pseudo_label) * thr_mask)
                    loss_dict['prob_thr'] = prob_threshold
                    loss_dict['loss_unsup'] = loss_unsup
                    loss += loss_unsup
                    '''

                    ceLoss_unsup = nn.CrossEntropyLoss(ignore_index=255, reduction='none')
                    un_pred_T_belief = torch.max(un_pred_T_prob, dim=1)[0]  # BHW
                    loss_unsup = (1 / (torch.mean(un_pred_T_belief) + 1e-5)) * torch.mean(
                        ceLoss_unsup(un_pred_S, un_pred_T_pseudo_label) * un_pred_T_belief)
                    loss_dict['loss_unsup'] = loss_unsup
                    loss += loss_unsup

                    if ep >= args.warm_up and ep % 40 == 0 and ii % 5 == 0:
                        for img_index, per_img in enumerate(unlabeled_img_w):  # CHW
                            ori_img_w = np.transpose((per_img.detach().cpu().numpy() * 255).astype(np.uint8),
                                                     (1, 2, 0))
                            ori_img_s = np.transpose(
                                (unlabeled_img_s.detach().cpu().numpy()[img_index] * 255).astype(np.uint8),
                                (1, 2, 0))

                            un_pred_T_pseudo_label_per = (
                                un_pred_T_pseudo_label[img_index]).detach().cpu().numpy()  # HW
                            un_pred_T_pseudo_label_per[un_pred_T_pseudo_label_per == 0] = 0
                            un_pred_T_pseudo_label_per[un_pred_T_pseudo_label_per == 1] = 128
                            un_pred_T_pseudo_label_per[un_pred_T_pseudo_label_per == 2] = 255
                            un_vis_belief_per = np.zeros_like(un_pred_T_pseudo_label_per)

                            un_pred_T_belief_per = (un_pred_T_belief[img_index]).detach().cpu().numpy()  # HW

                            # un_vis_belief_per[un_pred_T_belief_per < 0.5] = 0
                            un_vis_belief_per[un_pred_T_belief_per >= 0.5] = 128
                            un_vis_belief_per[un_pred_T_belief_per >= 0.8] = 255

                            vis_save_path = os.path.join(args.workspace, "vis", f'epoch_{ep}')
                            os.makedirs(vis_save_path, exist_ok=True)
                            save_img_1 = np.concatenate((ori_img_w, ori_img_s), axis=1)

                            un_pred_T_pseudo_label_per = np.repeat(
                                np.expand_dims(un_pred_T_pseudo_label_per, axis=2), 3, axis=2)
                            un_vis_belief_per = np.repeat(np.expand_dims(un_vis_belief_per, axis=2), 3, axis=2)
                            save_img_2 = np.concatenate((un_pred_T_pseudo_label_per, un_vis_belief_per), axis=1)

                            save_img_per = np.concatenate((save_img_1, save_img_2), axis=0)
                            cv2.imwrite(os.path.join(vis_save_path, f"img_{ii}_{img_index}.png"), save_img_per)

                    # teacher_model_para = teacher_model.state_dict()
                    # model_para = model.state_dict()
                    # avg_model_para = ema_update(teacher_model_para, model_para)
                    # teacher_model.load_state_dict(avg_model_para, strict=True)
                    # with torch.no_grad():
                    #     teacher_pred, teacher_pred_heat, teacher_pred_boundary = teacher_model(img)
                    #     teacher_pred = torch.softmax(teacher_pred, dim=1)
                    # if args.net_celoss:
                    #     loss += kldivloss(torch.log(pred), teacher_pred)
                    # if args.net_regression:
                    #     loss += kldivloss(torch.log(pred_heat), teacher_pred_heat)
                    #     loss += kldivloss(torch.log(pred_boundary), teacher_pred_boundary)
                elif args.un_consistency:
                    # loss_unsup_class = mseloss(un_pred_S_prob, un_pred_T_prob)
                    # loss_dict['loss_unsup_class'] = loss_unsup_class

                    # un_pred_heat_T[un_pred_heat_T < 0.5] = 0

                    loss_unsup_heat = 1 * mseloss(un_pred_heat_S, un_pred_heat_T)
                    loss_dict['loss_unsup_heat'] = loss_unsup_heat
                    loss_unsup_boundary = mseloss(un_pred_boundary_S, un_pred_boundary_T)
                    loss_dict['loss_unsup_boundary'] = loss_unsup_boundary
                    # loss += loss_unsup_class + loss_unsup_heat + loss_unsup_boundary
                    loss += loss_unsup_heat + loss_unsup_boundary

            if args.net_mean_teacher == True:
                teacher_model_para = teacher_model.state_dict()
                model_para = model.state_dict()
                avg_model_para = ema_update(teacher_model_para, model_para)
                teacher_model.load_state_dict(avg_model_para, strict=True)
                with torch.no_grad():
                    teacher_pred, teacher_pred_heat, teacher_pred_boundary = teacher_model(img)
                    teacher_pred = F.interpolate(teacher_pred, img.shape[-2:],
                                                 mode='bilinear')  # torch.softmax(F.interpolate(teacher_pred, img.shape[-2:], mode ='bilinear'), dim=1)
                    teacher_pred_heat = torch.sigmoid(F.interpolate(teacher_pred_heat, img.shape[-2:], mode='bilinear'))
                    teacher_pred_boundary = torch.sigmoid(
                        F.interpolate(teacher_pred_boundary, img.shape[-2:], mode='bilinear'))
                if args.net_celoss:
                    # loss_ce_kldiv = kldivloss(torch.log(pred), teacher_pred.detach())
                    loss_mse_ce_tchr = mseloss(pred, teacher_pred.detach())
                    loss_dict['loss_mse_ce_tchr'] = loss_mse_ce_tchr
                    loss += loss_mse_ce_tchr
                if args.net_regression:
                    loss_mse_tchr = mseloss(pred_heat, teacher_pred_heat.detach())
                    loss_dict['loss_mse_tchr'] = loss_mse_tchr
                    loss_mse_boundary_tchr = mseloss(pred_boundary, teacher_pred_boundary.detach())
                    loss_dict['loss_mse_boundary_tchr'] = loss_mse_boundary_tchr
                    loss += loss_mse_tchr + loss_mse_boundary_tchr

            loss.backward()
            optimizer.step()
            scheduler.step()

            if args.semi and ep >= args.warm_up:
                # model_T_past = copy.deepcopy(model_T)
                ema_update(model_T, model, moment=0.995)
                # assert model_T_past.state_dict() != model_T.state_dict(), "Model_T update incorrectly!"

            if ii % 5 == 0:
                logger.info("Epoch:{}/{} || iter:{}/{} || loss:{:.4f} || lr:{}".format(ep, args.net_num_epoches,
                                                                                       ii, len(train_dataloader),
                                                                                       loss,
                                                                                       scheduler.get_last_lr()[0]))
                logger.info("Loss details:{}".format(["{}:{:.8f}".format(k, v) for k, v in loss_dict.items()]))

        model.eval()
        if (ep + 1) % args.val_interval == 0:
            F1_score, heat_F1_score = validate(args, logger, model, val_dataloader)
            if F1_score > best_score:
                best_score = F1_score
                best_epoch = ep
                best_mode = 'Classification'
                torch.save(model.module.state_dict(), os.path.join(args.workspace, 'best_model.pth'))
                if args.semi:
                    torch.save(model_T.module.state_dict(), os.path.join(args.workspace, 'best_model_T.pth'))
                logger.info("Best mode: {}".format(best_mode))
                logger.info("Best model update: Epoch:{}, F1_score:{}, saved to {}".format(best_epoch, best_score,
                                                                                           os.path.join(args.workspace,
                                                                                                        'best_model.pth')))
            if heat_F1_score > best_score:
                best_score = heat_F1_score
                best_epoch = ep
                best_mode = 'Regression'
                torch.save(model.module.state_dict(), os.path.join(args.workspace, 'best_model.pth'))
                if args.semi:
                    torch.save(model_T.module.state_dict(), os.path.join(args.workspace, 'best_model_T.pth'))
                logger.info("Best mode: {}".format(best_mode))
                logger.info("Best model update: Epoch:{}, F1_score:{}, saved to {}".format(best_epoch, best_score,
                                                                                           os.path.join(args.workspace,
                                                                                                        'best_model.pth')))

        if (ep + 1) % args.save_interval == 0:
            torch.save(model.module.state_dict(), os.path.join(args.workspace, 'epoch_{}.pth'.format(ep + 1)))
            logger.info("Checkpoint saved to {}".format(os.path.join(args.workspace, 'epoch_{}.pth'.format(ep + 1))))
            if args.semi:
                torch.save(model_T.module.state_dict(), os.path.join(args.workspace, 'epoch_{}_T.pth'.format(ep + 1)))
                logger.info(
                    "Checkpoint saved to {}".format(os.path.join(args.workspace, 'epoch_{}_T.pth'.format(ep + 1))))

        logger.info("Best mode:{}, Best Epoch:{}, Best F1_score:{}".format(best_mode, best_epoch, best_score))
        model.train()
    torch.save(model.module.state_dict(), os.path.join(args.workspace, 'final.pth'))
    if args.semi:
        torch.save(model_T.module.state_dict(), os.path.join(args.workspace, 'final_T.pth'))
