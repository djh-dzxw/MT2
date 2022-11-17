import os
import argparse
from utils.logger import Logger
from train import train
from test import test
from infer import infer
from train_final import train as train_final


def str2bool(v):
    if v == "True":
        return True
    else:
        return False


def run():
    parser = argparse.ArgumentParser("CellSeg training argument parser.")
    parser.add_argument('--image_dir', default='/raid/zby/data/official/images_new')
    parser.add_argument('--anno_dir', default='/raid/zby/data/official/labels')
    parser.add_argument('--gt_dir', default='/raid/zby/data/official/gts')
    parser.add_argument('--distmap_dir', default='/raid/zby/data/official/dist_maps')
    parser.add_argument('--heatmap_dir', default='/raid/zby/data/official/heatmaps')
    parser.add_argument('--boundary_dir', default='/raid/zby/data/official/boundary_maps')

    parser.add_argument('--extend_image_dir', default='/raid/zby/data/extended/images')
    parser.add_argument('--extend_anno_dir', default='/raid/zby/data/extended/labels')
    parser.add_argument('--extend_gt_dir', default='/raid/zby/data/extended/gts')
    parser.add_argument('--extend_distmap_dir', default='/raid/zby/data/extended/dist_maps')
    parser.add_argument('--extend_heatmap_dir', default='/raid/zby/data/extended/heatmaps')
    parser.add_argument('--extend_boundary_dir', default='/raid/zby/data/extended/boundary_maps')

    parser.add_argument('--test_image_dir',
                        default='/raid/zby/data/Val_1_3class/images_new')
    parser.add_argument('--test_anno_dir', default='/raid/zby/data/Val_1_3class/labels')
    parser.add_argument('--split_info', default='./datasets/split_info_extend_v4.json')

    parser.add_argument('--batch_size', default=24, type=int)
    parser.add_argument('--num_worker', default=4, type=int)

    parser.add_argument('--scale_range', default=(0.5, 2.0))
    parser.add_argument('--crop_size', default=(512, 512))
    parser.add_argument('--rand_flip', default=0.5, type=float)
    parser.add_argument('--rand_rotate', default=True, type=bool)
    parser.add_argument('--rand_bright', default=0.5, type=float)
    parser.add_argument('--rand_contrast', default=0.5, type=float)
    parser.add_argument('--rand_saturation', default=0.5, type=float)
    parser.add_argument('--rand_hue', default=0, type=float)
    parser.add_argument('--cutmix', default=True, type=bool)
    parser.add_argument('--beta', default=1.0, type=float)
    parser.add_argument('--cutmix_prob', default=0.5, type=float)

    # semi
    parser.add_argument('--semi', default=True, type=bool)
    parser.add_argument('--unlabeled_image_dir', default='/raid/zby/data/official_unlabeled_1/images_new')
    parser.add_argument('--unlabeled_color_jittor_prob', default=0.5, type=float)
    parser.add_argument('--unlabeled_Gaussian_blur_prob', default=0., type=float)
    parser.add_argument('--unlabeled_gray_scale', default=False, type=bool)
    parser.add_argument('--unlabeled_batch_size', default=24, type=int)
    parser.add_argument('--warm_up', default=100, type=int)
    parser.add_argument('--unlabeled_cutmix', default=True, type=bool)
    parser.add_argument('--unlabeled_cutmix_prob', default=0.5, type=float)
    parser.add_argument('--un_pseudo_label', default=False, type=bool)
    parser.add_argument('--un_consistency', default=True, type=bool)

    # parser.add_argument('--init_prob_threshold', default=0.7, type=float)
    # parser.add_argument('--end_prob_threshold', default=0.5, type=float)
    # parser.add_argument('--power', default=0.9, type=float)

    parser.add_argument('--sampler', default=False, type=bool)
    parser.add_argument('--num_samples', default=5000, type=int)

    parser.add_argument('--net_preclass', default=False, type=str2bool)
    parser.add_argument('--net_preclass_checkpoint', default='./temp/classifer_all.pth', type=str)
    parser.add_argument('--net_name', default='unetplusplus', type=str)
    parser.add_argument('--net_stride', default=16, type=int)
    parser.add_argument('--net_num_classes', default=3, type=int)
    parser.add_argument('--net_num_epoches', default=500, type=int)
    parser.add_argument('--net_learning_rate', default=1e-3, type=float)
    parser.add_argument('--net_mean_teacher', default=False, type=bool)
    parser.add_argument('--net_celoss', default=False, type=str2bool)
    parser.add_argument('--net_diceloss', default=False, type=str2bool)
    parser.add_argument('--net_focalloss', default=False, type=str2bool)
    parser.add_argument('--net_regression', default=False, type=str2bool)
    parser.add_argument('--train_mode', default='all_extend', type=str)
    parser.add_argument('--train_resume', default=None, type=str)
    # parser.add_argument('--train_resume', default='/raid/zby/Cellseg_v2/workspace/extend_all_unetpp50_ep500_b12_minib6_crp512_reg_boundary_semi_t3/final.pth', type=str)
    # parser.add_argument('--train_resume_T',
    #                     default='/raid/zby/Cellseg_v2/workspace/extend_all_unetpp50_ep500_b12_minib6_crp512_reg_boundary_semi_t3/final_T.pth',
    #                     type=str)
    parser.add_argument('--infer_stride', default=(256, 256), type=tuple)
    parser.add_argument('--infer_threshold', default=0.5, type=float)
    parser.add_argument('--test_threshold', default=0.3, type=float)
    parser.add_argument('--test_multi_scale', default=[0.8, 1.0, 1.2], type=list)
    parser.add_argument('--test_fusion', default='max', type=str)

    parser.add_argument('--workspace', default='./workspace/debug', type=str)
    parser.add_argument('--results_val', default='results_val', type=str)
    parser.add_argument('--results_test', default='results_test', type=str)
    parser.add_argument('--checkpoint', default='final.pth', type=str)
    parser.add_argument('--val_interval', default=50, type=int)
    parser.add_argument('--save_interval', default=50, type=int)

    parser.add_argument("--train_pass", default=False, type=str2bool)
    parser.add_argument("--infer_pass", default=False, type=str2bool)
    parser.add_argument("--test_pass", default=False, type=str2bool)
    parser.add_argument("--train_final_pass", default=False, type=str2bool)

    args = parser.parse_args()

    os.makedirs(args.workspace, exist_ok=True)
    os.makedirs(os.path.join(args.workspace, args.results_val), exist_ok=True)
    os.makedirs(os.path.join(args.workspace, args.results_test), exist_ok=True)
    logger = Logger(args)
    logger.info(args)

    if args.train_pass == True:
        logger.info("Starting training pass....")
        train(args, logger)

    if args.train_final_pass == True:
        logger.info("Starting training_final pass....")
        train_final(args, logger)

    if args.infer_pass == True:
        logger.info("Starting inferring pass....")
        infer(args, logger)

    if args.test_pass == True:
        logger.info("Starting testing pass....")
        test(args, logger)

    return


if __name__ == "__main__":
    run()
