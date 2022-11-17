import os
import argparse
# from utils.logger import Logger
from test import test
from test_v2 import test_v2


def str2bool(v):
    if v == "True":
        return True
    else:
        return False


def run():
    parser = argparse.ArgumentParser("CellSeg training argument parser.")
    parser.add_argument('--test_image_dir', default='./inputs')
    parser.add_argument('--crop_size', default=[512,512], type=tuple)
    parser.add_argument('--net_num_classes', default=3, type=int)
    parser.add_argument('--infer_stride', default=(256, 256), type=tuple)
    parser.add_argument('--test_multi_scale', default=[0.8, 1.0, 1.2], type=list)
    parser.add_argument('--test_fusion', default='max', type=str)
    parser.add_argument('--output_dir', default='./outputs', type=str)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    test(args)
    # test_v2(args)
    return


if __name__ == "__main__":
    run()
