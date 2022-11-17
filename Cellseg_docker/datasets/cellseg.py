import os
import sys

# from turtle import pos
sys.path.append('/home/zby/Cellseg')
# import cv2
# import json
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, ColorJitter
from datasets.utils import *
import tifffile as tif
from skimage import io, exposure


def normalize_channel(img, lower=1, upper=99):
    non_zero_vals = img[np.nonzero(img)]
    percentiles = np.percentile(non_zero_vals, [lower, upper])
    if percentiles[1] - percentiles[0] > 0.001:
        img_norm = exposure.rescale_intensity(img, in_range=(percentiles[0], percentiles[1]), out_range='uint8')
    else:
        img_norm = img
    return img_norm.astype(np.uint8)


class CellSeg(Dataset):
    def __init__(self, args, mode=True, preprocess_flag=False):
        super(CellSeg, self).__init__()
        self.preprocess_flag = preprocess_flag
        # Test dataset
        self.test_image_dir = args.test_image_dir
        image_names = sorted(os.listdir(self.test_image_dir))
        self.img_list = [os.path.join(self.test_image_dir, image_name) for image_name in image_names]
        # Initialize the pre-processing setting
        self.mode = mode
        self.crop_size = args.crop_size
        self.test_multi_scale = args.test_multi_scale
        self.transform = ToTensor()

    def __getitem__(self, index):
        img_path = self.img_list[index]
        img_flag = False
        if self.preprocess_flag:
            if img_path.endswith('.tif') or img_path.endswith('.tiff'):
                img_data = tif.imread(img_path)
            else:
                img_data = io.imread(img_path)
            pre_img_data = normalize_channel(img_data, lower=1, upper=99)
            if len(pre_img_data.shape) == 2:
                pre_img_data = np.repeat(np.expand_dims(pre_img_data, axis=-1), 3, axis=-1)
            elif len(pre_img_data.shape) == 3 and pre_img_data.shape[-1] > 3:
                pre_img_data = pre_img_data[:, :, :3]
            else:
                pass
            img = pre_img_data.astype(np.uint8)  # HW3
            img = img[:, :, [2, 1, 0]]  # BGR
        else:
            img = cv2.imread(img_path)  # BGR

        if np.sum(img[:, :, 2]) == 0 and img.shape[0] < 5000 and img.shape[1] < 5000:
            img_flag = True

        img_meta = {'img_path': img_path, 'ori_shape': img.shape, 'img_flag': img_flag}
        img, valid_region = multi_scale_test_v2(img, scale=[1.5] if img_flag else self.test_multi_scale,
                                                crop_size=self.crop_size)
        img_meta['valid_region'] = valid_region
        if isinstance(img, list):
            img = [self.transform(i) for i in img]
        else:
            img = self.transform(img)
        return img, img_meta

    def __len__(self):
        return len(self.img_list)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("CellSeg training argument parser.")
    parser.add_argument('--image_dir', default='./data/fix_boundary/images')
    parser.add_argument('--anno_dir', default='./data/fix_boundary/labels')
    parser.add_argument('--test_image_dir', default='./data/Val_Labeled_3class/images')
    parser.add_argument('--test_anno_dir', default='./data/Val_Labeled_3class/labels')
    parser.add_argument('--split_info', default='/home/zby/Cellseg/data/split_info.json')

    parser.add_argument('--scale_range', default=(0.5, 2.0))
    parser.add_argument('--crop_size', default=(512, 512))
    parser.add_argument('--rand_flip', default=0.5, help="Horizonal and Vertical filp, 0 for unchange")
    parser.add_argument('--rand_rotate', default=False, type=bool)
    args = parser.parse_args()

    train_dataset = CellSeg(args, mode='train')
    val_dataset = CellSeg(args, mode='val')

    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=1)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)
    for ii, item in enumerate(train_dataloader):
        print("The {}/{} batches...".format(ii, len(train_dataloader)), end='\r')
    for ii, item in enumerate(val_dataloader):
        print("The {}/{} batches...".format(ii, len(val_dataloader)), end='\r')
