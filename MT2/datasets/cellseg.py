import os
import sys
from turtle import pos

sys.path.append('/home/zby/Cellseg')
import cv2
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, ColorJitter
from datasets.utils import *
from torchvision import transforms
import random


class CellSeg(Dataset):
    def __init__(self, args, mode=True):
        super(CellSeg, self).__init__()
        # Import the data paths
        self.args = args
        self.image_dir = args.image_dir
        self.anno_dir = args.anno_dir
        self.heatmap_dir = args.heatmap_dir
        self.boundary_dir = args.boundary_dir
        # Extend dataset
        self.extend_image_dir = args.extend_image_dir
        self.extend_anno_dir = args.extend_anno_dir
        self.extend_heatmap_dir = args.extend_heatmap_dir
        self.extend_boundary_dir = args.extend_boundary_dir
        # Test dataset
        self.test_image_dir = args.test_image_dir
        self.test_anno_dir = args.test_anno_dir
        self.split_info = json.load(open(args.split_info))

        if args.semi and mode == 'unlabeled':
            self.unlabeled_image_dir = args.unlabeled_image_dir

        # Initialize the training list
        self.using_unlabel = False
        if mode == 'train':
            image_names = [name + '.png' for name in self.split_info['train']]
            anno_names = [image_name.replace('.png', '_label.png') for image_name in image_names]
            self.img_list = [os.path.join(self.image_dir, image_name) for image_name in image_names]
            self.anno_list = [os.path.join(self.anno_dir, anno_name) for anno_name in anno_names]
            self.heatmap_list = [os.path.join(self.heatmap_dir, anno_name) for anno_name in anno_names]
            self.boundary_list = [os.path.join(self.boundary_dir, anno_name) for anno_name in anno_names]
        elif mode == 'val':
            image_names = [name + '.png' for name in self.split_info['val']]
            anno_names = [image_name.replace('.png', '_label.png') for image_name in image_names]
            self.img_list = [os.path.join(self.image_dir, image_name) for image_name in image_names]
            self.anno_list = [os.path.join(self.anno_dir, anno_name) for anno_name in anno_names]
            self.heatmap_list = [os.path.join(self.heatmap_dir, anno_name) for anno_name in anno_names]
            self.boundary_list = [os.path.join(self.boundary_dir, anno_name) for anno_name in anno_names]
        elif mode == 'test':
            image_names = sorted(os.listdir(self.test_image_dir))
            anno_names = [image_name.replace('.png', '_label.png') for image_name in image_names]
            self.img_list = [os.path.join(self.test_image_dir, image_name) for image_name in image_names]
            self.anno_list = [os.path.join(self.test_anno_dir, anno_name) for anno_name in anno_names]
            self.heatmap_list = [os.path.join(self.heatmap_dir, anno_name) for anno_name in anno_names]
            self.boundary_list = [os.path.join(self.boundary_dir, anno_name) for anno_name in anno_names]
        elif mode == 'all':
            image_names = [name + '.png' for name in self.split_info['train']] + [name + '.png' for name in
                                                                                  self.split_info['val']]
            anno_names = [image_name.replace('.png', '_label.png') for image_name in image_names]
            self.img_list = [os.path.join(self.image_dir, image_name) for image_name in image_names]
            self.anno_list = [os.path.join(self.anno_dir, anno_name) for anno_name in anno_names]
            self.heatmap_list = [os.path.join(self.heatmap_dir, anno_name) for anno_name in anno_names]
            self.boundary_list = [os.path.join(self.boundary_dir, anno_name) for anno_name in anno_names]
        elif mode == 'all_extend':
            image_names = [name + '.png' for name in self.split_info['train']] + [name + '.png' for name in
                                                                                  self.split_info['val']]
            anno_names = [image_name.replace('.png', '_label.png') for image_name in image_names]
            extend_image_names = [name + '.png' for name in self.split_info['extend']]
            extend_anno_names = [extend_image_name.replace('.png', '_label.png') for extend_image_name in
                                 extend_image_names]
            self.img_list = [os.path.join(self.image_dir, image_name) for image_name in image_names] + \
                            [os.path.join(self.extend_image_dir, extend_image_name) for extend_image_name in
                             extend_image_names]
            self.anno_list = [os.path.join(self.anno_dir, anno_name) for anno_name in anno_names] + \
                             [os.path.join(self.extend_anno_dir, extend_anno_name) for extend_anno_name in
                              extend_anno_names]
            self.heatmap_list = [os.path.join(self.heatmap_dir, anno_name) for anno_name in anno_names] + \
                                [os.path.join(self.extend_heatmap_dir, extend_anno_name) for extend_anno_name in
                                 extend_anno_names]
            self.boundary_list = [os.path.join(self.boundary_dir, anno_name) for anno_name in anno_names] + \
                                 [os.path.join(self.extend_boundary_dir, extend_anno_name) for extend_anno_name in
                                  extend_anno_names]
        elif mode == 'unlabeled':
            image_names = [name + '.png' for name in self.split_info['unlabeled']]
            anno_names = [image_name.replace('.png', '_label.png') for image_name in image_names]
            self.img_list = [os.path.join(self.unlabeled_image_dir, image_name) for image_name in image_names]
            self.anno_list = [os.path.join(self.unlabeled_image_dir, anno_name) for anno_name in anno_names]
            self.heatmap_list = [os.path.join(self.unlabeled_image_dir, anno_name) for anno_name in anno_names]
            self.boundary_list = [os.path.join(self.unlabeled_image_dir, anno_name) for anno_name in anno_names]
        else:
            raise NotImplementedError()
        # generate the random sample weight

        if args.sampler and mode == 'all_extend':
            img_weights = [1.0] * len(image_names)
            extend_weights = [1.0] * len(extend_image_names)

            for i, image_name in enumerate(image_names):
                if image_name.split(".")[0] in self.split_info['3Channel']:
                    img_weights[i] = 2.0
                    # print("bingli")
                # elif image_name.startswith("tissue"):
                #     img_weights[i] = 0.3
                # else:
                #     pass
            for i, extend_image_name in enumerate(extend_image_names):
                if extend_image_name.startswith("tissue"):
                    extend_weights[i] = 0.3

            self.weights = img_weights + extend_weights

            # extend_weights = []
            # for extend_image_name in extend_image_names:
            #     if 'tissue' in extend_image_name:
            #         weight = 0.05
            #     else:
            #         weight = 1.0
            #     extend_weights.append(weight)
            # self.weights = img_weights+extend_weights

        # Initialize the pre-processing setting
        self.mode = mode
        self.scale_range = args.scale_range
        self.crop_size = args.crop_size
        self.flip = args.rand_flip
        self.rotate = args.rand_rotate
        self.bright = args.rand_bright
        self.contrast = args.rand_contrast
        self.saturation = args.rand_saturation
        self.hue = args.rand_hue
        self.test_multi_scale = args.test_multi_scale
        self.transform = ToTensor()

        print(f"{mode}: {len(self.img_list)}")

    def __getitem__(self, index):
        img_path = self.img_list[index]
        anno_path = self.anno_list[index]
        heatmap_path = self.heatmap_list[index]
        boundary_path = self.boundary_list[index]
        img = cv2.imread(img_path)
        # try:
        thumbnail = cv2.resize(img, self.crop_size, interpolation=cv2.INTER_LINEAR)
        thumbnail = self.transform(thumbnail)
        # except:
        #     print(img_path)
        #     raise Exception("rerere")
        if os.path.exists(anno_path):
            anno = cv2.imread(anno_path, flags=0)
            valid_label = 1
        else:
            anno = np.zeros_like(img[:, :, 0])
            valid_label = 0

        if os.path.exists(heatmap_path):
            heatmap = cv2.imread(heatmap_path, flags=0)
        else:
            heatmap = np.zeros_like(img[:, :, 0])

        if os.path.exists(boundary_path):
            boundary = cv2.imread(boundary_path, flags=0)
        else:
            boundary = np.zeros_like(img[:, :, 0])

        img_meta = {'img_path': img_path, 'valid_label': valid_label, 'ori_shape': img.shape, 'thumbnail': thumbnail}

        if self.mode in ['train', 'all', 'all_extend']:
            if self.scale_range:
                # img, anno, heatmap, boundary = random_scale(img, anno, heatmap, boundary, self.scale_range, self.crop_size)
                img, anno, heatmap, boundary = random_scale_v2(img, anno, heatmap, boundary, self.scale_range,
                                                               self.crop_size)
            if self.crop_size:
                img, anno, heatmap, boundary = random_crop(img, anno, heatmap, boundary, self.crop_size)
                # img, anno, heatmap, boundary = random_crop_v2(img, anno, heatmap, boundary, self.crop_size)
            if self.flip:
                img, anno, heatmap, boundary = random_flip(img, anno, heatmap, boundary, self.flip)
            if self.rotate:
                img, anno, heatmap, boundary = random_rotate(img, anno, heatmap, boundary)

            img = self.transform(img)
            if self.bright:
                img = ColorJitter(brightness=self.bright)(img)
            if self.contrast:
                img = ColorJitter(contrast=self.contrast)(img)
            if self.saturation:
                img = ColorJitter(saturation=self.saturation)(img)
            if self.hue:
                img = ColorJitter(hue=self.hue)
            anno = torch.tensor(anno)
            heatmap = torch.tensor(heatmap) / 255
            boundary = torch.tensor(boundary) / 255
        elif self.mode == 'unlabeled':
            img_w = img.copy()
            anno_w = anno.copy()
            heatmap_w = heatmap.copy()
            boundary_w = boundary.copy()

            if self.scale_range:
                img_w, anno_w, heatmap_w, boundary_w = random_scale_v2(img_w, anno_w, heatmap_w, boundary_w,
                                                                       self.scale_range,
                                                                       self.crop_size)
            if self.crop_size:
                img_w, anno_w, heatmap_w, boundary_w = random_crop(img_w, anno_w, heatmap_w, boundary_w, self.crop_size)
            if self.flip:
                img_w, anno_w, heatmap_w, boundary_w = random_flip(img_w, anno_w, heatmap_w, boundary_w,
                                                                   self.flip)
            if self.rotate:
                img_w, anno_w, heatmap_w, boundary_w = random_rotate(img_w, anno_w, heatmap_w, boundary_w)

            img_s = np.transpose(img_w.copy(), (2, 0, 1))
            img_s = torch.tensor(img_s)
            # print(img_s.shape)

            img_w = self.transform(img_w)
            # anno_w = torch.tensor(anno_w)
            # heatmap_w = torch.tensor(heatmap_w) / 255
            # boundary_w = torch.tensor(boundary_w) / 255

            if random.random() <= self.args.unlabeled_color_jittor_prob:  # color jittor
                img_s = transforms.ColorJitter(0.1, 0.1, 0, 0)(img_s)

            # random gray
            if self.args.unlabeled_gray_scale:
                img_s = transforms.RandomGrayscale(p=0.2)(img_s)

            if random.random() <= self.args.unlabeled_Gaussian_blur_prob:
                kernel_size = random.choice([3, 5])
                img_s = transforms.GaussianBlur(kernel_size, sigma=(0.1, 2.0))(img_s)

            img_s = np.transpose(img_s.numpy(), (1, 2, 0))  # HWC
            img_s = self.transform(img_s)

            return img_w, img_s

        elif self.mode in ['val']:
            # img, anno, heatmap, boundary = rescale_to_min(img, anno, heatmap, boundary, self.crop_size)
            img, anno, heatmap, boundary, valid_region = multi_scale_test_v2(img, anno, heatmap, boundary, scale=[1.0],
                                                                             crop_size=self.crop_size)
            img_meta['valid_region'] = valid_region[0]
            img = self.transform(img[0])
            anno = torch.tensor(anno[0])
            heatmap = torch.tensor(heatmap[0]) / 255
            boundary = torch.tensor(boundary[0]) / 255
        else:
            # img, anno, heatmap, boundary = rescale_to_min(img, anno, heatmap, boundary, self.crop_size)
            # img, anno, heatmap, boundary = multi_scale_test(img, anno, heatmap, boundary, scale=self.test_multi_scale)
            img, anno, heatmap, boundary, valid_region = multi_scale_test_v2(img, anno, heatmap, boundary,
                                                                             scale=self.test_multi_scale,
                                                                             crop_size=self.crop_size)
            img_meta['valid_region'] = valid_region
            if isinstance(img, list):
                img = [self.transform(i) for i in img]
                anno = [torch.tensor(a) for a in anno]
                heatmap = [torch.tensor(h) / 255 for h in heatmap]
                boundary = [torch.tensor(b) / 255 for b in boundary]
            else:
                img = self.transform(img)
                anno = torch.tensor(anno)
                heatmap = torch.tensor(heatmap) / 255
                boundary = torch.tensor(boundary) / 255
        img_meta['heatmaps'] = heatmap
        img_meta['boundaries'] = boundary
        return img, anno, img_meta

    def use_unlabel(self):
        assert self.train == True
        self.using_unlabel = True
        self.img_list = self.image_paths + self.un_image_paths
        self.anno_list = self.anno_paths + self.un_anno_paths

    def no_unlabel(self):
        assert self.train == True
        self.using_unlabel = False
        self.img_list = self.image_paths
        self.anno_list = self.anno_paths

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
