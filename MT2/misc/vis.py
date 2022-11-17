import os
import cv2
import numpy as np
import glob


def vis_label(label_dir, output_dir):
    label_names = os.listdir(label_dir)
    label_paths = [os.path.join(label_dir, label_name) for label_name in label_names]
    os.makedirs(output_dir, exist_ok=True)
    for label_path in label_paths:
        img = cv2.imread(label_path, flags=0)
        img[img == 2] = 255
        img[img == 1] = 128
        output_path = label_path.replace(label_dir, output_dir)
        cv2.imwrite(output_path, img)
    pass


def vis_all(image_dir, label_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    image_paths = glob.glob(os.path.join(image_dir, 'cell_*.png'))
    # image_paths = [os.path.join(image_dir,name) for name in os.listdir(image_dir)]
    label_paths = [image_path.replace(image_dir, label_dir).replace('.png', '_label.png') for image_path in image_paths]
    for ii, image_path in enumerate(image_paths):
        img = cv2.imread(image_path)
        label = cv2.imread(label_paths[ii])
        vis_img = np.hstack((img, label))
        cv2.imwrite(image_path.replace(image_dir, output_dir), vis_img)


if __name__ == "__main__":
    # label_dir = '/home/zby/Cellseg/data/Train_All_3class/labels'
    # output_dir = '/home/zby/Cellseg/data/Train_All_3class/vis'
    # vis_label(label_dir,output_dir)

    image_dir = '/home/zby/Cellseg/data/all_3class/images'
    label_dir = '/home/zby/Cellseg/data/all_3class/vis'
    output_dir = '/home/zby/Cellseg/data/all_3class/vis_all'
    vis_all(image_dir, label_dir, output_dir)
