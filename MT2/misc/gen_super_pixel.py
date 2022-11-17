from skimage.segmentation import slic, mark_boundaries
from skimage import io
import numpy as np
import cv2
import os
import tifffile as tif


def gen_super_pixel(img_dir, output_dir):
    img_list = sorted(os.listdir(img_dir))
    for ii, img_name in enumerate(img_list):
        print("Processing the {}/{} images...".format(ii, len(img_list)), end='\r')
        img_path = os.path.join(img_dir, img_name)
        img = io.imread(img_path)
        seg = slic(img, n_segments=10000, compactness=10)
        save_path = os.path.join(output_dir, img_name)
        vis = mark_boundaries(img, seg)
        tif.imwrite(save_path.replace('.png', '.tif'), seg)
        cv2.imwrite(save_path.replace('.png', '_vis.png'), vis * 255)


if __name__ == "__main__":
    img_dir = '/home/zby/Cellseg/data/fix_boundary/images_new'
    output_dir = '/home/zby/Cellseg/data/fix_boundary/super_pixel'
    os.makedirs(output_dir, exist_ok=True)
    gen_super_pixel(img_dir, output_dir)

    # img_path = '/home/zby/Cellseg/data/fix_boundary/images_new/cell_00001.png'
    # img = io.imread(img_path)
    # segments = slic(img, n_segments=60, compactness=10)
    # out=mark_boundaries(img,segments)
    # cv2.imwrite("./test1.jpg", out*255)

    # segments2 = slic(img, n_segments=300, compactness=10)
    # out2=mark_boundaries(img,segments2)
    # cv2.imwrite('./test2.jpg', out2*255)
