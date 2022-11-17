import os
import cv2
import numpy as np
import glob
from tqdm import tqdm
import tifffile as tif
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
    # image_paths = glob.glob(os.path.join(image_dir, 'cell_*.png'))
    image_list = os.listdir(image_dir)
    for img_name in tqdm(image_list):
        ii_img_path = os.path.join(image_dir, img_name)
        ii_label_path = os.path.join(label_dir, img_name)
        img = cv2.imread(ii_img_path)
        label = cv2.imread(ii_label_path)
        vis_img = np.hstack((img, label))
        cv2.imwrite(os.path.join(output_dir, img_name), vis_img)

def vis_post_vis(image_dir, label_dir, label_post_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    image_list = os.listdir(image_dir)
    for img_name in tqdm(image_list):
        ii_img_path = os.path.join(image_dir, img_name)
        ii_label_path = os.path.join(label_dir, img_name)
        ii_label_post_path = os.path.join(label_post_dir, img_name.split('.')[0]+'_label.tiff')
        ii_img = cv2.imread(ii_img_path)
        ii_label = cv2.imread(ii_label_path)
        ii_label_post = tif.imread(ii_label_post_path).squeeze().astype(np.uint8)
        ii_label_post[ii_label_post != 0] = 128
        ii_label_post = np.repeat(np.expand_dims(ii_label_post, axis=2), 3, axis=2)
        vis_img = np.hstack((ii_img, ii_label, ii_label_post))
        cv2.imwrite(os.path.join(output_dir, img_name), vis_img)





if __name__ == "__main__":
    # label_dir = '/raid/zby/Cellseg_zby/workspace/extend_unet50_ep500_b24_minib6_crp512_reg_boundary_cutmix/results_test/postprocessed_cls_v2'
    # output_dir = '/raid/zby/Cellseg_zby/workspace/extend_unet50_ep500_b24_minib6_crp512_reg_boundary_cutmix/results_test/postprocessed_cls_v2_vispre'
    # vis_label(label_dir,output_dir)

    # image_dir = '/raid/zby/data/Val_1_3class/images_new'
    # label_dir = '/raid/zby/Cellseg_zby/workspace/extend_unet50_ep500_b24_minib6_crp512_reg_boundary_cutmix/results_test/pred'
    # output_dir = '/raid/zby/Cellseg_zby/workspace/extend_unet50_ep500_b24_minib6_crp512_reg_boundary_cutmix/results_test/vis_all_pred'
    # vis_all(image_dir, label_dir, output_dir)

    # 后处理做完以后的可视化
    image_dir = "/raid/zby/data/Val_1_3class/images_new"
    label_dir = '/raid/zby/Cellseg_zby/workspace/extend_unet50_ep500_b24_minib6_crp512_reg_boundary_cutmix/results_test/pred'
    label_post_dir = '/raid/zby/Cellseg_zby/workspace/extend_unet50_ep500_b24_minib6_crp512_reg_boundary_cutmix/results_test/postprocessed_cls_v2'
    output_dir = '/raid/zby/Cellseg_zby/workspace/extend_unet50_ep500_b24_minib6_crp512_reg_boundary_cutmix/results_test/postprocessed_cls_v2_vis'
    vis_post_vis(image_dir, label_dir, label_post_dir, output_dir)