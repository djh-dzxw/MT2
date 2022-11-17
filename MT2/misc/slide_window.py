import os
import sys
import cv2

sys.path.append('/home/zby/Cellseg')
from datasets.utils import slide_window


def crop_val_test_data(data_dir, out_dir):
    image_names = os.listdir(os.path.join(data_dir, 'images'))
    anno_names = [image_name.replace('.png', '_label.png') for image_name in image_names]
    image_paths = [os.path.join(data_dir, 'images', image_name) for image_name in image_names]
    anno_paths = [os.path.join(data_dir, 'labels', anno_name) for anno_name in anno_names]
    for ii in range(len(image_paths)):
        img = cv2.imread(image_paths[ii])
        anno = cv2.imread(anno_paths[ii], flags=0)
        img_list, anno_list, pos_list = slide_window(img, anno, (512, 512))
        for idx in range(len(img_list)):
            cv2.imwrite(
                os.path.join(
                    out_dir, 'images', image_names[idx].replace('.png',
                                                                '_{}_{}.png'.format(pos_list[idx][0], pos_list[idx][1])
                                                                )
                ),
                img_list[idx]
            )
            cv2.imwrite(
                os.path.join(
                    out_dir, 'labels', anno_names[idx].replace('_label.png',
                                                               '_{}_{}_label.png'.format(pos_list[idx][0],
                                                                                         pos_list[idx][1])
                                                               )
                ),
                anno_list[idx]
            )
    pass


if __name__ == "__main__":
    data_dir = '/home/zby/Cellseg/data/Val_Labeled_3class'
    out_dir = '/home/zby/Cellseg/data/Val_Labeled_3class_crp512_s384'

    os.makedirs(out_dir + '/labels/', exist_ok=True)
    os.makedirs(out_dir + '/images/', exist_ok=True)
    crop_val_test_data(data_dir, out_dir)
