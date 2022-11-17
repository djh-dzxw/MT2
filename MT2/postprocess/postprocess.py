import os
import cv2
import numpy as np
import tifffile as tif
from skimage import measure, color, morphology

def postprocess(img_dir, heat_dir, output_dir):
    threshold = 0.3
    heat_list = sorted(os.listdir(heat_dir))
    heat_paths = [os.path.join(heat_dir,heat_name) for heat_name in heat_list]
    for ii,heat_path in enumerate(heat_paths):
        print("Processing the {}/{} images....".format(ii,len(heat_paths)),end='\r')
        img_path = heat_path.replace(heat_dir,img_dir)
        heatmap = cv2.imread(heat_path,flags=0)/255
        heatmap[heatmap>=threshold]=1
        heatmap[heatmap<threshold]=0
        heatmap = measure.label(heatmap,background=0)
        H,W = heatmap.shape
        if H>5000 or W>5000:
            new_inst_map = heatmap
        else:
            inst_num = np.max(heatmap)
            new_inst_map = np.zeros_like(heatmap)
            new_inst_num = 0
            for inst_idx in range(1,inst_num):
                print("Reg: Processing the {}/{} images... {}/{} instances...".format(ii,len(heat_paths), inst_idx, inst_num), end='\r')
                if np.sum(heatmap==inst_idx)<16:
                    continue
                new_inst_num += 1
                temp_map = heatmap==inst_idx
                temp_map = temp_map*1
                temp_map = morphology.binary_dilation(temp_map,morphology.disk(1))*1
                new_inst_map[temp_map==1] = new_inst_num
        tif.imwrite(os.path.join(output_dir,heat_path.replace(heat_dir, output_dir).replace('.png','_label.tiff')), new_inst_map, compression='zlib')
    pass


def postprocess_cls(img_dir, heat_dir, output_dir):
    threshold = 0.3
    heat_list = sorted(os.listdir(heat_dir))
    heat_paths = [os.path.join(heat_dir,heat_name) for heat_name in heat_list]
    for ii,heat_path in enumerate(heat_paths):
        print("Processing the {}/{} images....".format(ii,len(heat_paths)),end='\r')
        img_path = heat_path.replace(heat_dir,img_dir)
        heatmap = cv2.imread(heat_path,flags=0)
        heatmap[heatmap==128]=1
        heatmap[heatmap==255]=0
        heatmap = measure.label(heatmap,background=0)
        H,W = heatmap.shape
        if H>5000 or W>5000:
            new_inst_map = heatmap
        else:
            inst_num = np.max(heatmap)
            new_inst_map = np.zeros_like(heatmap)
            new_inst_num = 0
            for inst_idx in range(1,inst_num):
                print("Cls: Processing the {}/{} images... {}/{} instances...".format(ii,len(heat_paths), inst_idx, inst_num), end='\r')
                if np.sum(heatmap==inst_idx)<16:
                    continue
                new_inst_num += 1
                temp_map = heatmap==inst_idx
                temp_map = temp_map*1
                temp_map = morphology.binary_dilation(temp_map,morphology.disk(1))*1
                new_inst_map[temp_map==1] = new_inst_num
        tif.imwrite(os.path.join(output_dir,heat_path.replace(heat_dir, output_dir).replace('.png','_label.tiff')), new_inst_map, compression='zlib')
    pass

if __name__=="__main__":
    img_dir = '/home/zby/Cellseg/data/fix_boundary/images_new'
    heat_dir = '/home/zby/Cellseg/workspace/extend_beit_ep500_b18_crp512_reg_boundary/results_test/heat_pred'
    output_dir = heat_dir.replace('heat_pred', 'postprocessed_reg')
    os.makedirs(output_dir,exist_ok=True)
    postprocess(img_dir, heat_dir, output_dir)

    img_dir = '/home/zby/Cellseg/data/fix_boundary/images_new'
    heat_dir = '/home/zby/Cellseg/workspace/extend_beit_ep500_b18_crp512_reg_boundary/results_test/pred'
    output_dir = heat_dir.replace('pred', 'postprocessed_cls')
    os.makedirs(output_dir,exist_ok=True)
    postprocess_cls(img_dir, heat_dir, output_dir)