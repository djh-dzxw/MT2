# MT2: Multi-task Mean Teacher for Semi-Supervised Cell Segmentation

## Environments and Requirements

- Ubuntu 18.04.4 LTS
- CPU Intel(R) Xeon(R) CPU E5-2698 v4 @ 2.20GHz
- GPU 4x NVIDIA Tesla V100 32G
- CUDA 10.2
- python 3.7

To install requirements:

```setup
cd MT2
pip install -r requirements.txt
```

## Dataset

* [Official Dataset ](https://neurips22-cellseg.grand-challenge.org/dataset/): Training set: 1000 labeled image patches from various microscopy types, tissue types, and staining types, and 1500+ unlabeled images. Validation/Tuning set: 101 images

* External Datasets: [Omnipose](http://www.cellpose.org/dataset_omnipose), [Livecell](https://github.com/sartorius-research/LIVECell), [Tiussenet](https://datasets.deepcell.org/)   
* Dataset Format

```
.
├── extended     # External Datasets
│   ├── boundary_maps  # boudary maps gts
│   │   ├── A172_Phase_A7_1_00d00h00m_1_label.png
│   │   ├── ...
│   ├── dist_maps  # not used, can be ignored
│   │   ├── A172_Phase_A7_1_00d00h00m_1_label.png
│   │   ├── ...
│   ├── heatmaps  # heatmaps gts
│   │   ├── A172_Phase_A7_1_00d00h00m_1_label.png
│   │   ├── ...
│   ├── images  # preprocessed images
│   │   ├── A172_Phase_A7_1_00d00h00m_1.png
│   │   ├── ...
│   └── labels  # preprocessed labels
│       ├── A172_Phase_A7_1_00d00h00m_1_label.png
│       ├── ...
├── official  # labeled data of Formal Dataset 
│   ├── boundary_maps  # boudary maps gts
│   │   ├── cell_00001_label.png
│   │   ├── ...
│   ├── heatmaps  # heatmaps gts
│   │   ├── cell_00001_label.png
│   │   ├── ...
│   ├── images_new  # preprocessed images
│   │   ├── cell_00001.png
│   │   ├── ...
│   └── labels  # preprocessed labels
│       ├── cell_00001_label.png
│       ├── ...
├── official_unlabeled_1  # unlabeled data of Formal Dataset 
│   └── images_new  # preprocessed images
│       ├── unlabeled_cell_00000.png
│       ├── ...
└── Val_1_3class  # Tuning Set
    └──images_new  # preprocessed images
        ├── cell_00001.png
        └──  ...
```

## Preprocessing

We only provide the preprocessing method here. More details about data augmentations, e.g., random scale, cropping, colorjitters can be found in cellseg.py.

- All the images are transferred into 3 channels.
- The pixel values in the images are normalized into [0, 255].
- Generate three-class maps with instance maps.

Next, we demonstrate how to generate the dataset form.

1. Preprocess the raw images and labels to generate file *images/images_new* and *labels* above. After download the raw datasets with images and instance labels. You can modify *misc/preprocess_dataset.py* and run it with ease to obtain the preprocessed results.

```
python preprocess_dataset.py
```

2. Generate boundary maps and heatmaps. You can run *misc/gen_heat_map.py* with minor modifications.

```
python gen_heat_map.py
```

## Training

To train the model in the paper, run this command:

```train
cd MT2
CUDA_VISIBLE_DEVICES=0,1,2,3 python run.py --train_mode all_extend --train_pass True --test_pass True --workspace ./workspace/extend_all_unetpp50_ep500_b24_minib6_crp512_reg_boundary_semi_t_final/ --batch_size 24 --net_regression True --net_celoss True --net_learning_rate 0.001 --semi True --unlabeled_batch_size 24 --net_diceloss True
```

Certainly, you can modify the parameters here or in *run.py*.

## Trained Models

You can download trained models here:

- Link: https://pan.baidu.com/s/16PQwNz7MTzlhP7KWZMNPFg      Password: cj6t

## Inference

To infer the testing cases, run this command:

```python
cp final.pth Cellseg_docker
cd Cellseg_docker
python run.py --test_image_dir <your_input_raw_images_dir>  --output_dir <output_label_dir>
```

## Evaluation

We have embedded evaluation to training process. You only need to set up the validation interval (*val_interval*) and provide the path to validation set (*test_image_dir* and *test_anno_dir*) to obtain the validation results during training. 

## Tuning Set Results

Our method achieves the following performance on [NeurIPS2022 Cellseg Competition](https://neurips22-cellseg.grand-challenge.org/evaluation/challenge/leaderboard/).

| Model name / Team name                     | F1 Score |         Rank          |
| ------------------------------------------ | :------: | :-------------------: |
| Multi-task Mean Teacher (MT2) / BUPT-MCPRL |  0.8690  | 7th（Team Rank: 4th） |

## Acknowledgement

We thank the organizers of NeurIPS2022 Cellseg Competition and the contributors of public datasets. 
