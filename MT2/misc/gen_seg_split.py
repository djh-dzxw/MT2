import json
import os


def gen_seg_split():
    original_split = json.load(open('/home/zby/Cellseg/data/split_info_extend.json', 'r'))
    unlabeled_split = {'unlabeled_brightfield': [],
                       'unlabeled_fluorescent': [],
                       'unlabeled_else': []}
    img_names = os.listdir('/home/zby/Cellseg/data/original/Train-Unlabeled/release-part1')
    for img_name in img_names:
        if img_name.endswith('.png') and int(img_name.split('_')[-1].replace('.png', '')) < 1635:
            unlabeled_split['unlabeled_brightfield'].append(img_name.replace('.png', ''))
        elif img_name.endswith('.png') and int(img_name.split('_')[-1].replace('.png', '')) > 1635:
            unlabeled_split['unlabeled_fluorescent'].append(img_name.replace('.png', ''))
        else:
            unlabeled_split['unlabeled_else'].append(img_name.replace('.tif', ''))
    for k, v in unlabeled_split.items():
        original_split[k] = v
    json.dump(original_split, open('/home/zby/Cellseg/data/split_info_extend_unlabel.json', 'w'), indent=2)


if __name__ == "__main__":
    gen_seg_split()
