import os
import json


def gen_cls_split():
    ori_split_json = json.load(open('/raid/zby/data/split_info.json', 'r'))
    img_dir = '/raid/zby/data/extended/images'
    extend_dict = {'extend': []}
    for img_name in os.listdir(img_dir):
        if img_name.startswith("omnipose"):
            img_name = img_name.split('.')[0]
            extend_dict['extend'].append(img_name)
    ori_split_json['extend'] = sorted(extend_dict['extend'])
    json.dump(ori_split_json, open('/raid/zby/data/split_info_extend.json', 'w'), indent=2)
    pass


if __name__ == "__main__":
    gen_cls_split()
