# -*- coding: utf-8 -*-
"""
# @file name  : split_info_extend_v2_unla_copy.py
# @author     : DJH
# @date       : 2022/10/10 10:34
# @brief      :
"""
import os
import json
source_file = r"C:\Users\djh\Desktop\dataset\split_info_extend_v2.json"
copy_path = r"C:\Users\djh\Desktop\dataset\extends\need_to_copy"
unlabel_path = r"C:\Users\djh\Desktop\dataset\unlabeled_1\images_new"
sf = open(source_file, 'r')
s_data = json.load(sf)
s_data['unlabeled'] = []
for un_file in os.listdir(unlabel_path):
    s_data['unlabeled'].append(un_file.split('.')[0])
for copy_file in os.listdir(copy_path):
    s_data['extend'].append(copy_file.split('.')[0])
with open(os.path.join(os.path.dirname(source_file), 'split_info_extend_v2_unla_copy.json'), 'wb+') as ff:
    json.dump(s_data, ff)