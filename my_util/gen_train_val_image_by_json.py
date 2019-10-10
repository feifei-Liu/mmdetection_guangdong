import os
import json
import shutil

t_json = '/home/remo/Desktop/cloth_flaw_detection/Dataset/COCO_format/instances_train_coco.json'
v_json = '/home/remo/Desktop/cloth_flaw_detection/Dataset/COCO_format/instances_val_coco.json'
t_path = '/home/remo/Desktop/cloth_flaw_detection/Dataset/COCO_format/Train_image'
v_path = '/home/remo/Desktop/cloth_flaw_detection/Dataset/COCO_format/Val_image'

dict = json.load(open(v_json,'r'))
with open('/home/remo/Desktop/cloth_flaw_detection/Dataset/COCO_format/val.txt','w') as f:
    images = dict['images']
    file_names = []
    for anno in images:
        file_name = anno['file_name']
        if file_name not in file_names:
            f.write(file_name+'\n')
            file_names.append(file_name)

