# draw mAP on val dataset; first gen gt file & detect file

# gt: bottle 6 234 39 128
# dt: bottle 0.14981 80 1 295 500

import json
import os

#############################  修改   #####################################
dt_path = "/home/remo/Desktop/cloth_flaw_detection/Results/result_29_val-30epoch_true.json"
##########################################################################

cat = {
    '破洞': 1, '水渍': 2, '油渍': 2, '污渍': 2, '三丝': 3, '结头': 4, '花板跳': 5, '百脚': 6, '毛粒': 7,
    '粗经': 8, '松经': 9, '断经': 10, '吊经': 11, '粗维': 12, '纬缩': 13, '浆斑': 14, '整经结': 15, '星跳': 16, '跳花': 16,
    '断氨纶': 17, '稀密档': 18, '浪纹档': 18, '色差档': 18, '磨痕': 19, '轧痕': 19, '修痕': 19, '烧毛痕': 19, '死皱': 20, '云织': 20,
    '双纬': 20, '双经': 20, '跳纱': 20, '筘路': 20, '纬纱不良': 20,
    }

gt_path = ["/home/remo/Desktop/cloth_flaw_detection/guangdong1_round1_train2_20190828/Annotations/anno_train.json",
           "/home/remo/Desktop/cloth_flaw_detection/guangdong1_round1_train1_20190818/Annotations/anno_train.json",
           "/home/remo/Desktop/cloth_flaw_detection/Dataset/COCO_format/layout/Aug_no_crop.json",
           ]

re_root = "/home/remo/Desktop/cloth_flaw_detection/Results/re_map/"
dir_name = dt_path.split('/')[-1].split('.')[0]
if not os.path.exists(re_root+dir_name):
    os.mkdir(re_root+dir_name)
det_path = re_root+dir_name+'/'+'detections'
if not os.path.exists(det_path):
    os.mkdir(det_path)

# 生成 result
with open(dt_path, "r") as f:
    json_data = json.loads(" ".join(f.readlines()))
all_names = {}
for it in json_data:
    name = it['name']
    bbox = it['bbox']
    score = it['score']

    if name in all_names:
        all_names[name].append([it['category'], score] + bbox)
    else:
        all_names[name] = [[it['category'], score] + bbox]

# 生成 ground_truth
# val = []
# f = open('/home/remo/Desktop/cloth_flaw_detection/Dataset/COCO_format/layout/val_no_crop.txt','r')
# [val.append(line.strip().split('.')[0]) for line in f]
# all_names = {}
# for gt in gt_path:
#     with open(gt, "r") as f:
#         json_data = json.loads(" ".join(f.readlines()))
#         for it in json_data:
#             name = it['name']
#             bbox = it['bbox']
#             label = it['defect_name']
#             if name.split('.')[0] in val:
#                 if name in all_names:
#                     all_names[name].append([str(cat[label])] + bbox)
#                 else:
#                     all_names[name] = [[str(cat[label])] + bbox]

for name, val in all_names.items():
    with open(det_path+"/%s.txt" % name, "w") as f:
        for box in val:
            box_info = " ".join([str(v) for v in box])
            f.write(box_info + "\n")