# -*- coding: utf-8 -*-
# @Time    : 19-10-11 上午10:13
# @Author  : Sloan
# @Email   : 630298149@qq.com
# @File    : inference.py
# @Software: PyCharm
import json
import torch
import cv2
import numpy as np
from torchvision.models import resnet50,resnet101
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--net', dest='net',type=str, default='resnet50',help='resnet101,resnet50')
parser.add_argument('--checkpoint_path', type=str, default='work_dirs/classcification/weights.pth',help='path where model checkpoints are saved')
parser.add_argument('--data_path', type=str, default='/tcdata/guangdong1_round2_testA_20190924/',help='directory where test data are saved')
parser.add_argument('--json_name', type=str, default='result.json',help='the name of commited json')

opt = parser.parse_args()


defect_label = {
        0: 'normal',
        1: 'defect'
    }

def inference():
    # RGB
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    w, h = 640, 640
    mean = np.array(mean).reshape(3, 1, 1)
    std = np.array(std).reshape(3, 1, 1)
    root_path = opt.data_path
    checkpoint_path = opt.checkpoint_path
    json_name = opt.json_name
    # load model
    if opt.net == "resnet50":
        model = resnet50(pretrained=True)
        model.avgpool = torch.nn.AdaptiveAvgPool2d(output_size=1)
        model.fc = torch.nn.Linear(model.fc.in_features, 61)  # 61?
    elif opt.net == "resnet101":
        model = resnet101(pretrained=True)
        model.avgpool = torch.nn.AdaptiveAvgPool2d(output_size=1)
        model.fc = torch.nn.Linear(model.fc.in_features, 61)  # 61?
    model.eval()
    try:
        model.load_state_dict(torch.load(checkpoint_path))
        model = torch.nn.DataParallel(model)
    except RuntimeError:
        model = torch.nn.DataParallel(model)
        model.load_state_dict(torch.load(checkpoint_path))
    model = model.cuda()
    #print(model)
    correct = []
    result = []
    for a, b, c in os.walk(root_path):
        for img_name in c:
            if img_name.endswith('jpg') and 'template' not in img_name:
                #print(a, img_name)
                img_path = os.path.join(a, img_name)
                img = cv2.imread(img_path)
                img = img[:, :, ::-1]  # BGR-->RGB
                img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
                img = (img.transpose((2, 0, 1)) / 255. - mean) / std
                img = torch.from_numpy(img[np.newaxis, :, :, :]).float().cuda()
                outputs = model(img)
                _, preds = torch.max(outputs, 1)
                # defect
                if preds.item() == 1:
                    result.append(
                        {'name': img_name, 'category': 1, 'bbox': [1, 1, 2, 2],
                         'score': 0.5})
                # pred_label = defect_label[preds.item()]
                # if pred_label == img_name.split('_')[0]:
                #     correct.append(1)
                # else:
                #     correct.append(0)
                # print("pred:",pred_label)
    # print("acc:",np.array(correct).mean())

    with open(json_name, 'w') as fp:
        json.dump(result, fp, indent=4, separators=(',', ': '))

if __name__ == "__main__":
    inference()