import sys
import torch
import torchvision
sys.path.append('/home/zhangming/work/kaggle/mmdetection/mmcv-master')
import mmcv
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.apis import inference_detector, show_result, init_detector
import json
import os
import numpy as np
import argparse
import time
import cv2
from tqdm import tqdm
# import utils_swa
import os.path as osp
from collections import OrderedDict
import copy

def weights_to_cpu(state_dict):
    """Copy a model state_dict to cpu.

    Args:
        state_dict (OrderedDict): Model weights on GPU.

    Returns:
        OrderedDict: Model weights on GPU.
    """
    state_dict_cpu = OrderedDict()
    for key, val in state_dict.items():
        state_dict_cpu[key] = val.cpu()
    return state_dict_cpu


def save_checkpoint(model, filename, optimizer=None, meta=None):
    """Save checkpoint to file.

    The checkpoint will have 3 fields: ``meta``, ``state_dict`` and
    ``optimizer``. By default ``meta`` will contain version and time info.

    Args:
        model (Module): Module whose params are to be saved.
        filename (str): Checkpoint filename.
        optimizer (:obj:`Optimizer`, optional): Optimizer to be saved.
        meta (dict, optional): Metadata to be saved in checkpoint.
    """
    if meta is None:
        meta = {}
    elif not isinstance(meta, dict):
        raise TypeError('meta must be a dict or None, but got {}'.format(
            type(meta)))
    meta.update(mmcv_version=mmcv.__version__, time=time.asctime())

    mmcv.mkdir_or_exist(osp.dirname(filename))
    if hasattr(model, 'module'):
        model = model.module

    checkpoint = {
        'meta': meta,
        'state_dict': weights_to_cpu(model.state_dict())
    }
    if optimizer is not None:
        checkpoint['optimizer'] = optimizer.state_dict()

    torch.save(checkpoint, filename)


def swa(models_path, config, save_path):

    models_list = []
    # build the model from a config file and a checkpoint file
    for model_path in models_path:
        model = init_detector(config, model_path, device='cuda:0')
        models_list.append(model)

    num_models = float(len(models_list))
    model_tmp = copy.deepcopy(models_list[0])
    parm_tmp = model_tmp.parameters()

    for parm1, parm2, parm3, tmp in zip(models_list[0].parameters(), models_list[1].parameters(), models_list[2].parameters(), parm_tmp):
        tmp.data = (parm1.data + parm2.data + parm3.data) / num_models

    save_checkpoint(model_tmp, save_path, )
    print(' save swa mode: ' , save_path)

if __name__ == "__main__":


    # models = ['/home/xjx/ding/86/latest.pth', '/home/xjx/ding/86/latest.pth', '/home/xjx/ding/86/latest.pth']
    # config = '/home/xjx/ding/86/faster_rcnn_dconv_c3-c5_x101_32x4d_fpn_1x_86.py'
    # save_path = '/home/xjx/ding/86/swa_86.pth'

    models = ['/home/zhangming/work/kaggle/mmdetection/work_dirs/faster_rcnn_dconv_c3-c5_x101_32x4d_fpn_1x_86/epoch_12.pth',
              '/home/zhangming/work/kaggle/mmdetection/work_dirs/faster_rcnn_dconv_c3-c5_x101_32x4d_fpn_1x_86/epoch_11.pth',
              '/home/zhangming/work/kaggle/mmdetection/work_dirs/faster_rcnn_dconv_c3-c5_x101_32x4d_fpn_1x_86/epoch_10.pth']

    config = '/home/zhangming/work/kaggle/mmdetection/cloth/faster_rcnn_dconv_c3-c5_x101_32x4d_fpn_1x_86.py'

    save_path = '/home/zhangming/work/kaggle/mmdetection/work_dirs/faster_rcnn_dconv_c3-c5_x101_32x4d_fpn_1x_86/swa_86.pth'
    swa(models, config, save_path )

