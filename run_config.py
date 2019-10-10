import pynvml
import os
import time

cfg_list = [
    'echo 10',
    'echo 9',
    'echo 8',
    'echo 7',
'./tools/dist_train.sh cloth/faster_rcnn_x101_64x4d_fpn_1x_95.py 8',
'./tools/dist_train.sh cloth/faster_rcnn_x101_64x4d_fpn_1x_96.py 8'

#    './tools/dist_train.sh cloth/faster_rcnn_dconv_c3-c5_x101_32x4d_fpn_1x_89.py 8', 
   
]
while True:
    time.sleep(60*5)
    pynvml.nvmlInit()
    # 这里的0是GPU id
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    meminfo_us = meminfo.used // 1000_000
    if meminfo_us < 1000:
        try:
            cfg_str = cfg_list.pop()
            # print(cfg_str + '\n')
            os.system(cfg_str)
        except:
            print(' ------ empty ----- ')
            break
    else:
        del handle
        del meminfo
