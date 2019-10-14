#encoding:utf/8
import sys
sys.path.append('/home/zhangming/work/kaggle/mmdetection/mmcv-master')
import mmcv
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.apis import inference_detector, show_result, init_detector,crop_inference_detector
import time
import json
import os
import numpy as np
import argparse
from tqdm import tqdm
from copy import deepcopy as dcopy
from easydict import EasyDict as edict
# def restore(results):
#     '''
#
#     :param results: [{'result':result,'flag_crop':True,'img_shape':(h,w,3),'patch_size':(col,row)},{},...]
#     # result:[ array([ [ [x1,y1,x2,y2,score],[x1,y1,x2,y2,score],[x1,y1,x2,y2,score] ] ]),
#                 array([ [ [x1,y1,x2,y2,score],[x1,y1,x2,y2,score],[x1,y1,x2,y2,score] ] ])
#                 ...] (num_class,num_det,5)
#
#     :return:
#     '''
#     results_ = [[] for i in range(len(results[0]['result']))]
#     for result in results:
#         bboxes = result['result']
#         flag_crop = result['flag_crop']
#         if flag_crop:
#             img_shape = result['img_shape']
#             patch_size = result['patch_size']
#             for i in range(len(bboxes)):
#                 bboxes[i][:,0]+=patch_size[1]*img_shape[1]
#                 bboxes[i][:,1]+=patch_size[0]*img_shape[0]
#                 bboxes[i][:,2]+=patch_size[1]*img_shape[1]
#                 bboxes[i][:,3]+=patch_size[0]*img_shape[0]
#                 results_[i]+=bboxes[i].tolist()
#         else:
#             for i in range(len(bboxes)):
#                 results_[i]+=bboxes[i].tolist()
#     return  results_
#
# def nms(boxes, overlap_threshold=0.5, mode='Union'):
#     # if there are no boxes, return an empty list
#     if len(boxes) == 0:
#         return []
#
#     # if the bounding boxes integers, convert them to floats
#     if boxes.dtype.kind == "i":
#         boxes = boxes.astype("float")
#
#     # initialize the list of picked indexes
#     pick = []
#
#     # grab the coordinates of the bounding boxes
#     x1, y1, x2, y2, score = [boxes[:, i] for i in range(5)]
#
#     area = (x2 - x1 + 1) * (y2 - y1 + 1)
#     idxs = np.argsort(score)
#
#     while len(idxs) > 0:
#         last = len(idxs) - 1
#         i = idxs[last]
#         pick.append(i)
#
#         xx1 = np.maximum(x1[i], x1[idxs[:last]])
#         yy1 = np.maximum(y1[i], y1[idxs[:last]])
#         xx2 = np.minimum(x2[i], x2[idxs[:last]])
#         yy2 = np.minimum(y2[i], y2[idxs[:last]])
#
#         w = np.maximum(0, xx2 - xx1 + 1)
#         h = np.maximum(0, yy2 - yy1 + 1)
#
#         inter = w * h
#         if mode == 'Min':
#             overlap = inter / np.minimum(area[i], area[idxs[:last]])
#         else:
#             overlap = inter / (area[i] + area[idxs[:last]] - inter)
#
#
#         idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlap_threshold)[0])))
#     return pick
#
# #线上裁剪用
# def result_frompic_crop():
#     config_file = config2make_json
#     checkpoint_file = model2make_json
#
#     # build the model from a config file and a checkpoint file
#     model = init_detector(config_file, checkpoint_file, device='cuda:0')
#     dirs = os.listdir(pic_path)
#     meta = []
#     for dir in dirs:
#         root = os.path.join(pic_path,dir)
#         im_name = dir+'.jpg'
#         img = os.path.join(root,im_name)
#         results = crop_inference_detector(model, img)
#         result_ = restore(results)
#         for i ,boxes in enumerate(result_,1):
#             keep = nms(np.array(boxes))
#             defect_label = i
#             for index in keep:
#                 box = boxes[index]
#                 anno = {}
#                 anno['name'] = im_name
#                 anno['category'] = defect_label
#                 anno['bbox'] = [round(float(i), 2) for i in box[0:4]]
#                 anno['score'] = float(box[4])
#                 meta.append(anno)
#     with open(json_out_path, 'w') as fp:
#         json.dump(meta, fp, cls=MyEncoder, indent=4, separators=(',', ': '))

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

#线上不裁剪用
def result_frompic_no_crop():
    config_file = config2make_json
    checkpoint_file = model2make_json

    # build the model from a config file and a checkpoint file
    model = init_detector(config_file, checkpoint_file, device='cuda:0')
    dirs = os.listdir(pic_path)
    meta = []
    for dir in dirs:
        t1 = time.time()
        root = os.path.join(pic_path,dir)
        im_name = dir+'.jpg'
        img = os.path.join(root,im_name)
        result_ = inference_detector(model, img)
        for i ,boxes in enumerate(result_,1):
            if len(boxes):
                defect_label = i
                for box in boxes:
                    anno = {}
                    anno['name'] = im_name
                    anno['category'] = defect_label
                    anno['bbox'] = [round(float(i), 2) for i in box[0:4]]
                    anno['score'] = float(box[4])
                    meta.append(anno)
        t2 = time.time()
        print("time one im ",str(t2-t1))
    with open(json_out_path, 'w') as fp:
        json.dump(meta, fp, cls=MyEncoder, indent=4, separators=(',', ': '))

#本地验证集用
def result_frompic_val():
    config_file = config2make_json
    checkpoint_file = model2make_json

    # build the model from a config file and a checkpoint file
    model = init_detector(config_file, checkpoint_file, device='cuda:0')
    pics = os.listdir(pic_path)
    meta = []
    for im in tqdm(pics):
        t1 = time.time()
        img = os.path.join(pic_path,im)
        result_ = inference_detector(model, img)
        for i ,boxes in enumerate(result_,1):
            if len(boxes):
                defect_label = i
                for box in boxes:
                    anno = {}
                    anno['name'] = im
                    anno['category'] = defect_label
                    anno['bbox'] = [round(float(i), 2) for i in box[0:4]]
                    anno['score'] = float(box[4])
                    meta.append(anno)
        t2 = time.time()
        print("time one im ",str(t2-t1))
    with open(json_out_path, 'w') as fp:
        json.dump(meta, fp, cls=MyEncoder, indent=4, separators=(',', ': '))

def nms(dets,thr=0.5, mode='iou'):
    # dets:[N,5]
    assert dets.shape[-1] % 5 == 0
    if dets.shape[0] == 1:
        return 0
    x1, y1, x2, y2, score = dets[:,0], dets[:,1], dets[:,2], dets[:,3], dets[:,4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    keep = []
    orders = score.argsort()[::-1]
    while orders.size > 0:
        i = orders[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[orders[1:]])
        yy1 = np.maximum(y1[i], y1[orders[1:]])
        xx2 = np.minimum(x2[i], x2[orders[1:]])
        yy2 = np.minimum(y2[i], y2[orders[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        if mode == 'iou':
            ovr = inter / (areas[i] + areas[orders[1:]] - inter)
        elif mode == 'iof':
            arr_areas = dcopy(areas[orders[1:]])
            arr_areas[arr_areas > areas[i]] = areas[i]
            ovr = inter / arr_areas
#        ovr = inter / (areas[i] + areas[orders[1:]] - inter)
        inds = np.where(ovr <= thr)[0]
        orders = orders[inds + 1]
    return keep

#线下裁剪用
def gen_commit_result_round2_down(pic_path):

    # build the model from a config file and a checkpoint file
    model = init_detector(config2make_json, model2make_json, device='cuda:0')

    img_list = []
    for img_name in os.listdir(pic_path):
        if img_name.endswith('.jpg'):
            img_list.append(img_name)

    result = []
    for img_name in tqdm(img_list):
        model.cfg.data.test.pipeline[1].img_scale = (2048, 850)
        model.cfg.test_cfg.rcnn.score_thr = 0.05

        model.cfg.data.test.pipeline[1].flip = False

        t1 = time.time()
        full_img = os.path.join(pic_path, img_name)
        full_img = mmcv.imread(full_img)
        img_h,img_w = full_img.shape[:2]
        patches = [np.array((0, 0, img_w // 2, img_h // 2)), np.array((img_w // 2, 0, img_w, img_h // 2)),
                   np.array((0, img_h // 2, img_w // 2, img_h)), np.array((img_w // 2, img_h // 2, img_w, img_h))]
        predicts = []
        for patch_idx, patch in enumerate(patches):
            patch_img = full_img[patch[1]:patch[3],patch[0]:patch[2]]
            predicts.append(inference_detector(model, patch_img))

#        model.cfg.data.test.pipeline[1].img_scale = [(2048,850)]
        # model.cfg.data.test.pipeline[1].img_scale = (2048,850)
        # model.cfg.test_cfg.rcnn.score_thr = 0.05

        # model.cfg.data.test.pipeline[1].flip = True

        predicts.append(inference_detector(model, full_img))

        for i, (bboxes1, bboxes2,bboxes3,bboxes4,bboxes5) in enumerate(zip(predicts[0], predicts[1],predicts[2],predicts[3],predicts[4])):

            bboxes1[:,:4] += np.tile(patches[0][:2], 2)
            bboxes2[:,:4] += np.tile(patches[1][:2], 2)
            bboxes3[:,:4] += np.tile(patches[2][:2], 2)
            bboxes4[:,:4] += np.tile(patches[3][:2], 2)
            merge_bboxes = np.concatenate([bboxes1, bboxes2,bboxes3,bboxes4,bboxes5], axis=0)

            #if i+1 in [5, 6, 9, 10, 11, 14, 13,15]: # only 
            #    merge_bboxes =  bboxes5

            # print("merged:",merge_bboxes)
            if merge_bboxes.size:
                #iof_classes = [5, 6, 8, 9, 10, 11, 14, 15]
                #if i + 1 == iof_classes:
                #    keep_inds = nms(merge_bboxes,thr=0.7,   mode='iof')
                #else:
                keep_inds = nms(merge_bboxes,thr=0.5,   mode='iou')
                merge_bboxes = merge_bboxes[keep_inds].reshape(-1,5)
                defect_label = i + 1
                image_name = img_name

                for bbox in merge_bboxes:

                    x1, y1, x2, y2, score = bbox.tolist()
                    # cv2.rectangle(full_img,(int(x1),int(y1)),(int(x2),int(y2)),(0,0,255),4)
                    # cv2.putText(full_img,"{}|{}".format(defect_label,str(score)),(int(x1),int(y1+20)),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,0),2)

                    x1, y1, x2, y2 = round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)
                    result.append(
                        {'name': image_name, 'category': defect_label, 'bbox': [x1, y1, x2, y2],
                         'score': score})
        t2 = time.time()
        print("time one im ", str(t2 - t1))
        # full_img = cv2.resize(full_img, (img_w // 3, img_h // 3))
        # cv2.imshow("full", full_img)
        # c = cv2.waitKey(0) & 0xFF
        # if c == ord('q'):
        #     break
        # cv2.imwrite(img_name,full_img)
    with open(json_out_path, 'w') as fp:
        json.dump(result, fp, indent=4, separators=(',', ': '))

#线上裁剪用
def gen_commit_result_round2(pic_path):

    # build the model from a config file and a checkpoint file
    model = init_detector(config2make_json, model2make_json, device='cuda:0')

    img_list = []
    # for img_name in os.listdir(pic_path):
    #     if img_name.endswith('.jpg'):
    #         img_list.append(img_name)

    for root,dirs,files in os.walk(pic_path):
        for filename in files:
            if filename.endswith('.jpg') and "template" not in filename:
                img_list.append(filename)

    result = []
    for img_name in tqdm(img_list):
        t1 = time.time()
        img_dir = img_name.split('.')[0]
        pic_path_ = os.path.join(pic_path, img_dir)
        full_img = os.path.join(pic_path_, img_name)
        full_img = mmcv.imread(full_img)
        img_h,img_w = full_img.shape[:2]
        patches = [np.array((0, 0, img_w // 2, img_h // 2)), np.array((img_w // 2, 0, img_w, img_h // 2)),
                   np.array((0, img_h // 2, img_w // 2, img_h)), np.array((img_w // 2, img_h // 2, img_w, img_h))]
        predicts = []
        for patch_idx, patch in enumerate(patches):
            patch_img = full_img[patch[1]:patch[3],patch[0]:patch[2]]
            predicts.append(inference_detector(model, patch_img))
        predicts.append(inference_detector(model, full_img))
        for i, (bboxes1, bboxes2,bboxes3,bboxes4,bboxes5) in enumerate(zip(predicts[0], predicts[1],predicts[2],predicts[3],predicts[4])):

            bboxes1[:,:4] += np.tile(patches[0][:2], 2)
            bboxes2[:,:4] += np.tile(patches[1][:2], 2)
            bboxes3[:,:4] += np.tile(patches[2][:2], 2)
            bboxes4[:,:4] += np.tile(patches[3][:2], 2)
            merge_bboxes = np.concatenate([bboxes1, bboxes2,bboxes3,bboxes4,bboxes5], axis=0)
            # print("merged:",merge_bboxes)
            if merge_bboxes.size:
                keep_inds = nms(merge_bboxes,thr=0.5)
                merge_bboxes = merge_bboxes[keep_inds].reshape(-1,5)
                defect_label = i + 1
                image_name = img_name

                for bbox in merge_bboxes:

                    x1, y1, x2, y2, score = bbox.tolist()
                    # cv2.rectangle(full_img,(int(x1),int(y1)),(int(x2),int(y2)),(0,0,255),4)
                    # cv2.putText(full_img,"{}|{}".format(defect_label,str(score)),(int(x1),int(y1+20)),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,0),2)

                    x1, y1, x2, y2 = round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)
                    result.append(
                        {'name': image_name, 'category': defect_label, 'bbox': [x1, y1, x2, y2],
                         'score': score})
        t2 = time.time()
        print("time one im ", str(t2 - t1))
        # full_img = cv2.resize(full_img, (img_w // 3, img_h // 3))
        # cv2.imshow("full", full_img)
        # c = cv2.waitKey(0) & 0xFF
        # if c == ord('q'):
        #     break
        # cv2.imwrite(img_name,full_img)
    with open(json_out_path, 'w') as fp:
        json.dump(result, fp, indent=4, separators=(',', ': '))

def result_frompic_val_dirs():
    config_file = config2make_json
    checkpoint_file = model2make_json

    # build the model from a config file and a checkpoint file
    model = init_detector(config_file, checkpoint_file, device='cuda:0')
    dirs = os.listdir(pic_path)
    meta = []
    for dir in dirs:
        t1 = time.time()
        root = os.path.join(pic_path,dir)
        im_name = dir+'.jpg'
        img = os.path.join(root, im_name)

        if flag_template:
            template_img_name = "template"+'_'+dir.split('_')[0]+'.jpg'
            template_img = os.path.join(root,template_img_name)
            temp_img = mmcv.imread(img)-mmcv.imread(template_img)
        result_ = inference_detector(model, img)
        for i ,boxes in enumerate(result_,1):
            if len(boxes):
                defect_label = i
                for box in boxes:
                    anno = {}
                    anno['name'] = im_name
                    anno['category'] = defect_label
                    anno['bbox'] = [round(float(i), 2) for i in box[0:4]]
                    anno['score'] = float(box[4])
                    if flag_template:
                        box_temp = temp_img[int(anno['bbox'][1]):int(anno['bbox'][3]),int(anno['bbox'][0]):int(anno['bbox'][2]),:]
                        area = (anno['bbox'][3]-anno['bbox'][1])*(anno['bbox'][2]-anno['bbox'][0])
                        percen = len(np.where(box_temp>1)[0])/float(area)/3
                        print(percen)
                        if percen > 0.5:
                            meta.append(anno)
                        else:
                            print("不满足要求 %d"%percen)
                    else:
                        meta.append(anno)
        t2 = time.time()
        print("time one im ",str(t2-t1))
    with open(json_out_path, 'w') as fp:
        json.dump(meta, fp, cls=MyEncoder, indent=4, separators=(',', ': '))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate result"
    )
    parser.add_argument(
        "-p", "--phase",
        default="test",
        help="Test val data or test data",
        type=str,
    )
    parser.add_argument(
        "-m", "--model",
        help="Model path",
        type=str,
    )
    parser.add_argument(
        "-c", "--config",
        help="Config path",
        type=str,
    )
    parser.add_argument(
        "-im", "--im_dir",
        help="Image path",
        type=str,
    )
    parser.add_argument(
        '-o', "--out",
        help="Save path",
        type=str,
    )
    parser.add_argument(
        "-temp","--flag_template",
        default=True,
        help="flag use_template",
        type=bool,
    )
    args = parser.parse_args()
    model2make_json = args.model
    config2make_json = args.config
    json_out_path = args.out
    flag_template = args.flag_template
    if args.phase == 'test':
        pic_path = "/home/zhangming/Models/Results/cloth_flaw_detection/Datasets_2/guangdong1_round2_train_part1_20190924/defect/"
    # else:
    if args.phase == "val":
        pic_path = "/home/zhangming/Models/Results/cloth_flaw_detection/Datasets_2/guangdong1_round2_train_part1_20190924/defect/"
        #pic_path = "/home/zhangming/Models/Results/cloth_flaw_detection/Datasets_2/val_925/"

    flag = 4

    if flag == 0: # 线下不裁剪测试
        result_frompic_val()
    if flag == 1:# 线上不裁剪测试
        result_frompic_no_crop()
    if flag == 2: #线上裁剪测试
        gen_commit_result_round2(pic_path)
    if flag == 3: # 线下裁剪测试
        gen_commit_result_round2_down(pic_path)
    if flag == 4: # 线下不裁剪测试按文件夹(数据集格式和线上一样)
        result_frompic_val_dirs()

# python my_util/make_re.py -p val -m workdir/XXX -c round2/XXX -o results/result_XXX.json

