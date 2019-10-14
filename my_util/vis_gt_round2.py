# encoding:utf/8
import sys
# reload(sys)
# sys.setdefaultencoding('utf8')
import os
import json
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import xml.dom.minidom
from xml.dom.minidom import Document
from tqdm import tqdm
from easydict import EasyDict as edict
import os.path as osp
import math
from tqdm import tqdm

import imgaug as ia
from imgaug import augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import imageio
import getpass  # 获取用户名
import random
import copy
import shutil
import copy
import sys
# sys.path.append('/home/zhangming/work/kaggle/mmdetection/mmcv-master')
# import mmcv
# from mmdet.apis import inference_detector, init_detector
import time
import json
import argparse

USER = getpass.getuser()

s1 = """<object>
<name>{0}</name>
<pose>Unspecified</pose>
<truncated>0</truncated>
<difficult>0</difficult>
<bndbox>
<xmin>{1}</xmin>
<ymin>{2}</ymin>
<xmax>{3}</xmax>
<ymax>{4}</ymax>
</bndbox>
</object>"""


s2 = """<annotation>
<folder>VOC2007</folder>
<filename>{0}</filename>
<source>
<database>My Database</database>
<annotation>VOC2007</annotation>
<image>flickr</image>
<flickrid>NULL</flickrid>
</source>
<owner>
<flickrid>NULL</flickrid>
<name>sloan</name>
</owner>
<size>
<width>{1}</width>
<height>{2}</height>
<depth>3</depth>
</size>
<segmented>0</segmented>{3}
</annotation>
"""


def set_seed():
    print("Fixing random seed for reproducibility...")
    SEED = 2019  # 123  #35202   #int(time.time()) #
    random.seed(SEED)
    np.random.seed(SEED)
    print('\tSetting random seed to {}.'.format(SEED))

class Config:
    def __init__(self):
        self.json_paths = ['']  # train json
        self.val_json_paths = ['']     # val json

        self.allimg_path = ''   # 训练图片集
        self.val_img_path = ''  # 验证图片集
        self.add_num = 0        # add_aug_data 扩增数据数量

        self.result_json = '' # 模型对val 的输出结果
        self.divide_json = ''

        self.submit_json = ['']
        self.submit_path = ''
class DataAnalyze:
    '''
    bbox 分析类，
        1. 每一类的bbox 尺寸统计
        2.
    '''
    def __init__(self, cfg: Config,flag_coco=False):
        self.cfg = cfg
        self.category = {
            '沾污': 1, '错花': 2, '水印': 3, '花毛': 4, '缝头': 5, '缝头印': 6, '虫粘': 7,
            '破洞': 8, '褶子': 9, '织疵': 10, '漏印': 11, '蜡斑': 12, '色差': 13, '网折': 14, '其他': 15
        }
        self.reverse_category = {
            1:'沾污', 2:'错花',3: '水印', 4:'花毛', 5:'缝头', 6:'缝头印', 7:'虫粘',
            8:'破洞', 9:'褶子', 10:'织疵', 11:'漏印', 12:'蜡斑', 13:'色差', 14:'网折', 15:'其他'
        }
        self.num_classes = 15  # 前景类别

        self.all_instance, self.cla_instance, self.img_instance, self.img_defect = self._create_data_dict(cfg.json_paths, cfg.allimg_path)
        # if hasattr(cfg, 'val_json_paths') and not  cfg.val_json_paths == '' :
        #     self.val_all_instance, self.val_cla_instance, self.val_img_instance = self._create_data_dict(cfg.val_json_paths, cfg.val_img_path)
        # else:
        #     self.val_all_instance, self.val_cla_instance, self.val_img_instance = (None, None, None)
        set_seed()

        '''
        all_instance 
        [
            {'bbox': [2000.66, 326.38, 2029.87, 355.59],
             'defect_name': '结头',
             'name': 'd6718a7129af0ecf0827157752.jpg',
             'abs_path' : 'xxx/xxx.jpg',
             'classes': 1
             'w':1,
             'h':1,
             'area':1,
             }
        ]

        cla_instance 
            {'1':[], '2':[] }
        '''

        self.num_data = len(self.all_instance)

    def _create_data_dict(self, json_path, data_file, flag_ins_list=False):
        '''
        flag_ins_list: True 传入的json_path 为 [instance1, instance2, ] 不用json 文件读取
        :return:
            instance:
                {'bbox': [2000.66, 326.38, 2029.87, 355.59],
                 'defect_name': '结头',
                 'name': 'd6718a7129af0ecf0827157752.jpg',
                 'abs_path' : 'xxx/xxx.jpg',
                 'w':1,
                 'h':1,
                 'area':1,
                 'im_w':1
                 'im_h':2
                 }

        all_instance
            [instance1, instance2, instance3]

        cla_instance
            {'1':[instance, instances2], '2'[instance, ]}

        img_instance
            {'xx1.jpg': [instance]  'xxx.jpg':[instance, instance]}

        img_defect
            {'xx1.jpg': [defeact1,defeact2]  'xxx.jpg':[defeact1,defeact2]}

        '''
        if flag_ins_list:
            json_path = [json_path ]# 为 [ [ins1, ins2] ] 2维数组

        all_defect = []
        all_instance = []
        key_classes = list(range(1, self.num_classes + 1))  # 1 ... num_classes

        cla_instance = edict({str(k): [] for k in key_classes})  # key 必须是字符串
        img_instance = edict()
        img_defect = edict()

        if isinstance(json_path, str):
            json_path = [json_path]

        if isinstance(json_path, list):
            for path in json_path:
                if flag_ins_list:
                    gt_list = path
                else:
                    gt_list = json.load(open(path, 'r'))

                for instance in tqdm(gt_list):
                    instance = edict(instance)
                    if not hasattr(instance, 'defect_name') and hasattr(instance, 'category'):
                        instance.defect_name = self.reverse_category[instance.category]
                    instance.classes = int(self.category[instance.defect_name])  # add classes int
                    w, h = compute_wh(instance.bbox)
                    instance.w = round(w, 2)  # add w
                    instance.h = round(h, 2)  # add h
                    instance.area = round(w * h, 2)  # add area
                    name = instance.name
                    dir = name.split('.')[0]
                    instance.abs_path = osp.join(data_file, instance.name)  # add 绝对路径
                    # im = cv2.imread(instance.abs_path)
                    # instance.im_w = im.shape[2]
                    # instance.im_h = im.shape[1]
                    instance.im_w = 4096
                    instance.im_h = 1696
                    all_instance.append(instance)  # 所有instance

                    cla_instance[str(instance.classes)].append(instance)  # 每类的instance

                    if instance.name not in img_instance.keys():  # 每张图片的instance
                        img_instance[instance.name] = [instance]
                    else:
                        img_instance[instance.name].append(instance)

                    if instance.name not in img_defect.keys():
                        img_defect[instance.name] = [instance.classes]
                    else:
                        img_defect[instance.name].append(instance.classes)


        return all_instance, cla_instance, img_instance,img_defect

    # def load_coco_format(self):
    #     all_instance = []
    #     key_classes = list(range(1, self.num_classes + 1))  # 1 ... num_classes
    #
    #     cla_instance = edict({str(k): [] for k in key_classes})  # key 必须是字符串
    #     img_instance = edict()
    #     if isinstance(self.cfg.json_paths, str):
    #         self.cfg.json_paths = [self.cfg.json_paths]
    #     if isinstance(self.cfg.json_paths, list):
    #         for path in self.cfg.json_paths:
    #             gt_list = json.load(open(path, 'r'))

    def ana_classes(self):
        ws_all = []
        hs_all = []
        for cla_name, bboxes_list in self.cla_instance.items():  # str '1': [edict()]
            ws = []
            hs = []
            for instance in bboxes_list:
                ws.append(instance.w)
                hs.append(instance.h)
                ws_all.append(instance.w)
                hs_all.append(instance.h)
            # plt.title(cla_name, fontsize='large',fontweight = 'bold')
            # plt.scatter(ws, hs, marker='x', label=cla_name, s=30)
        plt.scatter(ws_all, hs_all, marker='x', s=30)

        plt.grid(True)
        plt.show()

    def ana_boxes(self):
        '''

        :return:
        '''

        import seaborn as sns
        from numpy.random import randn
        import matplotlib as mpl
        from scipy import stats
        from scipy.stats import multivariate_normal
        from mpl_toolkits.mplot3d import Axes3D

        # style set 这里只是一些简单的style设置
        # sns.set_palette('deep', desat=.6)
        # sns.set_context(rc={'figure.figsize': (8, 5)})
        sns.set(color_codes=True)

        # data = np.random.multivariate_normal([0, 0], [[1, 2], [2, 20]], size=1000)
        # data = pd.DataFrame(data, columns=["X", "Y"])
        # mpl.rc("figure", figsize=(6, 6))
        # # sns.kdeplot(data.X, data.Y, shade=True, bw="silverman", gridsize=50, clip=(-11, 11))
        # with sns.axes_style('white'):
        #     sns.jointplot('X', 'Y', data, kind='kde')
        # plt.show()

        # x = stats.gamma(2).rvs(5000)
        # y = stats.gamma(50).rvs(5000)
        # del_instance = self.all_instance
        for i in range(1, 16):
            del_instance = self.cla_instance[str(i)]
            ws = []
            hs = []
            areas = []
            ratiowh = []
            for ins in del_instance:
                ws.append(ins.w)
                hs.append(ins.h)
                areas.append(ins.area)
                ratiowh.append(ins.w / float(ins.h))
            # with sns.axes_style("dark"):
            #     sns.jointplot(ws, hs, kind="hex", )

            # bins = np.linspace(0, 2500)
            # bins = 500
            # plt.hist(areas, bins, normed=False, color="#FF0000", alpha=.9, )
            # plt.hist(ratiowh, bins, density=False, color="#C1F320", alpha=.9, )
            # plt.hist(ws, bins, normed=False, color="#FF0000", alpha=.9, )
            # plt.hist(hs, bins, normed=False, color="#C1F320", alpha=.5)

            # sns.jointplot(ws, hs, kind="reg", )
            # sns.jointplot(ws, hs, kind="hex", )

            # ---------------
            #  2维分布图
            g = sns.jointplot(x=ws, y=hs, kind="kde", color="m", cbar=True)
            g.plot_joint(plt.scatter, c="b", s=20, linewidth=0.5, marker="+")
            g.ax_joint.collections[0].set_alpha(0)
            g.set_axis_labels("$X$", "$Y$")
            plt.savefig('/home/remo/Desktop/cloth_flaw_detection/Round2/Analyses/%d.png' % i)
        # plt.show()
        # ----------------
        # f, ax = plt.subplots(figsize=(6, 6))
        # cmap = sns.cubehelix_palette(as_cmap=True, dark=0, light=1, reverse=True)
        # sns.kdeplot(ws, hs, cmap=cmap, n_levels=60, shade=True)
        # sns.kdeplot(ws, hs, shade=True, bw="silverman", gridsize=500, clip=(0, 3000))

        # -------------
        # ax = sns.kdeplot(ws, hs, cmap = "Reds", shade = True, shade_lowest = False, cbar=True) # cbar 设置刻度
        # ax = sns.kdeplot(virginica.sepal_width, virginica.sepal_length, cmap = "Blues", shade = True, shade_lowest = False)

    def draw_cls(self):
        myfont = matplotlib.font_manager.FontProperties(fname='/home/remo/Desktop/simkai_downcc/simkai.ttf')
        cls = [self.reverse_category[i] for i in range(1,self.num_classes+1)]
        cls_each = [len(self.cla_instance[str(i)]) for i in range(1,self.num_classes+1)]
        plt.xticks(range(0, len(cls)), cls, font_properties=myfont, rotation=0)
        plt.bar(cls, cls_each, color='rgb')
        plt.legend()
        plt.grid(True)
        plt.show()

    def add_aug_data(self, add_num=2000, aug_save_path=None, json_file_path=None):
        '''
        1. 设定补充的数据量
        2. 低于这些类的才需要补充
        3. 补充增广函数
            1. 每张图片增广多少张
        :return:
        '''
        if aug_save_path is None or json_file_path is None:
            raise NameError

        if not osp.exists(aug_save_path):
            os.makedirs(aug_save_path)

        transformer = Transformer()

        aug_json_list = []
        auged_image_dict = {}
        for cla_name, bboxes_list in self.cla_instance.items():  # str '1': [edict()]
            cla_num = len(bboxes_list)
            # 按需增广
            if cla_num >= add_num:
                continue
            # 补充数据
            cla_add_num = add_num - cla_num  #

            # 每张图进行增广
            # cla_add_num = cla_num

            each_num = int(math.ceil(cla_add_num / cla_num * 1.0))  # 向上取整 每张图片都进行增广
            # 每张图进行增广扩充
            for instance in tqdm(bboxes_list, desc='cla %s ;process %d: ' % (cla_name, each_num)):  # img_box edict
                # instance attr: bbox, defect_name, name, abs_path, classes, w, h, area
                all_defect = self.img_defect[instance.name]
                if 5 in all_defect or 6 in all_defect:
                    continue
                img = cv2.imread(instance.abs_path)
                try:  # 检测图片是否可用
                    h, w, c = img.shape
                    img_info = edict({'img_h':h, "img_w":w, 'name':instance.name, 'aug_save_path':aug_save_path})
                except:
                    print("%s is wrong " % instance.abs_path)
                import copy
                for ind in range(each_num):  # 循环多次进行增广保存            print(each_num)
                    # print(each_num)
                    img_info.ind = ind
                    if instance.name not in auged_image_dict.keys():
                        aug_name = '%s_aug%d.jpg' % (osp.splitext(instance.name)[0],0)  # 6598413.jpg -> 6598413_aug0.jpg, 6598413_aug1.jpg
                        auged_image_dict[instance.name] = 1
                    else:
                        auged_image_dict[instance.name] += 1
                        aug_name = '%s_aug%d.jpg' % (osp.splitext(instance.name)[0], auged_image_dict[instance.name])
                    img_info.aug_name = aug_name
                    img_ins = copy.deepcopy(self.img_instance[instance.name])
                    aug_img, img_info_tmp = transformer.aug_img(img, img_ins, img_info = img_info) # list
                    if img_info_tmp is not None:
                        aug_json_list += img_info_tmp # 融合
                        # aug_json_list.append(img_info_tmp) # 融合
        print(auged_image_dict)

        # # 保存aug_json 文件
        random.shuffle(aug_json_list)
        with open(json_file_path, 'w') as f:
            json.dump(aug_json_list, f, indent=4, separators=(',', ': '), cls=MyEncoder)

    def vis_gt(self, flag_show_raw_img=False, test_img=None):
        '''
        可视化gt， 按键盘d 向前， 按a 向后
        :return:
        '''
        transformer = Transformer()
        cur_node = 0
        set_img_name = list(self.img_instance.keys())  # 所有图片的名称的 list
        flag_set_test_img = False
        while True:
            img_name = set_img_name[cur_node]

            instances_list = self.img_instance[img_name]
            if test_img is not None and flag_set_test_img == False:
                if isinstance(test_img, str):
                    instances_list = self.img_instance[test_img]

                elif isinstance(test_img, edict):
                    if test_img.flag_continue == True:  # 继续读取接下来的 img list
                        test_img_name = test_img.name
                        cur_node = set_img_name.index(test_img_name)  # 设置节点位置
                        instances_list = self.img_instance[test_img_name]
                        flag_set_test_img = True


            cv2.namedWindow('img', 0)
            cv2.resizeWindow('img', 1920, 1080)
            print('num gt: ', len(instances_list))
            print('img_name: %s ' % (img_name))

            ins_init = instances_list[0]
            img = cv2.imread(ins_init.abs_path)
            img_aug, _ = transformer(img, ins_init)
            img_h, img_w, c = img_aug.shape

            cv2.rectangle(img_aug, (int(0), int(0)), (int(img_w // 2), int(img_h // 2)), (255, 0, 0), 2)
            cv2.rectangle(img_aug, (int(img_w // 2), int(img_h // 2)), (int(img_w), int(img_h)), (255, 0, 0), 2)

            for instance in instances_list:
                box = instance.bbox
                w, h = compute_wh(box)
                cv2.rectangle(img_aug, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)

                cv2.putText(img_aug, '%d' % instance.classes, (int(box[0]), int(box[1] + 40)), cv2.FONT_HERSHEY_COMPLEX,
                            0.8,
                            (0, 255, 0), 2)
                cv2.putText(img_aug, '%dx%d' % (w, h), (int(box[0]), int(box[3] + 40)), cv2.FONT_HERSHEY_COMPLEX,
                            0.8,
                            (0, 255, 0), 2)

            k = cv2.waitKey(0)
            if k == ord('d'):
                cur_node += 1
            if k == ord('a'):
                cur_node -= 1
                if cur_node <= 0:
                    cur_node = 0
            if k == ord('q'):
                cv2.destroyAllWindows()
                break
            cv2.imshow('img', img_aug)
            if flag_show_raw_img:
                cv2.imshow('raw', img)


    def vis_crop_quarter(self,min_iou=0.1,flag_show_raw_img=False,test_img=None):
        '''
        可视化gt after crop， 按键盘d 向前， 按a 向后
        :return:
        '''
        transformer = Transformer()
        cur_node = 0
        set_img_name = list(self.img_instance.keys())  # 所有图片的名称的 list
        flag_set_test_img = False
        while True:
            img_name = set_img_name[cur_node]

            instances_list = self.img_instance[img_name]
            if test_img is not None and flag_set_test_img == False:
                if isinstance(test_img, str):
                    instances_list = self.img_instance[test_img]

                elif isinstance(test_img, edict):
                    if test_img.flag_continue == True:  # 继续读取接下来的 img list
                        test_img_name = test_img.name
                        cur_node = set_img_name.index(test_img_name)  # 设置节点位置
                        instances_list = self.img_instance[test_img_name]
                        flag_set_test_img = True

            # instance attr: bbox, defect_name, name, abs_path, classes, w, h, area

            cv2.namedWindow('img', 0)

            cv2.resizeWindow('img', 1920, 1080)
            print('num gt: ', len(instances_list))
            print('img_name: %s ' % (img_name))

            ins_init = instances_list[0]
            img = cv2.imread(ins_init.abs_path)
            img_aug, _ = transformer(img, ins_init)
            img_h, img_w, c = img_aug.shape

            crop_img = img_aug.copy()


            cv2.rectangle(img_aug, (int(0), int(0)), (int(img_w // 2), int(img_h // 2)), (255, 0, 0), 4)
            cv2.rectangle(img_aug, (int(img_w // 2), int(img_h // 2)), (int(img_w), int(img_h)), (255, 0, 0), 4)
            boxes = []
            labels = []
            for instance in instances_list:
                box = instance.bbox

                boxes.append(box)
                labels.append(instance.classes)

                w, h = compute_wh(box)
                cv2.rectangle(img_aug, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)

                cv2.putText(img_aug, '%d' % instance.classes, (int(box[0]), int(box[1] + 40)), cv2.FONT_HERSHEY_COMPLEX,
                            0.8,
                            (0, 255, 0), 2)
                cv2.putText(img_aug, '%dx%d' % (w, h), (int(box[0]), int(box[3] + 40)), cv2.FONT_HERSHEY_COMPLEX,
                            0.8,
                            (0, 255, 0), 2)

            boxes = np.array(boxes).reshape(-1,4)
            labels = np.array(labels)
            print("before boxes:",boxes)
            print("before labels:",labels)
            patches = [np.array((0,0,img_w//2,img_h//2)),np.array((img_w//2,0,img_w,img_h//2)),
                       np.array((0,img_h//2,img_w//2,img_h)),np.array((img_w//2,img_h//2,img_w,img_h))]
            for patch_idx,patch in enumerate(patches):
                cv2.namedWindow('patch1', 0)     #cv2.WINDOW_AUTOSIZE
                cv2.resizeWindow('patch1', 960, 540)
                cv2.namedWindow('patch2', 0)
                cv2.resizeWindow('patch2', 960, 540)
                cv2.namedWindow('patch3', 0)
                cv2.resizeWindow('patch3', 960, 540)
                cv2.namedWindow('patch4', 0)
                cv2.resizeWindow('patch4', 960, 540)


                temp_boxes = boxes.copy()
                temp_labels = labels.copy()
                overlaps = bbox_overlaps(temp_boxes,patch.reshape(-1,4),mode='iof').reshape(-1)
                print("i:",patch_idx+1,"patch:", patch)
                print("overlaps:",overlaps)
                if overlaps.max()<min_iou:
                    cv2.destroyWindow("patch{}".format(patch_idx + 1))
                    continue
                # Keep those boxes larger than the min_iou
                mask = (overlaps>min_iou)
                temp_boxes = temp_boxes[mask]
                temp_labels = temp_labels[mask]
                patch_img = crop_img[patch[1]:patch[3],patch[0]:patch[2]]


                temp_boxes[:, 2:] = temp_boxes[:, 2:].clip(max=patch[2:])
                temp_boxes[:, :2] = temp_boxes[:, :2].clip(min=patch[:2])
                temp_boxes -= np.tile(patch[:2], 2)

                print("mask:",mask.any(),mask)
                print("after boxes:",temp_boxes)
                print("after labels:",temp_labels)
                for label_idx,box in enumerate(temp_boxes):
                    w, h = compute_wh(box)
                    cv2.rectangle(patch_img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)

                    cv2.putText(patch_img, '%d' % temp_labels[label_idx], (int(box[0]), int(box[1] + 40)),
                                cv2.FONT_HERSHEY_COMPLEX,
                                0.8,
                                (0, 255, 0), 2)
                    cv2.putText(patch_img, '%dx%d' % (w, h), (int(box[0]), int(box[3] + 40)), cv2.FONT_HERSHEY_COMPLEX,
                                0.8,
                                (0, 255, 0), 2)
                cv2.imshow("patch{}".format(patch_idx + 1), patch_img)


            cv2.imshow('img', img_aug)
            if flag_show_raw_img:
                cv2.imshow('raw', img)

            k = cv2.waitKey(0)
            if k == ord('d'):
                cur_node += 1
            if k == ord('a'):
                cur_node -= 1
                if cur_node <= 0:
                    cur_node = 0
            if k == ord('q'):
                cv2.destroyAllWindows()
                break

    def crop_quarter(self,min_iou=0.1,crop_img_save_path=None,
                     crop_json_file_path=None,abn_img_save_path=None,abn_json_file_path=None):
        import shutil
        '''
        generate crop
        :return:
        '''
        transformer = Transformer()
        set_img_name = list(self.img_instance.keys())  # 所有图片的名称的 list
        print("img_name lenth:",len(set_img_name))
        new_all_instances = []
        # i = 0
        abn_instances = []
        for cur_node in range(len(set_img_name)):

            # i += 1
            # if i>=100:break

            img_name = set_img_name[cur_node]

            instances_list = self.img_instance[img_name]

            # instance attr: bbox, defect_name, name, abs_path, classes, w, h, area
            if len(instances_list) > 100:
                continue
            print('num gt: ', len(instances_list))
            print('img_name: %s ' % (img_name))
            # print("ins_list:", instances_list)
            ins_init = instances_list[0]
            img = cv2.imread(ins_init.abs_path)
            img_aug, _ = transformer(img, ins_init)
            img_h, img_w, c = img_aug.shape

            crop_img = img_aug.copy()

            boxes = []
            labels = []
            labelnames = []
            for instance in instances_list:
                box = instance.bbox
                boxes.append(box)
                labels.append(instance.classes)
                labelnames.append(instance.defect_name)
            boxes = np.array(boxes).reshape(-1,4)
            labels = np.array(labels)
            labelnames = np.array(labelnames)
            #judge node inside box
            center_x, center_y = img_w // 2, img_h // 2
            inter_judge = (center_x > boxes[:, 0]) * (center_y > boxes[:, 1]) * \
                          (center_x < boxes[:, 2]) * (center_y < boxes[:, 3])
            if not osp.exists(abn_img_save_path):
                os.makedirs(abn_img_save_path)

            if inter_judge.any():
                shutil.copy(os.path.join(self.cfg.allimg_path, img_name),os.path.join(abn_img_save_path,img_name))
                abn_instances += instances_list
            # print("before boxes:",boxes)
            # print("before labels:",labels)
            patches = [np.array((0,0,img_w//2,img_h//2)),np.array((img_w//2,0,img_w,img_h//2)),
                       np.array((0,img_h//2,img_w//2,img_h)),np.array((img_w//2,img_h//2,img_w,img_h))]
            for patch_idx,patch in enumerate(patches):
                temp_boxes = boxes.copy()
                temp_labels = labels.copy()
                temp_labelnames = labelnames.copy()

                overlaps = bbox_overlaps(temp_boxes,patch.reshape(-1,4),mode='iof').reshape(-1)
                # print("i:",patch_idx+1,"patch:", patch)
                # print("overlaps:",overlaps)
                if overlaps.max()<min_iou:
                    continue
                # Keep those boxes larger than the min_iou
                mask = (overlaps>min_iou)
                temp_boxes = temp_boxes[mask]
                temp_labels = temp_labels[mask]
                temp_labelnames = temp_labelnames[mask]
                patch_img = crop_img[patch[1]:patch[3],patch[0]:patch[2]]

                temp_boxes[:, 2:] = temp_boxes[:, 2:].clip(max=patch[2:])
                temp_boxes[:, :2] = temp_boxes[:, :2].clip(min=patch[:2])
                temp_boxes -= np.tile(patch[:2], 2)

                # print("mask:",mask)
                # print("after boxes:",temp_boxes)
                # print("after labels:",temp_labels)
                save_name = img_name.split('.')[0] + '_patch{}'.format(patch_idx + 1) + '.jpg'
                if not osp.exists(crop_img_save_path):
                    os.makedirs(crop_img_save_path)
                img_path = os.path.join(crop_img_save_path, save_name)
                # print("img_path:",img_path)
                for label_idx,box in enumerate(temp_boxes):
                    w, h = compute_wh(box)

                    new_instance = edict()
                    new_instance.name = save_name
                    new_instance.defect_name = temp_labelnames[label_idx]
                    new_instance.bbox = box.tolist()
                    new_instance.classes = temp_labels[label_idx]  # add classes int
                    new_instance.w = round(w, 2)  # add w
                    new_instance.h = round(h, 2)  # add h
                    new_instance.area = round(w * h, 2)  # add area
                    new_instance.abs_path = img_path  # add 绝对路径
                    new_instance.im_w = patch_img.shape[1]
                    new_instance.im_h = patch_img.shape[0]
                    # print("ins:",new_instance)
                    new_all_instances.append(new_instance)
                cv2.imwrite(img_path,patch_img)

        with open(crop_json_file_path, 'w') as f:
            json.dump(new_all_instances, f, indent=4, separators=(',', ': '), cls=MyEncoder)
        with open(abn_json_file_path, 'w') as f:
            json.dump(abn_instances, f, indent=4, separators=(',', ': '), cls=MyEncoder)



    def vis_ins_list(self, img, ins_list, flag_show_raw_img=False):
        '''
        可视化gt， 按键盘d 向前， 按a 向后
        :return:
        '''
        transformer = Transformer()
        cur_node = 0

        while True:
            # img_name = set_img_name[cur_node]
            instances_list = ins_list

            # instance attr: bbox, defect_name, name, abs_path, classes, w, h, area

            cv2.namedWindow('img', 0)
            pad_size = 200
            cv2.resizeWindow('img', 1333 + pad_size * 2, 800 + pad_size * 2)
            print('num gt: ', len(instances_list))

            ins_init = instances_list[0]
            # img_resize = cv2.resize(img, (1333,800))
            img_aug, _ = transformer(img, ins_init)
            for instance in instances_list:
                box = instance.bbox
                w, h = compute_wh(box)
                cv2.rectangle(img_aug, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)
                cv2.putText(img_aug, '%d ' % instance.classes, (int(box[0]), int(box[1] )),
                            cv2.FONT_HERSHEY_COMPLEX, 0.8,
                            (0, 255, 0), 1)

                cv2.putText(img_aug, '%d x %d' % (w, h), (int(box[0]), int(box[3] + 20)), cv2.FONT_HERSHEY_COMPLEX,
                            0.8, (0, 255, 0), 1)

            k = cv2.waitKey(0)
            if k == ord('d'):
                cur_node += 1
            if k == ord('a'):
                cur_node -= 1
                if cur_node <= 0:
                    cur_node = 0
            if k == ord('q'):
                cv2.destroyAllWindows()
                break
            cv2.imshow('img', img_aug)
            if flag_show_raw_img:
                cv2.imshow('raw', img)

    def vis_res(self, flag_show_raw_img=False, test_name=None):
        '''
        可视化gt， 按键盘d 向前， 按a 向后
        :return:
        '''
        res_ins_list = self.load_res_json(self.cfg.submit_json)
        all_instance, cla_instance, img_instance = self._create_data_dict(res_ins_list, self.cfg.submit_path, flag_ins_list=True)
        transformer = Transformer()
        cur_node = 0

        set_img_name = list(img_instance.keys())  # 所有图片的名称的 list
        while True:
            img_name = set_img_name[cur_node]
            instances_list = img_instance[img_name]
            if test_name is not None:
                instances_list = img_instance[test_name]

            # instance attr: bbox, defect_name, name, abs_path, classes, w, h, area

            cv2.namedWindow('img', 0)
            cv2.resizeWindow('img', 1920, 1080)
            print('num res : ', len(instances_list))
            print('img_name: %s ' % (img_name))

            ins_init = instances_list[0]
            img = cv2.imread(ins_init.abs_path)
            img_aug, _ = transformer(img, ins_init)
            for instance in instances_list:
                box = instance.bbox
                cv2.rectangle(img_aug, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)
                cv2.putText(img_aug, '%d %0.3f' % (instance.classes, instance.score), (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_COMPLEX,
                            0.8,
                            (0, 255, 0), 1)

            k = cv2.waitKey(0)
            if k == ord('d'):
                cur_node += 1
            if k == ord('a'):
                cur_node -= 1
                if cur_node <= 0:
                    cur_node = 0
            if k == ord('q'):
                cv2.destroyAllWindows()
                break
            cv2.imshow('img', img_aug)
            if flag_show_raw_img:
                cv2.imshow('raw', img)

    def visRes_gt(self, gt_img_ins, res_img_ins, gtline=0.8,resline=0.8):
        '''
        1. 可视化gt 和 result 效果

        :param gt_img_ins: gt 的img_instance
        :param res_img_ins:  result 的img_instance
        :return:
        '''
        empty_ins = [
            edict(
                {'abs_path': '',
                 'area': 1,
                 'bbox': [0,0,0,0],
                 'classes': -1,
                 'defect_name': '',
                 'h': 0,
                 'name': '',
                 'w': 0,
                 'score':0}
            )
        ]

        cur_node = 0
        set_img_name = list(gt_img_ins.keys())  # 所有图片的名称的 list
        while True:
            img_name = set_img_name[cur_node]
            gt_ins_list = gt_img_ins[img_name] # gt instance 列表
            if img_name in res_img_ins.keys():
                res_ins_list = res_img_ins[img_name] # result
                res_num = len(res_ins_list)
            else:
                res_ins_list = empty_ins
                res_num = 0
            # instance attr: bbox, defect_name, name, abs_path, classes, w, h, area

            cv2.namedWindow('img', 0)
            cv2.resizeWindow('img', 1920, 1080)


            ins_init = gt_ins_list[0]
            img = cv2.imread(ins_init.abs_path)

            for gt_ins in gt_ins_list :
                gt_box = gt_ins.bbox

                # 绘制gt
                cv2.rectangle(img, (int(gt_box[0]), int(gt_box[1])), (int(gt_box[2]), int(gt_box[3])), (0, 0, 255), 2)
                cv2.putText(img, '%d' % gt_ins.classes, (int(gt_box[0]), int(gt_box[1]+40)), cv2.FONT_HERSHEY_COMPLEX,gtline,
                            (0, 0, 255), 1)

            for res_ins in res_ins_list:
                res_box = res_ins.bbox

                # 绘制result
                cv2.rectangle(img, (int(res_box[0]), int(res_box[1])), (int(res_box[2]), int(res_box[3])), (0, 255, 0), 2)
                cv2.putText(img, '%d %0.3f' % (res_ins.classes, res_ins.score), (int(res_box[2]), int(res_box[1]+40)), cv2.FONT_HERSHEY_COMPLEX, resline,
                            (0, 255, 0), 1)

            k = cv2.waitKey(0)
            if k == ord('d'):
                cur_node += 1
            if k == ord('a'):
                cur_node -= 1
                if cur_node <= 0:
                    cur_node = 0
            if k == ord('q'):
                cv2.destroyAllWindows()
                break
            print('img_name: %s ' % (img_name))
            print('num gt : ', len(gt_ins_list))
            print('num res: ', res_num)
            cv2.imshow('img', img)
        pass

    def val_analyze(self, flag_coco_json=False):
        '''
        1. val 作为gt的 instance
        2. 模型在val 上的输出 instace
        3. 可视化比较
        4. 指标比较
        :return:q
        '''
        if not (hasattr(self.cfg, 'result_json') and self.cfg.result_json != ''):
            raise(" no result_json ")

        if flag_coco_json :
            res_ins_list = self.load_res_json(self.cfg.result_json)
            valr_all_instance, valr_cla_instance, valr_img_instance = self._create_data_dict(res_ins_list, self.cfg.val_img_path, flag_ins_list=True)
        else:
            valr_all_instance, valr_cla_instance, valr_img_instance = self._create_data_dict(self.cfg.result_json,
                                                                                             self.cfg.val_img_path,
                                                                                             flag_ins_list=False)

        self.val_all_instance, self.val_cla_instance, self.val_img_instance = self._create_data_dict(self.cfg.val_json_paths,
                                                                                                    self.cfg.val_img_path)
        self.visRes_gt(self.val_img_instance, valr_img_instance)

    def load_res_json(self, path):
        '''
        结果json 转换为 raw json 方式，
        增加 defect_name
        :param path:
        :return:
        '''

        raw_ins_list = []
        for i in range(len(path)):
            ins_list = json.load(open(path[i], 'r'))
            for instance in ins_list:
                instance = edict(instance)
                instance.defect_name = self.reverse_category[instance.category]

                raw_ins_list.append(instance)

        return raw_ins_list

    def load_coco_format(self, json_path, data_file):
        all_instance = []
        key_classes = list(range(1, self.num_classes + 1))  # 1 ... num_classes

        cla_instance = edict({str(k): [] for k in key_classes})  # key 必须是字符串
        img_instance = edict()
        if isinstance(json_path, str):
            json_path = [json_path]

        if isinstance(json_path, list):
            for path in json_path:
                gt_dict = json.load(open(path, 'r'))
                im_anno = edict()
                bbox_anno = edict()
                for anno in gt_dict['images']:
                    anno = edict(anno)
                    im_name = anno['file_name']
                    image_id = anno['id']
                    width = anno['width']
                    height = anno['height']
                    if str(image_id) not in im_anno.keys():
                        im_anno[str(image_id)] = {'im_name':im_name,'width':width,'height':height}
                for anno in gt_dict['annotations']:
                    image_id = anno['image_id']
                    category_id = anno['category_id']
                    bbox = anno['bbox']
                    bbox[2] = bbox[0]+bbox[2]
                    bbox[3] = bbox[1]+bbox[3]
                    area = anno['area']
                    if str(image_id) not in bbox_anno.keys():
                        bbox_anno[str(image_id)] = [{'category_id':category_id,'bbox':bbox,'area':area}]
                    else:
                        bbox_anno[str(image_id)].append({'category_id': category_id, 'bbox': bbox, 'area': area})

                for image_id in im_anno.keys():
                    im_anno_temp = im_anno[image_id]
                    bbox_anno_temp = bbox_anno[image_id]
                    for bbox in bbox_anno_temp:
                        instance = edict()
                        instance.imgid = image_id
                        instance.name = im_anno_temp['im_name']
                        instance.im_w = im_anno_temp['width']
                        instance.im_h = im_anno_temp['height']
                        instance.area = bbox['area']
                        instance.bbox = bbox['bbox']
                        instance.w = bbox['bbox'][2] - bbox['bbox'][0]
                        instance.h = bbox['bbox'][3] - bbox['bbox'][1]
                        instance.classes = bbox['category_id']
                        instance.abs_path = osp.join(data_file, im_anno_temp['im_name'])  # add 绝对路径
                        instance.defect_name = self.reverse_category[bbox['category_id']]
                        all_instance.append(instance)
                        cla_instance[str(bbox['category_id'])].append(instance)  # 每类的instance

                        if instance.name not in img_instance.keys():  # 每张图片的instance
                            img_instance[instance.name] = [instance]
                        else:
                            img_instance[instance.name].append(instance)
                return all_instance, cla_instance, img_instance

    def divide_trainval(self, ratio=0.2, del_json=None, del_path=None, train_json='', val_json=''):
        import random

        train_ins_list = []
        val_ins_list = []
        if del_json is None:
            divide_jsons = self.cfg.divide_json
        if del_path is None:
            del_path = self.cfg.allimg_path

        if isinstance(divide_jsons, str):
            divide_jsons = [divide_jsons]
        if not isinstance(divide_jsons, list):
            raise ("divide_jsons error !!")

        # for divide_json in divide_jsons:
        all_instance, cla_instance, img_instance = self._create_data_dict(divide_jsons, del_path)
        all_ins_keys = set(img_instance.keys())
        num_ins = len(list(all_ins_keys))
        num_val = int(num_ins  * ratio)
        print("total num : " ,num_ins)
        print("val   num : " ,num_val)
        print("train num : " ,num_ins - num_val)

        val_ins_keys = random.sample(all_ins_keys, num_val)
        train_ins_keys = set(all_ins_keys) - set(val_ins_keys)
        for ins_key in all_ins_keys:
            if ins_key in val_ins_keys:
                val_ins_list += (img_instance[ins_key])
            elif ins_key in train_ins_keys:
                train_ins_list += (img_instance[ins_key])

        with open(train_json, 'w') as f:
            json.dump(train_ins_list, f, indent=4, separators=(',', ': '), cls=MyEncoder)
            print('train json save : ', train_json)
        with open(val_json, 'w') as f:
            json.dump(val_ins_list, f, indent=4, separators=(',', ': '), cls=MyEncoder)
            print('val json save : ', val_json)

    def pick_gt(self, normal_Images, img_save, json_file_path, condition=None, transcfg=None):


        '''
        1. 判断图片中instance 进行抠gt的条件
        2. 抠取gt得到 新的 gt图， (gt 个数， flip 等操作)
        3. (根据条件替代原来gt图(gt太少))gt 图加入到batch 中进行训练

         all_instance
        [
            {'bbox': [2000.66, 326.38, 2029.87, 355.59],
             'defect_name': '结头',
             'name': 'd6718a7129af0ecf0827157752.jpg',
             'abs_path' : 'xxx/xxx.jpg',
             'classes': 1
             'w':1,
             'h':1,
             'area':1,
             }
        ]

        :param results:
        :return:
        '''
        def _get_gtroi(list_ins, condition):
            init_ins = list_ins[0]
            img = cv2.imread(init_ins.abs_path)
            h, w, c = img.shape
            gt_roi_list = []
            new_list_ins = _check_pick_gt(list_ins, condition=condition)

            for ind, instance in enumerate(new_list_ins ):
                pick_ins = edict()
                x1, y1, x2, y2 = [round(i) for i in instance.bbox]  # x1, y1, x2, y2
                pick_ins.im_w = int(w)
                pick_ins.im_h = int(h)
                pick_ins.roi = img[y1:y2, x1:x2, :]
                pick_ins.w = x2 - x1
                pick_ins.h = y2 - y1
                pick_ins.classes = instance.classes
                pick_ins.defect_name = instance.defect_name
                pick_ins.area = pick_ins.w * pick_ins.h

                gt_roi_list.append(pick_ins)
            return gt_roi_list

        def get_normal_img(imgs_path):
            imgs = []
            if isinstance(imgs_path, str):
                imgs_path = [imgs_path]
            if isinstance(imgs_path, list):

                for imgs_p in imgs_path:
                    imgs += [osp.join(imgs_p, l) for l in os.listdir(imgs_p)]
            return imgs

        def compute_iou(rec1, rec2):
            """
            computing IoU
            :param rec1: (x0, y0, x1, y1), which reflects
                    (top, left, bottom, right)
            :param rec2: (x0, y0, x1, y1)
            :return: scala value of IoU
            """
            # computing area of each rectangles
            S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
            S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])

            # computing the sum_area
            sum_area = S_rec1 + S_rec2

            # find the each edge of intersect rectangle
            top_line = max(rec1[0], rec2[0])
            left_line = max(rec1[1], rec2[1])
            bottom_line = min(rec1[2], rec2[2])
            right_line = min(rec1[3], rec2[3])

            # judge if there is an intersect
            if left_line >= right_line or top_line >= bottom_line:
                return 0
            else:
                intersect = (right_line - left_line) * (bottom_line - top_line)
                return (intersect / (sum_area - intersect)) * 1.0

        def judeg_big(instance, condition=None):
            flag = False
            img_scale = condition.img_scale
            if instance.w / instance.im_w >= img_scale or instance.h / instance.im_h > img_scale:
                flag = True
            return flag

        def _check_pick_gt(img_ins, condition):
            new_img_ins = []
            mix_matrix = np.zeros((len(img_ins), len(img_ins)))
            for i in range(len(img_ins)):
                instance = img_ins[i]
                for j in range(len(img_ins)):
                    instance_temp = img_ins[j]
                    iou = compute_iou(instance.bbox,instance_temp.bbox)
                    mix_matrix[i][j] = iou
                if judeg_big(instance,condition):
                    mix_matrix[i][i] = 1
                else:
                    mix_matrix[i][i] = 0

            for i in range(len(img_ins)):
                col = mix_matrix[:,i]
                row = mix_matrix[i,:]
                if sum(col) == 0 and sum(row) == 0:
                    new_img_ins.append(img_ins[i])

            return new_img_ins

        def _get_pick_gt(all_ins, condition, transform=None, transcfg=None):
            gt_ins_list = []
            num = random.randint(condition.min_num_per_image, condition.max_num_per_image)
            num = min(num, len(all_ins))
            for i in range(num):
                gt_ins_temp = transform.aug_ins(all_ins[0], transcfg)
                gt_ins_list.append(gt_ins_temp)
                all_ins.pop(0)
            return gt_ins_list


        def compute_list_iou(ins, ins_list):
            mix_matrix = np.zeros((1, len(ins_list)))
            for i in range(len(ins_list)):
                mix_matrix[0, i] = compute_iou(ins.bbox, ins_list[i].bbox)
            return mix_matrix

        def _changebg(img, ins_list, img_info, info=None):
            '''
            1. 可以粘贴区域
            2. 去除有 iou box ， 循环n次 删除改box
                pick_ins.roi = img[y1:y2, x1:x2, :]
                pick_ins.w = instance.w
                pick_ins.h = instance.h
                pick_ins.area = instance.area
            :param img:
            :param ins_list:
            :param img_info:
            :return:
            '''
            if info is None:
                info = edict()
                info.pick = 0.5
                info.ins = 0.5

            img_tmp = copy.deepcopy(img)
            img_info.im_h, img_info.im_w, c = img_tmp.shape


            loop_times = 10
            normal_ins = []
            for ins in ins_list:
                new_ins = edict()
                paste_roi = np.asarray([0, 0, img_info.im_w - ins.w, img_info.im_h - ins.h])  # 在paste roi 中选取一点进行贴图
                if ins.w > img_info.im_w or ins.h > img_info.im_h:
                    continue
                x1s = np.random.randint(paste_roi[0], paste_roi[2], loop_times)  # 产生 loop_times个 随机点，
                y1s = np.random.randint(paste_roi[1], paste_roi[3], loop_times)

                new_ins.im_w = ins.im_w
                new_ins.im_h = ins.im_h
                new_ins.w = ins.w
                new_ins.h = ins.h
                new_ins.classes = ins.classes
                new_ins.defect_name = ins.defect_name
                new_ins.area = ins.area
                new_ins.name = img_info.name
                new_ins.abs_path = img_info.abs_path
                # new_ins.h = ins.h
                # new_ins.area = ins.w * ins.h
                new_ins.bbox = [round(l) for l in [x1s[0], y1s[0], ins.w + x1s[0], ins.h + y1s[0]]]
                if len(normal_ins) == 0:
                    x1, y1, x2, y2 = new_ins.bbox
                    pick = img_tmp[y1:y2, x1:x2, :]
                    img_tmp[y1:y2, x1:x2, :] = cv2.addWeighted(pick, info.pick, ins.roi, info.ins, 0)
                    # ins.roi
                    normal_ins.append(new_ins)
                else:

                    for i in range(1, loop_times):
                        iou_m = compute_list_iou(new_ins, normal_ins)  # 计算iou
                        if iou_m.max() == 0:  # 有iou 交叠重新选择 ins
                            x1, y1, x2, y2 = new_ins.bbox
                            pick = img_tmp[y1:y2, x1:x2, :]
                            img_tmp[y1:y2, x1:x2, :] = cv2.addWeighted(pick, info.pick, ins.roi, info.ins, 0)
                            normal_ins.append(new_ins)
                            break
                        new_ins.bbox = [x1s[i], y1s[i], ins.w + x1s[i], ins.h + y1s[i]]
                    # 超过loop times 不append

            return img_tmp, normal_ins


        if not osp.exists(img_save):
            os.makedirs(img_save)

        transformer = Transformer()


        all_pickgt = []
        all_ins = self.img_instance
        normal_imgs_path =  get_normal_img(normal_Images)
        # 1. 是否要进行抠图
        i = 0
        for img_name, list_ins in tqdm(all_ins.items()):
            i += 1
            if i > 700: break
            all_pickgt += _get_gtroi(list_ins, condition=condition.pickgt)

        # 2. 换背景
        bg_all_ins = []
        all_pickgt_temp = copy.deepcopy(all_pickgt)
        # all_pickgt_temp = all_pickgt
        for ind , back_img_p in enumerate(tqdm(normal_imgs_path)):
            img_info = edict()
            name = osp.basename(back_img_p).replace('.jpg', '_%d.jpg' % (ind))
            bg_name = osp.join(img_save, name)
            # change instance
            img_info.name = name
            img_info.abs_path = bg_name

            back_img = cv2.imread(back_img_p)
            if len(all_pickgt_temp):
                gt_ins_list = _get_pick_gt(all_pickgt_temp, condition=condition.pickgt,
                                           transform=transformer, transcfg=transcfg)  # 获得 instanddddddddqdqdqddce list
            else:
                random.shuffle(all_pickgt)
                all_pickgt_temp = copy.deepcopy(all_pickgt)
                gt_ins_list = _get_pick_gt(all_pickgt_temp, condition=condition.pickgt,
                                           transform=transformer, transcfg=transcfg)  # 获得 instance list
            # if len(all_pickgt_temp) == 0:
            #     break
            bg_img, instance = _changebg(back_img, gt_ins_list, img_info, condition.add_info)

            # self.vis_ins_list(bg_img, instance)
            bg_all_ins += instance


            cv2.imwrite(bg_name, bg_img)

        random.shuffle(bg_all_ins)
        with open(json_file_path, 'w') as f:
            json.dump(bg_all_ins, f, indent=4, separators=(',', ': '), cls=MyEncoder)

    def view_diff(self, path):
        from skimage.measure import compare_ssim
        dirs = os.listdir(path)
        for dir in dirs:
            root = os.path.join(path,dir)
            pre_name = dir.split('_')[0]
            temp_name = 'template_'+pre_name+'.jpg' # 无瑕疵图片名
            def_name = dir+'.jpg'                            # 有瑕疵图片名
            im_temp = cv2.imread(os.path.join(root, temp_name))
            im_def = cv2.imread(os.path.join(root, def_name))
            # im_diff = im_def - im_temp
            gray_im_temp = cv2.cvtColor(im_temp,cv2.COLOR_BGR2GRAY)
            gray_im_def = cv2.cvtColor(im_def,cv2.COLOR_BGR2GRAY)
            (score, diff) = compare_ssim(gray_im_temp, gray_im_def, full=True)
            diff = (diff * 255).astype("uint8")
            cv2.putText(diff, str(score), (30, 30),cv2.FONT_HERSHEY_COMPLEX, 0.8,(0, 255, 0), 1)
            anno = self.img_instance[def_name]
            for ins in anno:
                box = ins.bbox
                defect_name = ins.defect_name
                # 在有瑕疵和diff图片上画上瑕疵位置
                cv2.rectangle(im_def, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0),
                              2)
                cv2.rectangle(diff, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0),
                              2)
            cv2.namedWindow("im_temp", 0)
            cv2.resizeWindow('im_temp', 1024, 512)
            cv2.imshow("im_temp", im_temp)
            cv2.namedWindow("im_def", 0)
            cv2.resizeWindow('im_def', 1024, 512)
            cv2.imshow("im_def", im_def)
            cv2.namedWindow("diff", 0)
            cv2.resizeWindow('diff', 1024, 512)
            cv2.imshow("diff", diff)
            cv2.waitKey(0)

    def crop_quarter2(self,min_iou=0.1,flag_show_raw_img=False,test_img=None):
        '''
        可视化gt after crop， 按键盘d 向前， 按a 向后
        :return:
        '''
        transformer = Transformer()
        cur_node = 0
        set_img_name = list(self.img_instance.keys())  # 所有图片的名称的 list
        flag_set_test_img = False
        while True:
            img_name = set_img_name[cur_node]

            instances_list = self.img_instance[img_name]
            if test_img is not None and flag_set_test_img == False:
                if isinstance(test_img, str):
                    instances_list = self.img_instance[test_img]

                elif isinstance(test_img, edict):
                    if test_img.flag_continue == True:  # 继续读取接下来的 img list
                        test_img_name = test_img.name
                        cur_node = set_img_name.index(test_img_name)  # 设置节点位置
                        instances_list = self.img_instance[test_img_name]
                        flag_set_test_img = True

            # instance attr: bbox, defect_name, name, abs_path, classes, w, h, area

            cv2.namedWindow('img', 0)

            cv2.resizeWindow('img', 1920, 1080)
            print('num gt: ', len(instances_list))
            print('img_name: %s ' % (img_name))

            ins_init = instances_list[0]
            img = cv2.imread(ins_init.abs_path)
            img_aug, _ = transformer(img, ins_init)
            img_h, img_w, c = img_aug.shape

            crop_img = img_aug.copy()


            cv2.rectangle(img_aug, (int(0), int(0)), (int(img_w // 2), int(img_h // 2)), (255, 0, 0), 4)
            cv2.rectangle(img_aug, (int(img_w // 2), int(img_h // 2)), (int(img_w), int(img_h)), (255, 0, 0), 4)
            boxes = []
            labels = []
            for instance in instances_list:
                box = instance.bbox

                boxes.append(box)
                labels.append(instance.classes)

                w, h = compute_wh(box)
                cv2.rectangle(img_aug, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)

                cv2.putText(img_aug, '%d' % instance.classes, (int(box[0]), int(box[1] + 40)), cv2.FONT_HERSHEY_COMPLEX,
                            0.8,
                            (0, 255, 0), 2)
                cv2.putText(img_aug, '%dx%d' % (w, h), (int(box[0]), int(box[3] + 40)), cv2.FONT_HERSHEY_COMPLEX,
                            0.8,
                            (0, 255, 0), 2)

            boxes = np.array(boxes).reshape(-1,4)
            labels = np.array(labels)
            print("before boxes:",boxes)
            print("before labels:",labels)
            patches = [np.array((0,0,img_w//2,img_h//2)),np.array((img_w//2,0,img_w,img_h//2)),
                       np.array((0,img_h//2,img_w//2,img_h)),np.array((img_w//2,img_h//2,img_w,img_h))]
            for patch_idx,patch in enumerate(patches):
                cv2.namedWindow('patch1', 0)     #cv2.WINDOW_AUTOSIZE
                cv2.resizeWindow('patch1', 960, 540)
                cv2.namedWindow('patch2', 0)
                cv2.resizeWindow('patch2', 960, 540)
                cv2.namedWindow('patch3', 0)
                cv2.resizeWindow('patch3', 960, 540)
                cv2.namedWindow('patch4', 0)
                cv2.resizeWindow('patch4', 960, 540)


                temp_boxes = boxes.copy()
                temp_labels = labels.copy()
                overlaps = bbox_overlaps(temp_boxes, patch.reshape(-1,4),mode='iof').reshape(-1)
                print("i:",patch_idx+1,"patch:", patch)
                print("overlaps:",overlaps)
                if overlaps.max()<min_iou:
                    cv2.destroyWindow("patch{}".format(patch_idx + 1))
                    continue
                # Keep those boxes larger than the min_iou
                mask = (overlaps>min_iou)
                temp_boxes = temp_boxes[mask]
                temp_labels = temp_labels[mask]
                patch_img = crop_img[patch[1]:patch[3],patch[0]:patch[2]]


                temp_boxes[:, 2:] = temp_boxes[:, 2:].clip(max=patch[2:])
                temp_boxes[:, :2] = temp_boxes[:, :2].clip(min=patch[:2])
                temp_boxes -= np.tile(patch[:2], 2)

                print("mask:",mask.any(),mask)
                print("after boxes:",temp_boxes)
                print("after labels:",temp_labels)
                for label_idx,box in enumerate(temp_boxes):
                    w, h = compute_wh(box)
                    cv2.rectangle(patch_img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)

                    cv2.putText(patch_img, '%d' % temp_labels[label_idx], (int(box[0]), int(box[1] + 40)),
                                cv2.FONT_HERSHEY_COMPLEX,
                                0.8,
                                (0, 255, 0), 2)
                    cv2.putText(patch_img, '%dx%d' % (w, h), (int(box[0]), int(box[3] + 40)), cv2.FONT_HERSHEY_COMPLEX,
                                0.8,
                                (0, 255, 0), 2)
                cv2.imshow("patch{}".format(patch_idx + 1), patch_img)


            cv2.imshow('img', img_aug)
            if flag_show_raw_img:
                cv2.imshow('raw', img)

            k = cv2.waitKey(0)
            if k == ord('d'):
                cur_node += 1
            if k == ord('a'):
                cur_node -= 1
                if cur_node <= 0:
                    cur_node = 0
            if k == ord('q'):
                cv2.destroyAllWindows()
                break

    def prepara_cut_pic(self, num_thre, iou_thre):
        '''
        {'bbox': [2000.66, 326.38, 2029.87, 355.59],
                 'defect_name': '结头',
                 'name': 'd6718a7129af0ecf0827157752.jpg',
                 'abs_path' : 'xxx/xxx.jpg',
                 'w':1,
                 'h':1,
                 'area':1,
                 'im_w':1
                 'im_h':2
                 }

        预切割 划分为 3个文件夹
            1. 大gt 包含小gt
            2. 大gt 不包含小gt
            3. 小gt被切割
            4. 可以被切割的
        :return:
        '''

        def check_biggt(ins_list):
            num_smgt = 0
            ins_init = ins_list[0]
            im_w, im_h = ins_init.im_w, ins_init.im_h
            flag_has_big = False
            for ins in ins_list:
                if (ins.w < (im_w // 2)) or (ins.h < (im_h // 2)): # 小gt
                    num_smgt += 1
                else:
                    flag_has_big = True

            return flag_has_big, num_smgt


        def check_cut_iou(ins_list):
            # 0. gt 与 切割线重合 break
            # 1. 横纵切割线 (0,im_h//2) (im_w, im_h//2) 与 ins 的对角线有交点，判断为裁剪box
            # 2. 被切到的gt 被line_h 切割对 得到x1 和line-h 交A， x2 和line-h 交B，
            #    新gt为 (x1，y1, B) (A, x2,y2)
            # 3. 被切到的gt 被line_w 切割对 得到y1 和line_w 交A， y2 和line_w 交B，
            #    新gt为 (x1，y1, B) (A, x2,y2)
            # 4. ins 和之前进行比较得到iou 矩阵
            ins_init = ins_list[0]
            im_w, im_h = ins_init.im_w, ins_init.im_h
            line_w = [0, im_h//2, im_w, im_h//2] # x1,y1,x2,y2
            line_h = [im_w//2, 0, im_w//2, im_h] # x1,y1,x2,y2

            w_cut_ins = [edict(raw=None, cut=None)]
            h_cut_ins = [edict(raw=None, cut=None)]

            def get_cut_ins(ins, line):
                x1, y1, x2, y2 = ins.bbox
                A = findIntersection([x1,y1,x1,y2], line)
                B = findIntersection([x2,y1,x2,y2], line)
                ins_cutA = copy.deepcopy(ins)
                ins_cutB = copy.deepcopy(ins)
                ins_cutA.bbox = [x1, y1, B[0], B[1]]
                ins_cutB.bbox = [A[0], A[1], x2, y2]

                return ins, [ins_cutA, ins_cutB]

            for ins in ins_list:
                intersect_p = findIntersection(ins.bbox, line_w) # 交点, 无交点返回None
                # 1. 是否被切割： 判断box对角线和 切割线是否有交
                if intersect_p is not None:
                    raw_ins, cut_ins_w = get_cut_ins(ins, line_w)

                    w_cut_ins.append(edict(raw=raw_ins, cut=cut_ins_w,))

                    raw_ins, cut_ins_h = get_cut_ins(ins, line_h)
                    h_cut_ins.append(edict(raw=raw_ins, cut=cut_ins_h))

            return

        # https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection
        def findIntersection(lineA, lineB):
            [x1, y1, x2, y2], [x3, y3, x4, y4] = lineA, lineB
            try:
                px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / (
                            (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
                py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / (
                            (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
            except:
                return None
            return [px, py]

        # save_path = ['biggt_wsm', 'biggt', 'cut_sm_gt', 'correct']
        # save_root = ''
        cut_ins_list = edict(biggt_wsm=[], biggt=[], cut_sm_gt=[], correct=[])

        set_img_name = list(self.img_instance.keys())  # 所有图片的名称的 list
        for img_name in set_img_name:
            instances_list = self.img_instance[img_name]
            img_ins_dict = {img_name: instances_list} # {xxx.jpg: [ins, ins1]}

            # 1. 判断大gt
            flag_big, num_sm = check_biggt(instances_list)
            if flag_big:
                # 2. 判断是否包含较多小gt > 5~10
                if num_sm > num_thre: # 大于阈值为True

                    cut_ins_list.biggt_wsm.append(img_ins_dict)
                else:
                    cut_ins_list.biggt.append(img_ins_dict)
            else:
                # 3. 判断小gt切割后满足阈值
                after_cut_iou = check_cut_iou(instances_list)
                if after_cut_iou < iou_thre:
                    cut_ins_list.cut_sm_gt.append(img_ins_dict)
                else:
                    cut_ins_list.correct.append(img_ins_dict)

    def compute_mAP(self,gt_json_file=None,pred_json_file=None,val_img_path=None,gt_voc_file='val_gt/',pred_txt_file='val_pred/'):
        from voc_eval import voc_eval

        gt_all_instance, gt_cla_instance, gt_img_instance = self._create_data_dict(gt_json_file, val_img_path)
        pred_res_ins_list = self.load_res_json(pred_json_file)
        pred_all_instance, pred_cla_instance, pred_img_instance = self._create_data_dict(pred_res_ins_list, val_img_path,flag_ins_list=True)

        gt_set_img_name = list(gt_img_instance.keys())  # 所有图片的名称的 list
        #generate gt voc format and val.txt, val.txt record val image's name
        if not os.path.exists(gt_voc_file):
            os.mkdir(gt_voc_file)
            for img_name in gt_set_img_name:
                xml_path = os.path.join(gt_voc_file, img_name.replace('jpg', 'xml'))
                instances_list = gt_img_instance[img_name]
                ob2 = ''
                for ins in instances_list:
                    bbox = ins['bbox']
                    defect_name = ins['classes']
                    x1, y1, x2, y2 = bbox
                    ob2 += '\n' + s1.format(defect_name, x1, y1, x2, y2)
                w, h = ins['im_w'], ins['im_h']
                ob1 = s2.format(img_name, w, h, ob2)
                with open(xml_path, 'w') as f:
                    f.write(ob1)
        if not os.path.exists('val.txt'):
            with open('val.txt','a') as fw:
                for line in gt_set_img_name:
                    line = line.split('.')[0]+'\n'
                    fw.write(line)
        #generate predict result by txt format
        if os.path.exists(pred_txt_file):
            shutil.rmtree(pred_txt_file)
        os.mkdir(pred_txt_file)

        for key,values in pred_cla_instance.items():
            txt_name = key+'.txt'
            txt_path = os.path.join(pred_txt_file,txt_name)
            with open(txt_path,'a') as fw:
                for value in values:
                    file_name = value['name'].split('.')[0]
                    save_content = file_name + ' ' + str(value['score']) + ' ' + ' '.join(map(str,value['bbox']))+'\n'
                    fw.write(save_content)
        #compute mAP for per class
        voc_names = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15']
        iou_thr = [0.1, 0.3, 0.5]
        sample_sum = 0
        fpr_90_sum = 0
        fpr_75_sum = 0
        fpr_50_sum = 0
        mean_total_mAP = 0
        save_txt_path = 'result.txt'
        content = ''
        for idx, thr in enumerate(iou_thr):
            ap_sum = 0

            for i in range(len(voc_names)):
                rec, prec, fpr, ap = voc_eval(os.path.join(pred_txt_file,'{}.txt'),
                                              os.path.join(gt_voc_file,'{}.xml'),
                                              'val.txt', voc_names[i], './', ovthresh=thr)
                print("class:", voc_names[i], ", ap:", ap)  # , ", rec:", rec, ", prec:", prec
                if isinstance(rec, int):
                    continue
                _fpr_90 = fpr[rec >= 0.90]
                _fpr_75 = fpr[rec >= 0.75]
                _fpr_50 = fpr[rec >= 0.50]

                fpr_limit = 2
                fpr_90 = _fpr_90[0] if len(_fpr_90) > 0 else fpr_limit
                fpr_75 = _fpr_75[0] if len(_fpr_75) > 0 else fpr_limit
                fpr_50 = _fpr_50[0] if len(_fpr_50) > 0 else fpr_limit

                n_sample = len(rec)
                content += str(ap) + '\t'
                fpr_90_sum += fpr_90 * n_sample
                fpr_75_sum += fpr_75 * n_sample
                fpr_50_sum += fpr_50 * n_sample
                # ap_sum += ap * n_sample
                sample_sum += n_sample
                ap_sum += ap
                a = plt.figure(num=idx + 1, figsize=(10, 30))
                plt.subplot(4, 5, i + 1)
                # plt.subplot2grid((4, 5), (i // 5, i % 5))
                plt.xlim((-0.1, 1.0))
                plt.ylim((0.2, 1.2))
                plt.xlabel('recall')
                plt.ylabel('precision')
                plt.plot(rec, prec, lw=2, color='r')
                plt.title("{}, ap:{:.3f}".format(voc_names[i], ap))
                plt.subplots_adjust(left=0.1, top=0.9, right=0.9, bottom=0.1, wspace=0.5, hspace=2)
                # plt.savefig("%d.png" % (i+1))
                # b = plt.figure(num=2,figsize=(15,15))
                # plt.subplot(7, 5, i + 1)
                # # plt.subplot2grid((4, 5), (i // 5, i % 5))
                # plt.xlim((-0.1, 1.0))
                # plt.ylim((-0.1, 2.0))
                # plt.xlabel('recall')
                # plt.ylabel('fpr')
                # plt.plot(rec, fpr, lw=2, color='darkorange')
                # plt.title("{}".format(voc_names[i]))
                # plt.subplots_adjust(left=0.1, top=0.9, right=0.9, bottom=0.1, wspace=0.3, hspace=0.8)
            # a.savefig("1.png")
            # b.savefig("2.png")
            # total_mAP = ap_sum / sample_sum
            total_mAP = ap_sum / len(voc_names)
            total_mFPR_90 = fpr_90_sum / sample_sum
            total_mFPR_75 = fpr_75_sum / sample_sum
            total_mFPR_50 = fpr_50_sum / sample_sum
            content += str(total_mAP) + '\t{}\t{}\t{}'.format(total_mFPR_50, total_mFPR_75, total_mFPR_90)
            # with open('result.txt', 'a') as f:
            #     f.write(content)
            a.suptitle('thr:{}  total mAP:{:.4f}'.format(thr, total_mAP))
            # b.suptitle('The relationship between fpr and recall')

            mean_total_mAP += total_mAP
            print("overthresh:", thr, "total_mAP:", total_mAP)
        print("overthesh 0.1,0.3,0.5:", mean_total_mAP / 3.)
        plt.show()

    def gen_commit_result_round2_compare(self, pic_path,config2make_json, model2make_json):

        # build the model from a config file and a checkpoint file
        model = init_detector(config2make_json, model2make_json, device='cuda:0')

        img_list = []
        for img_name in os.listdir(pic_path):
            if img_name.endswith('.jpg'):
                img_list.append(img_name)

        for img_name in tqdm(img_list):
            model.cfg.test_cfg.rcnn.score_thr = 0.5

            gt_box = self.img_instance[img_name]
            full_img = os.path.join(pic_path, img_name)
            full_img = mmcv.imread(full_img)
            img_temp = copy.deepcopy(full_img)
            img_h, img_w = full_img.shape[:2]
            patches = [np.array((0, 0, img_w // 2, img_h // 2)), np.array((img_w // 2, 0, img_w, img_h // 2)),
                       np.array((0, img_h // 2, img_w // 2, img_h)),
                       np.array((img_w // 2, img_h // 2, img_w, img_h))]
            predicts = []
            for patch_idx, patch in enumerate(patches):
                patch_img = full_img[patch[1]:patch[3], patch[0]:patch[2]]
                predicts.append(inference_detector(model, patch_img))

            model.cfg.test_cfg.rcnn.score_thr = 0.05

            predicts.append(inference_detector(model, full_img))
            for i, (bboxes1, bboxes2, bboxes3, bboxes4, bboxes5) in enumerate(
                    zip(predicts[0], predicts[1], predicts[2], predicts[3], predicts[4])):

                bboxes1[:, :4] += np.tile(patches[0][:2], 2)
                bboxes2[:, :4] += np.tile(patches[1][:2], 2)
                bboxes3[:, :4] += np.tile(patches[2][:2], 2)
                bboxes4[:, :4] += np.tile(patches[3][:2], 2)
                merge_bboxes = np.concatenate([bboxes1, bboxes2, bboxes3, bboxes4, bboxes5], axis=0)
                # print("merged:",merge_bboxes)
                if merge_bboxes.size:
                    keep_inds = self.nms(merge_bboxes, thr=0.5)
                    merge_bboxes = merge_bboxes[keep_inds].reshape(-1, 5)
                    defect_label = i + 1

                    for bbox in merge_bboxes:
                        x1, y1, x2, y2, score = bbox.tolist()
                        cv2.rectangle(full_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                        cv2.putText(full_img, "{}|{}".format(defect_label, str(score)), (int(x1), int(y1 + 20)),
                                    cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 2)
                    for anno in gt_box:
                        bbox = anno.bbox
                        cv2.rectangle(full_img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 4)
            for i, (bboxes5) in enumerate(predicts[4]):
                defect_label = i + 1

                for bbox in bboxes5:
                    x1, y1, x2, y2, score = bbox.tolist()
                    cv2.rectangle(img_temp, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                    cv2.putText(img_temp, "{}|{}".format(defect_label, str(score)), (int(x1), int(y1 + 20)),
                                cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 2)

                for anno in gt_box:
                    bbox = anno.bbox
                    cv2.rectangle(img_temp, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 4)

            full_img = cv2.resize(full_img, (img_w // 3, img_h // 2))
            img_temp = cv2.resize(img_temp, (img_w // 3, img_h // 2))
            cv2.imshow("full", full_img)
            cv2.imshow("temp", img_temp)
            c = cv2.waitKey(0) & 0xFF
            if c == ord('q'):
                break

    def nms(self,dets, thr=0.5):
        # dets:[N,5]
        assert dets.shape[-1] % 5 == 0
        if dets.shape[0] == 1:
            return 0
        x1, y1, x2, y2, score = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3], dets[:, 4]
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
            ovr = inter / (areas[i] + areas[orders[1:]] - inter)
            inds = np.where(ovr <= thr)[0]
            orders = orders[inds + 1]
        return keep

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


class Transformer:
    def __init__(self):
        self.aug_img_seq = iaa.Sequential([
            iaa.Fliplr(0.8),
            iaa.Flipud(0.8),
            # iaa.Invert(0.5),
            iaa.Crop(percent=0.10)
        ], random_order=True)
        # pass

    def __call__(self, imgBGR, instance=None):
        imgRGB = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB)

        # imgRGB = self.aug_img_seq.augment_images(imgRGB)
        imgBGR_aug = cv2.cvtColor(imgRGB, cv2.COLOR_RGB2BGR)

        # save json format
        if instance is not None:

            img_info_tmp = edict()
            img_info_tmp.bbox = instance.bbox
            img_info_tmp.defect_name = instance.defect_name
            img_info_tmp.name = instance.name
            return imgBGR_aug, img_info_tmp
        else:
            return imgBGR_aug, None

    def aug_img(self, imgBGR, instance=None, img_info = None):
        bbs = self._mk_bbs(instance, img_info)
        imgRGB = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB)
        imgRGB_aug, bbs_aug = self.aug_img_seq(image = imgRGB, bounding_boxes = bbs)
        bbs_aug = bbs_aug.clip_out_of_image()
        imgBGR_aug = cv2.cvtColor(imgRGB_aug, cv2.COLOR_RGB2BGR)

        # for debug to show
        # imgRGB_aug_with_box = bbs_aug.draw_on_image(imgRGB_aug,size = 2)
        # imgRGB_aug_with_box = cv2.cvtColor(imgRGB_aug_with_box, cv2.COLOR_RGB2BGR)
        # imgRGB_aug_with_box = cv2.resize(imgRGB_aug_with_box,(1333,800))
        # imgRGB_with_box = bbs.draw_on_image(imgRGB, size=2)
        # imgRGB_with_box = cv2.resize(imgRGB_with_box,(1333,800))
        # imgRGB_with_box = cv2.cvtColor(imgRGB_with_box, cv2.COLOR_RGB2BGR)
        # cv2.imshow('aug',imgRGB_aug_with_box)
        # cv2.imshow('raw',imgRGB_with_box)
        # cv2.waitKey(0)

        # save json format
        if len(bbs_aug.bounding_boxes) != 0:
            instance_aug = []
            aug_abs_path = osp.join(img_info.aug_save_path, img_info.aug_name)
            # print(len(bbs_aug.bounding_boxes))
            for i in range(len(bbs_aug.bounding_boxes)):
                anno = edict()
                box = []
                # print(bbs_aug.bounding_boxes[i].x1>=0 and bbs_aug.bounding_boxes[i].x2>=0 and bbs_aug.bounding_boxes[i].y1>=0 and bbs_aug.bounding_boxes[i].y2>=0)
                # print(bbs_aug.bounding_boxes[i].x1,bbs_aug.bounding_boxes[i].x2, bbs_aug.bounding_boxes[i].y1,bbs_aug.bounding_boxes[i].y2)
                box.append(bbs_aug.bounding_boxes[i].x1)
                box.append(bbs_aug.bounding_boxes[i].y1)
                box.append(bbs_aug.bounding_boxes[i].x2)
                box.append(bbs_aug.bounding_boxes[i].y2)
                if self._check_box(box, img_info):
                    continue
                anno.bbox = box
                anno.defect_name = bbs_aug.bounding_boxes[i].label

                anno.name = img_info.aug_name
                anno.abs_path = aug_abs_path
                instance_aug.append(anno)
            cv2.imwrite(aug_abs_path, imgBGR_aug)
            return imgBGR_aug, instance_aug
        else:
            return imgBGR_aug, None

    def _mk_bbs(self, instance, img_info):
        BBox = [] #[ Bounding_box, Bounding_box,]
        w = img_info.img_w
        h = img_info.img_h
        for ins in instance:
            box = ins.bbox
            BBox.append(BoundingBox(x1 = box[0], y1 = box[1], x2 = box[2], y2 = box[3],label=ins.defect_name))

        return BoundingBoxesOnImage(BBox,shape = (h,w))

    def _check_box(self, box, img_info):
        '''
        img_info = edict({'img_h':h, "img_w":w, 'name':instance.name, 'aug_save_path':aug_save_path})
        :param box:
        :param img_info:
        :return:
        '''
        # img_w = img_info.img_w
        # img_h = img_info.img_h
        # if box[0] == 0 or box[1]== 0 or box[2] == img_w or box[3] == img_h:
        #     x, y = get_center(box)
        #     min_dis = min(abs(img_h - y), abs(img_w - x),x, y)
        #     if min_dis

        pass

    def aug_ins(self, ins, transcfg=None):
        if transcfg is None:
            transcfg = edict()
            transcfg.flipProb = 1
            transcfg.fx = [1, 1]
            transcfg.fy = [1, 1]
        fx = round(random.uniform(transcfg.fx[0], transcfg.fx[1]), 2)
        fy = round(random.uniform(transcfg.fy[0], transcfg.fy[1]), 2)

        imgBGR = ins.roi
        imgBGR_aug = cv2.resize(imgBGR, dsize=(0,0), fx=fx, fy=fy, interpolation=cv2.INTER_LINEAR)

        p = np.random.randint(0, 10, 1)
        if p > transcfg.flipProb:
            type_flip = random.choice([0, 1, -1])
            imgBGR_aug = cv2.flip(imgBGR_aug, type_flip)

        ins.roi = imgBGR_aug
        h, w, c = imgBGR_aug.shape
        ins.w = w
        ins.h = h

        return ins

def bbox_overlaps(bboxes1, bboxes2, mode='iou'):
    """Calculate the ious between each bbox of bboxes1 and bboxes2.

    Args:
        bboxes1(ndarray): shape (n, 4)
        bboxes2(ndarray): shape (k, 4)
        mode(str): iou (intersection over union) or iof (intersection
            over foreground)

    Returns:
        ious(ndarray): shape (n, k)
    """

    assert mode in ['iou', 'iof']

    bboxes1 = bboxes1.astype(np.float32)
    bboxes2 = bboxes2.astype(np.float32)
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    ious = np.zeros((rows, cols), dtype=np.float32)
    if rows * cols == 0:
        return ious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        ious = np.zeros((cols, rows), dtype=np.float32)
        exchange = True
    area1 = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (
        bboxes1[:, 3] - bboxes1[:, 1] + 1)
    area2 = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (
        bboxes2[:, 3] - bboxes2[:, 1] + 1)
    for i in range(bboxes1.shape[0]):
        x_start = np.maximum(bboxes1[i, 0], bboxes2[:, 0])
        y_start = np.maximum(bboxes1[i, 1], bboxes2[:, 1])
        x_end = np.minimum(bboxes1[i, 2], bboxes2[:, 2])
        y_end = np.minimum(bboxes1[i, 3], bboxes2[:, 3])
        overlap = np.maximum(x_end - x_start + 1, 0) * np.maximum(
            y_end - y_start + 1, 0)
        if mode == 'iou':
            union = area1[i] + area2 - overlap
        else:
            union = area1[i] if not exchange else area2
        ious[i, :] = overlap / union
    if exchange:
        ious = ious.T
    return ious

def compute_wh(box):
    x1, y1, x2, y2 = box
    x1 = max(x1, 0)
    y1 = max(y1, 0)
    x2 = max(x2, 0)
    y2 = max(y2, 0)
    w = x2 - x1
    h = y2 - y1
    return w, h

def esamble(result_json, result_json_template, save_path, flag_both = False):
    num_template = len(result_json_template)
    pic_template = edict()
    anno_temp = edict()
    for json_file in result_json_template:
        f = json.load(open(json_file,'r'))
        pic_temp = []
        for ins in f:
            pic_temp.append(ins['name'])
            if ins['name'] not in anno_temp.keys():
                anno_temp[ins['name']] = [ins]
            else:
                anno_temp[ins['name']] += ins
        pic_temp = set(pic_temp)
        for pic in pic_temp:
             if pic in pic_template.keys():
                 pic_template[pic] += 1
             else:
                 pic_template[pic] = 1
    print("acc高的模型预测出了%d张图："%(len(pic_temp)))
    f = json.load(open(result_json[0],'r'))
    f_temp = []
    pic_raw = []
    for ins in f:
        if ins['name'] in pic_template.keys() and pic_template[ins['name']] == num_template:
            f_temp.append(ins)
        pic_raw.append(ins['name'])
    pic_raw = set(pic_raw)
    print("map高的模型预测出了%d张图"%(len(pic_raw)))
    if flag_both:
        for im in anno_temp.keys():
            print(im)
            print(pic_raw)
            if im not in pic_raw:
                f_temp += anno_temp[im]
    with open(save_path, 'w') as f:
        json.dump(f_temp, f, indent=4, separators=(',', ': '), cls=MyEncoder)

def generate_gt_for_each_pic(im_path,save_path):
    re = []
    cls = [i for i in range(1,21)]
    for im in tqdm(os.listdir(im_path)):
        ins = edict()
        ins.name = im
        ins.category = random.choice(cls)
        x1 = round(random.random()*2446,2)
        y1 = round(random.random()*1000,2)
        x1 = min(x1,2200)
        y1 = min(y1,850)
        x2 = round(x1+100.02,2)
        y2 = round(y1+100.03,2)
        ins.bbox = [x1,y1,x2,y2]
        ins.score = random.random()
        re.append(ins)
    with open(save_path, 'w') as f:
        json.dump(re, f, indent=4, separators=(',', ': '), cls=MyEncoder)




def bbox_overlaps(bboxes1, bboxes2, mode='iou'):
    """Calculate the ious between each bbox of bboxes1 and bboxes2.

    Args:
        bboxes1(ndarray): shape (n, 4)
        bboxes2(ndarray): shape (k, 4)
        mode(str): iou (intersection over union) or iof (intersection
            over foreground)

    Returns:
        ious(ndarray): shape (n, k)
    """

    assert mode in ['iou', 'iof']

    bboxes1 = bboxes1.astype(np.float32)
    bboxes2 = bboxes2.astype(np.float32)
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    ious = np.zeros((rows, cols), dtype=np.float32)
    if rows * cols == 0:
        return ious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        ious = np.zeros((cols, rows), dtype=np.float32)
        exchange = True
    area1 = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (
        bboxes1[:, 3] - bboxes1[:, 1] + 1)
    area2 = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (
        bboxes2[:, 3] - bboxes2[:, 1] + 1)
    for i in range(bboxes1.shape[0]):
        x_start = np.maximum(bboxes1[i, 0], bboxes2[:, 0])
        y_start = np.maximum(bboxes1[i, 1], bboxes2[:, 1])
        x_end = np.minimum(bboxes1[i, 2], bboxes2[:, 2])
        y_end = np.minimum(bboxes1[i, 3], bboxes2[:, 3])
        overlap = np.maximum(x_end - x_start + 1, 0) * np.maximum(
            y_end - y_start + 1, 0)
        if mode == 'iou':
            union = area1[i] + area2 - overlap
        else:
            union = area1[i] if not exchange else area2
        ious[i, :] = overlap / union
    if exchange:
        ious = ious.T
    return ious