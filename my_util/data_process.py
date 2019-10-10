'''
数据准备：
1.将下载的数据分成训练集验证集，生成对应json文件
    $ data2coco()
2.根据json文件将训练数据和验证数据分开生成train.txt val.txt
    $ devide_train_val()
3.根据train.txt 和 val.txt 分开数据
    $ move_data2_trainval_dir()

mAP计算：
1.通过vis_result.py脚本生成验证集的提交json文件
2.通过generate_det.py脚本将结果json转化为计算map所需格式
3.通过Object-Detection-Metrics-master/samples/sample_2/sample_2.py脚本计算mAP
'''

def devide_train_val():
    import os
    import json

    root = ""
    phase = "train" # or val
    json_file = root+'instances_balance_%s_coco.json' % phase

    f1 = open(root+'%s_balance.txt' % phase,'w')
    f = json.load(open(json_file,'r'))
    ed = []
    for anno in f['images']:
        name = anno['file_name']
        if name not in ed:
            f1.write(name+'\n')
            ed.append(name)

def data2coco():
    import os
    import json
    import numpy as np
    import shutil
    import pandas as pd

    defect_name2label = {
        '破洞': 1, '水渍': 2, '油渍': 2, '污渍': 2, '三丝': 3, '结头': 4, '花板跳': 5, '百脚': 6, '毛粒': 7,
        '粗经': 8, '松经': 9, '断经': 10, '吊经': 11, '粗维': 12, '纬缩': 13, '浆斑': 14, '整经结': 15, '星跳': 16, '跳花': 16,
        '断氨纶': 17, '稀密档': 18, '浪纹档': 18, '色差档': 18, '磨痕': 19, '轧痕': 19, '修痕': 19, '烧毛痕': 19, '死皱': 20, '云织': 20,
        '双纬': 20, '双经': 20, '跳纱': 20, '筘路': 20, '纬纱不良': 20,
    }

    class Fabric2COCO:

        def __init__(self, mode="train"):
            self.images = []
            self.annotations = []
            self.categories = []
            self.img_id = 0
            self.ann_id = 0
            self.mode = mode
            if not os.path.exists("coco/images/{}".format(self.mode)):
                os.makedirs("coco/images/{}".format(self.mode))

        def to_coco(self, anno_file, img_dir):
            self._init_categories()
            for i in range(len(anno_file)):
                anno_f = anno_file[i]
                anno_result = pd.read_json(open(anno_f, "r"))
                name_list = anno_result["name"].unique()
                for img_name in name_list:
                    img_anno = anno_result[anno_result["name"] == img_name]
                    bboxs = img_anno["bbox"].tolist()
                    defect_names = img_anno["defect_name"].tolist()
                    assert img_anno["name"].unique()[0] == img_name

                    img_path = os.path.join(img_dir, img_name)
                    # img =cv2.imread(img_path)
                    # h,w,c=img.shape
                    h, w = 1000, 2446
                    self.images.append(self._image(img_path, h, w))

                    # self._cp_img(img_path)

                    for bbox, defect_name in zip(bboxs, defect_names):
                        label = defect_name2label[defect_name]
                        annotation = self._annotation(label, bbox)
                        self.annotations.append(annotation)
                        self.ann_id += 1
                    self.img_id += 1

            train_num = int(self.img_id / 5 * 4)
            print("There are %d train data" % train_num)
            print("There are %d val data" % (self.img_id - train_num))
            train_instance = {}
            train_instance['info'] = 'fabric defect'
            train_instance['license'] = ['none']
            train_instance['images'] = self.images[:train_num]
            train_instance['annotations'] = self.annotations[:train_num]
            train_instance['categories'] = self.categories[:train_num]

            val_instance = {}
            val_instance['info'] = 'fabric defect'
            val_instance['license'] = ['none']
            val_instance['images'] = self.images[train_num:]
            val_instance['annotations'] = self.annotations[train_num:]
            val_instance['categories'] = self.categories[train_num:]

            return train_instance, val_instance

        def _init_categories(self):
            for v in range(1, 21):
                print(v)
                category = {}
                category['id'] = v
                category['name'] = str(v)
                category['supercategory'] = 'defect_name'
                self.categories.append(category)
            # for k, v in defect_name2label.items():
            #     category = {}
            #     category['id'] = v
            #     category['name'] = k
            #     category['supercategory'] = 'defect_name'
            #     self.categories.append(category)

        def _image(self, path, h, w):
            image = {}
            image['height'] = h
            image['width'] = w
            image['id'] = self.img_id
            image['file_name'] = os.path.basename(path)
            return image

        def _annotation(self, label, bbox):
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            points = [[bbox[0], bbox[1]], [bbox[2], bbox[1]], [bbox[2], bbox[3]], [bbox[0], bbox[3]]]
            annotation = {}
            annotation['id'] = self.ann_id
            annotation['image_id'] = self.img_id
            annotation['category_id'] = label
            annotation['segmentation'] = [np.asarray(points).flatten().tolist()]
            annotation['bbox'] = self._get_box(points)
            annotation['iscrowd'] = 0
            annotation['area'] = area
            return annotation

        def _cp_img(self, img_path):
            shutil.copy(img_path, os.path.join("coco/images/{}".format(self.mode), os.path.basename(img_path)))

        def _get_box(self, points):
            min_x = min_y = np.inf
            max_x = max_y = 0
            for x, y in points:
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x)
                max_y = max(max_y, y)
            '''coco,[x,y,w,h]'''
            return [min_x, min_y, max_x - min_x, max_y - min_y]

        def save_coco_json(self, instance, save_path):
            import json
            with open(save_path, 'w') as fp:
                json.dump(instance, fp, indent=1, separators=(',', ': '))

    ##################################################  修改   ##################################################
    '''转换有瑕疵的样本为coco格式'''
    img_dir = "/home/remo/Desktop/cloth_flaw_detection/guangdong1_round1_train2_20190828/defect_Images"
    anno_dir = ["/home/remo/Desktop/cloth_flaw_detection/guangdong1_round1_train1_20190818/Annotations/anno_train.json",
                '/home/remo/Desktop/cloth_flaw_detection/guangdong1_round1_train2_20190828/Annotations/anno_train.json',
                '/home/remo/Desktop/cloth_flaw_detection/Dataset/COCO_format/Aug_json.json'
                ]
    save_dir = "/home/remo/Desktop/cloth_flaw_detection/Dataset/COCO_format/"
    ############################################################################################################
    fabric2coco = Fabric2COCO()
    train_instance, val_instance = fabric2coco.to_coco(anno_dir, img_dir)
    # if not os.path.exists("coco/annotations/"):
    #     os.makedirs("coco/annotations/")
    fabric2coco.save_coco_json(train_instance, save_dir + 'instances_balance_{}.json'.format("train"))
    fabric2coco.save_coco_json(val_instance, save_dir + 'instances_balance_{}.json'.format("val"))

def move_data2_trainval_dir():
    import shutil

    phase = "train"  # or val
    root = ""
    Image_root = ""
    Dt_image_root = ""
    file = root+'%s.txt' % phase

    f = open(file, 'r')
    for line in f:
        line = line.strip()
        shutil.copy(Image_root + line, Dt_image_root + line)

def modify_pretrain_class():
    # for cascade rcnn
    import torch
    num_classes = 21
    model_coco = torch.load("cascade_rcnn_x101_32x4d_fpn_2x_20181218-28f73c4c.pth")

    # weight
    model_coco["state_dict"]["bbox_head.0.fc_cls.weight"].resize_(num_classes, 1024)
    model_coco["state_dict"]["bbox_head.1.fc_cls.weight"].resize_(num_classes, 1024)
    model_coco["state_dict"]["bbox_head.2.fc_cls.weight"].resize_(num_classes, 1024)
    # bias
    model_coco["state_dict"]["bbox_head.0.fc_cls.bias"].resize_(num_classes)
    model_coco["state_dict"]["bbox_head.1.fc_cls.bias"].resize_(num_classes)
    model_coco["state_dict"]["bbox_head.2.fc_cls.bias"].resize_(num_classes)
    # save new model
    torch.save(model_coco, "coco_pretrained_weights_classes_%d.pth" % num_classes)

if __name__ == "__main__":
    data2coco()
    devide_train_val()
    move_data2_trainval_dir()