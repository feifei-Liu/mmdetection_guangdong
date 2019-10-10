import numpy as np
from pycocotools.coco import COCO
from .coco import CocoDataset
from .registry import DATASETS
from .custom import CustomDataset

@DATASETS.register_module
class MyDataset(CocoDataset):

    CLASSES = ('1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
           '11', '12', '13', '14', '15', '16', '17', '18', '19', '20')

    def data2coco(self,anno_file):
        # anno_file : list [json_1, json_2....]
        import os
        import json
        import numpy as np
        import shutil
        import pandas as pd

        img_dir = "/home/zhangming/Models/Results/cloth_flaw_detection/Datasets/COCO_format/Train_images_no_crop"  # 图片路径
        save_dir = "/home/zhangming/Models/Results/cloth_flaw_detection/Datasets/COCO_format/Coco_json/" # 待保存的coco格式json路径
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
                # if not os.path.exists("coco/images/{}".format(self.mode)):
                #     os.makedirs("coco/images/{}".format(self.mode))

            def to_coco(self, anno_file, phase):
                self._init_categories()
                # for i in range(len(anno_file)):
                #     anno_f = anno_file[i]
                anno_result = pd.read_json(open(anno_file, "r"))
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

                print("一个%d张%s图片"%(self.img_id,phase))
                instance = {}
                instance['info'] = 'fabric defect'
                instance['license'] = ['none']
                instance['images'] = self.images
                instance['annotations'] = self.annotations
                instance['categories'] = self.categories

                return instance

            def _init_categories(self):
                for v in range(1, 21):
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

        '''转换有瑕疵的样本为coco格式'''
        # phase = anno_file.split('/')[-1].split('.')[0].split('_')[-1]
        phase = anno_file.split('/')[-1]
        # coco_path = save_dir + 'instances_{}.json'.format(phase)
        coco_path = save_dir + 'instances_'+phase
        if os.path.exists(coco_path):
            print("Json已存在 %s，直接加载"%coco_path)
            return coco_path
        else:
            fabric2coco = Fabric2COCO()
            instance= fabric2coco.to_coco(anno_file,phase)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            fabric2coco.save_coco_json(instance, coco_path)
            print("Json已创建 %s"%coco_path)
        return coco_path

    def load_annotations(self, ann_file):
        ann_file_coco = self.data2coco(ann_file)
        self.coco = COCO(ann_file_coco)
        self.cat_ids = self.coco.getCatIds()
        self.cat2label = {
            cat_id: i + 1
            for i, cat_id in enumerate(self.cat_ids)
        }
        self.img_ids = self.coco.getImgIds()
        img_infos = []
        for i in self.img_ids:
            info = self.coco.loadImgs([i])[0]
            info['filename'] = info['file_name']
            img_infos.append(info)
        return img_infos

    def get_ann_info(self, idx):
        img_id = self.img_infos[idx]['id']
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        ann_info = self.coco.loadAnns(ann_ids)
        return self._parse_ann_info(self.img_infos[idx], ann_info)

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())
        for i, img_info in enumerate(self.img_infos):
            if self.img_ids[i] not in ids_with_ann:
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
                labels, masks, mask_polys, poly_lens.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []

        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            bbox = [x1, y1, x1 + w - 1, y1 + h - 1]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
            gt_masks_ann.append(ann['segmentation'])

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map)

        return ann
