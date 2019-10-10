from vis_gt import DataAnalyze
from easydict import EasyDict as edict
class Config:
    def __init__(self):
        # self.json_paths = [
        #         # "/home/xjx/data/Kaggle/Json/Aug_json.json",
        #         '/home/xjx/data/Kaggle/guangdong1_round1_train1_20190818/Annotations/anno_train.json',
        #         '/home/xjx/data/Kaggle/guangdong1_round1_train2_20190828/Annotations/anno_train.json',
        #         '/home/xjx/ding/Aug_json.json'
        #                    ] # 训练json
        # ------------------------train json, path -----------------------------
        self.json_paths = [
                # "/home/xjx/data/Kaggle/Json/Aug_json.json",
                # '/home/xjx/data/Kaggle/rawAddAug_train.json',
                '/home/zhangming/Models/Results/cloth_flaw_detection/Datasets/raw_Annotations/guangdong1/anno_train.json',
                '/home/zhangming/Models/Results/cloth_flaw_detection/Datasets/raw_Annotations/guangdong2/anno_train.json',
                # '/home/remo/Desktop/cloth_flaw_detection/Dataset/COCO_format/layout/rawAddAug_no_crop_train.json',
                # '/home/remo/Desktop/cloth_flaw_detection/Dataset/COCO_format/layout/rawAddAug_no_crop_val.json',
                # '/home/remo/test/Aug_json.json',
                # '/home/remo/Desktop/cloth_flaw_detection/Dataset/COCO_format/rawAddAug_train.json',
                # '/home/remo/Desktop/cloth_flaw_detection/Dataset/COCO_format/rawAddAug_val.json',
                # '/home/remo/test/Aug_json.json'
        ] # 训练json

        # self.allimg_path = '/home/remo/test' # 训练 图片地址
        self.allimg_path = '/home/zhangming/Models/Results/cloth_flaw_detection/Datasets/COCO_format/Train_images_no_crop' # 训练 图片地址
        # self.allimg_path = '/home/remo/Desktop/cloth_flaw_detection/Dataset/COCO_format/defect_images' # 训练 图片地址

        # ----------------------val json, path, result---------------------------
        # val result gt vis
        self.val_json_paths = ['/home/remo/Desktop/cloth_flaw_detection/Dataset/COCO_format/layout/rawAddAug_no_crop_val.json'] # val json coco格式
        self.val_img_path = '/home/remo/Desktop/cloth_flaw_detection/Dataset/COCO_format/Val_images'    # val 图片地址

        self.result_json = ['/home/remo/Desktop/cloth_flaw_detection/Results/result_28_val_30epoch.json'] # 结果 json 地址

        # ---------------------divide json --------------------------------------
        self.divide_json = [
                        '/home/zhangming/Models/Results/cloth_flaw_detection/Datasets/raw_Annotations/guangdong1/anno_train.json',
                        '/home/zhangming/Models/Results/cloth_flaw_detection/Datasets/raw_Annotations/guangdong2/anno_train.json',
                        # '/home/zhangming/Models/Results/cloth_flaw_detection/Datasets/COCO_format/layout/norm_images_aug.json',
                        # '/home/zhangming/Models/Results/cloth_flaw_detection/Datasets/COCO_format/layout/norm_images_twoperimg_aug.json',
                        '/home/zhangming/Models/Results/cloth_flaw_detection/Datasets/COCO_format/layout/Aug_no_crop.json',
                            ]

        # ---------------------aug path, json -----------------------------------
        self.aug_save_path = '/home/remo/Desktop/cloth_flaw_detection/Dataset/COCO_format/Aug_Images_crop'
        # self.json_file_path = '/home/remo/Desktop/cloth_flaw_detection/Dataset/COCO_format/Aug_json.json'
        self.json_file_path = '/home/remo/Desktop/cloth_flaw_detection/Dataset/COCO_format/defect_images/Aug_json_crop.json'
        self.add_num = 500



if __name__ == "__main__":

    cfg = Config()
    dataer = DataAnalyze(cfg)
    flag_tool = 2

    if flag_tool == 0:
        # 可视化gt
        dataer.vis_gt(flag_show_raw_img=False, test_img='7e7e6359d8fb8e9c1332319214_aug7.jpg') # 使用json_paths 和  allimg_path 路径

    elif flag_tool == 1:
        # 可视化 gt 和结果分析
        dataer.val_analyze(flag_coco_json=True) # 使用val_json_paths 和 val_img_path 作为val的路径，   result_json  作为结果的json

    elif flag_tool == 2:
        # 划分训练验证    # 使用 divide_json
        dataer.divide_trainval(ratio=0.0, train_json='/home/zhangming/Models/Results/cloth_flaw_detection/Datasets/COCO_format/rawAddAug_no_crop_train_all.json', val_json='/home/zhangming/Models/Results/cloth_flaw_detection/Datasets/COCO_format/rawAddAug_no_crop_val_all.json')

    elif flag_tool == 3:
        # 绘制类别曲线
        dataer.add_aug_data(add_num = cfg.add_num, aug_save_path=cfg.aug_save_path,
                            json_file_path= cfg.json_file_path)
        # dataer.draw_cls()

    elif flag_tool == 4:
        ins = dataer.img_instance
        ins_k = list(ins.keys())
        name_l = []
        gt_l = []
        for k in ins_k:
            gt_num = len(ins[k])
            ins
            # img_w = img_info.img_w
            # img_h = img_info.img_h
            # if box[0] == 0 or box[1]== 0 or box[2] == img_w or box[3] == img_h:
            #     x, y = get_center(box)
            #     min_dis = min(abs(img_h - y), abs(img_w - x),x, y)
            #     if min_dis
            gt_l.append(gt_num)
            name_l.append(k)

        gt_l, name_l = zip(*sorted(zip(gt_l, name_l), reverse=True))  # 相同规则排序


    elif flag_tool == 7:
        normal_imgs = ['/home/zhangming/Models/Results/cloth_flaw_detection/Datasets/COCO_format/normal_images_selected', ]
        img_save = '/home/zhangming/Models/Results/cloth_flaw_detection/Datasets/COCO_format/norm_images_twoperimg_aug'
        json_p = '/home/zhangming/Models/Results/cloth_flaw_detection/Datasets/COCO_format/layout/norm_images_twoperimg_aug.json'

        condition = edict()
        condition.pickgt = edict()
        condition.pickgt.img_scale = 0.5 #图片占原图尺寸
        condition.pickgt.min_num_per_image = 1 # 每张图片gt个数最小值
        condition.pickgt.max_num_per_image = 2 # 每张图片gt个数最大值
        condition.changebg = None

        condition.add_info = edict()
        condition.add_info.pick = 0.3  # 原图背景
        condition.add_info.ins = 0.7  # gt

        transcfg = edict()
        transcfg.fx = [1, 2]
        transcfg.fy = [1, 2]
        transcfg.flipProb = 1
        dataer.pick_gt(normal_Images=normal_imgs, img_save=img_save, json_file_path=json_p, condition=condition,
                       transcfg=transcfg)

