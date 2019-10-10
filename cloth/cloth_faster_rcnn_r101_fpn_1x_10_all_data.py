# coding=utf-8
# model settings
model = dict(
    type='FasterRCNN',
    pretrained=None,                # 可以直接使用 load_from
    backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,               # resnet的stage数量
        out_indices=(0, 1, 2, 3),   # 输出的stage的序号
        frozen_stages=1,            # 冻结的stage数量，即该stage不更新参数，-1表示所有的stage都更新参数
        style='pytorch'),           # 网络风格：如果设置pytorch，则stride为2的层是conv3x3的卷积层；如果设置caffe，则stride为2的层是第一个conv1x1的卷积层
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],     # 输入的各个stage的通道数
        out_channels=256,                       # 输出的特征层的通道数
        num_outs=5),                            # 输出的特征层的数量
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,                      # 特征层的通道数
        anchor_scales=[8],                      # 生成的anchor的baselen，baselen = sqrt(w*h)，w和h为anchor的宽和高
        anchor_ratios=[0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 10.0],        # anchor的宽高比
        anchor_strides=[4, 8, 16, 32, 64],                          # 在每个特征层上的anchor的步长
        target_means=[.0, .0, .0, .0],
        target_stds=[1.0, 1.0, 1.0, 1.0],
        use_sigmoid_cls=True),                     # 是否使用sigmoid来进行分类，如果False则使用softmax来分类
    bbox_roi_extractor=dict(
        type='SingleRoIExtractor',                 # RoIExtractor类型
        roi_layer=dict(type='RoIAlign',
                       out_size=7, sample_num=2),  # ROI具体参数：ROI类型为ROIalign，输出尺寸为7，sample数为2
        out_channels=256,                          # 输出通道数
        featmap_strides=[4, 8, 16, 32]),           # 特征图的步长
    bbox_head=dict(
        type='SharedFCBBoxHead',                   # 全连接层类型
        num_fcs=2,
        in_channels=256,
        fc_out_channels=1024,
        roi_feat_size=7,                           # ROI特征层尺寸
        num_classes=21,                            # 分类器的类别数量+1，+1是因为多了一个背景的类别
        target_means=[0., 0., 0., 0.],
        target_stds=[0.1, 0.1, 0.2, 0.2],
        reg_class_agnostic=False))
# model training and testing settings
train_cfg = dict(
    rpn=dict(
        assigner=dict(
            type='MaxIoUAssigner',                 # RPN网络的正负样本划分
            pos_iou_thr=0.7,
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',                  # 正负样本提取器类型
            num=256,                               # 需提取的正负样本数量
            pos_fraction=0.5,                      # 正样本比例
            neg_pos_ub=-1,
            add_gt_as_proposals=False),            # 把ground truth加入proposal作为正样本
        allowed_border=0,                          # 允许在bbox周围外扩一定的像素
        pos_weight=-1,
        smoothl1_beta=1 / 9.0,                     # 平滑L1系数
        debug=False),                              # debug模式
    rcnn=dict(
        assigner=dict(
            type='MaxIoUAssigner',                 # RCNN网络正负样本划分
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0.5,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=512,                               # 需提取的正负样本数量
            pos_fraction=0.25,                     # 正样本比例
            neg_pos_ub=-1,
            add_gt_as_proposals=True),
        pos_weight=-1,
        debug=False))
test_cfg = dict(
    rpn=dict(
        nms_across_levels=False,                   # 在所有的fpn层内做nms
        nms_pre=2000,                              # 在nms之前保留的的得分最高的proposal数量
        nms_post=2000,                             # 在nms之后保留的的得分最高的proposal数量
        max_num=2000,
        nms_thr=0.7,                               # nms阈值
        min_bbox_size=0),                          # 最小bbox尺寸
    rcnn=dict(
        score_thr=0.05, nms=dict(type='nms', iou_thr=0.5), max_per_img=100)   # max_per_img表示最终输出的det bbox数量
    # soft-nms is also supported for rcnn testing
    # e.g., nms=dict(type='soft_nms', iou_thr=0.5, min_score=0.05)
)
# dataset settings
dataset_type = 'MyDataset'
data_root = '/home/zhangming/Models/Results/cloth_flaw_detection/Datasets/COCO_format/'
img_norm_cfg = dict(
    mean=[90.5, 89.3, 96.3], std=[25.6637689, 26.93750831, 26.83956949], to_rgb=True)    # 输入图像初始化，减去均值mean并处以方差std，to_rgb表示将bgr转为rgb
data = dict(
    imgs_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'instances_train.json',
        img_prefix=data_root + 'Images/',
        img_scale=(1333, 800),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,     # 对图像进行resize时的最小单位，32表示所有的图像都会被resize成32的倍数
        flip_ratio=0.5,
        with_mask=False,
        with_crowd=True,     # 训练时附带difficult的样本
        with_label=True),    # 训练时附带label
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'instances_train.json',
        img_prefix=data_root + 'Images/',
        img_scale=(1333, 800),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0,
        with_mask=False,
        with_crowd=True,
        with_label=True),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'instances_train.json',
        img_prefix=data_root + 'Images/',
        img_scale=(1333, 800),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0,
        with_mask=False,
        with_label=False,
        test_mode=True))
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))   # 梯度均衡参数
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[8, 11])
checkpoint_config = dict(interval=1)   # 每1个epoch存储一次模型
# yapf:disable
log_config = dict(
    interval=50,                       # 每50个batch输出一次信息
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 12
dist_params = dict(backend='nccl')      # 分布式参数
log_level = 'INFO'
work_dir = './work_dirs/faster_rcnn_r101_fpn_1x_10_all_data'
load_from = './checkpoints/faster_rcnn_r101_fpn_1x_20181129-d1468807_21.pth'    # 加载模型的路径，None表示从预训练模型加载
resume_from = None                      # 恢复训练模型的路径
workflow = [('train', 1)]               # 当前工作区名称
