###1.修改测试方式
    flag = 0
        
    if flag == 0: # 线下不裁剪测试
        result_frompic_val()
    if flag == 1:# 线上不裁剪测试
        result_frompic_no_crop()
    if flag == 2: #线上裁剪测试
        gen_commit_result_round2(pic_path)
    if flag == 3: # 线下裁剪测试
        gen_commit_result_round2_down(pic_path)

###2.测试时修改crop和resize图尺寸
    #示例如下:
        for img_name in tqdm(img_list):
        # model.cfg.data.test.pipeline[1].img_scale = (1024, 425)
        # model.cfg.test_cfg.rcnn.score_thr = 0.8
        # model.cfg.data.test.pipeline[1].flip = False

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

        # model.cfg.data.test.pipeline[1].img_scale = (2048,850)
        # model.cfg.test_cfg.rcnn.score_thr = 0.05
        # model.cfg.data.test.pipeline[1].flip = True
        predicts.append(inference_detector(model, full_img))
        
    #号位置即修改的地方,若要让裁剪图和resize图测试尺寸不同,去掉上下两个model.cfg.data.test.pipeline[1].img_scale的注释
    并设置自定义大小.若要该score也是一样的操作.切记上下都要同步打开,因为crop图和resize图是交替前传的.