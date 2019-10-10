#encoding:utf/8
import os


image_path = "/home/remo/Desktop/cloth_flaw_detection/guangdong1_round1_train1_20190809/defect_Images/"
images = os.listdir(image_path)
trainval = open("/home/remo/Desktop/cloth_flaw_detection/guangdong1_round1_train1_20190809/xml.txt",'w')
test = open("/home/remo/Desktop/cloth_flaw_detection/guangdong1_round1_train1_20190809/Datasets/ImageSets/Main/test.txt",'w')

num = len(images)
trainval_num = int(0.7*num)
print('一共{}训练图片'.format(str(trainval_num)))
print('一共{}测试图片'.format(str(num-trainval_num)))

# i = 0
# for image in images:
#     image_name,ext = os.path.splitext(image)
#     if i < trainval_num:
#         if i == trainval_num-1:
#             trainval.write(image_name)
#         else:
#             trainval.write(image_name+'\n')
#     else:
#         if i == num-1:
#             test.write(image_name)
#         else:
#             test.write(image_name+'\n')
#     i += 1

i = 0
for image in images:
    image_name,ext = os.path.splitext(image)
    xml_name = image_name+'.xml'
    if i == num-1:
        trainval.write(xml_name)
    else:
        trainval.write(xml_name+'\n')
    i+=1

