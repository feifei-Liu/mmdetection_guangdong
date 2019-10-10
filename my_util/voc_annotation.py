import xml.etree.ElementTree as ET
from os import getcwd

sets=[('2007', 'train'), ('2007', 'val'), ('2007', 'test')]

# classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
classes = [i for i in range(1,21)]


def convert_annotation(image_id, list_file):
    in_file = open('/home/remo/Desktop/cloth_flaw_detection/guangdong1_round1_train1_20190809/Datasets/Annotations/%s.xml'%(image_id))
    tree=ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        # if cls not in classes or int(difficult)==1:
        #     continue
        cls_id = int(cls)
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        # print(b)
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))

wd = getcwd()

for year, image_set in sets:
    image_ids = open("/home/remo/Desktop/cloth_flaw_detection/guangdong1_round1_train1_20190809/Datasets/ImageSets/Main/trainval.txt").read().strip().split()
    list_file = open('/home/remo/Desktop/cloth_flaw_detection/guangdong1_round1_train1_20190809/anno.txt', 'w')
    for image_id in image_ids:
        list_file.write('/home/remo/Desktop/cloth_flaw_detection/guangdong1_round1_train1_20190809/Datasets/JPEGImages/%s.jpg'%(image_id))
        convert_annotation(image_id, list_file)
        list_file.write('\n')
    list_file.close()

