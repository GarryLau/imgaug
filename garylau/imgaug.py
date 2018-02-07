# 导入所需package
import sys
sys.path.append('D:\\imgaug-master')
import imgaug as ia
from imgaug import augmenters as iaa
import cv2
import pandas as pd
import time
import xml.dom.minidom

# 数据增强示例
ia.seed(1)
father_path = 'D:\\garylau\\srd\\'
ImgBb = pd.read_csv('D:\\garylau\\srd\\train-GroundTruth.txt',header=None)
# 建立ImgBb中images与bounding boxes之间的key-value关系
kv_ImgBb = dict()
for i in range(ImgBb.shape[0]):
    str_ImgBb = ImgBb[0][i]
    list_ImgBb = str_ImgBb.split(" ")
    str_vImgBb = list_ImgBb[1]+" "+list_ImgBb[2]+ " "+ list_ImgBb[3]+ " "+list_ImgBb[4]
    kv_ImgBb[list_ImgBb[0]] = str_vImgBb
# 数据增强操作
# 每幅图片产生10个增强图像
for augnum in range(10):
    for key_ImgBb in kv_ImgBb:
        v_ImgBb = kv_ImgBb[key_ImgBb].split(' ')
        # xml路径
        xml_path = father_path + 'xml\\' + key_ImgBb.split('.')[0] + '.xml'
        xml_augname = key_ImgBb.split('.')[0] + '_' + str(augnum) + '.xml'
        img_augname = key_ImgBb.split('.')[0] + '_' + str(augnum) + '.jpg'
        aug_xml_path = father_path + 'imgaug_xml\\' + xml_augname
        aug_img_path = father_path + 'imgaug\\' + img_augname
        # 打开文件
        augxmlfile = open(aug_xml_path, 'w')
        # 解析xml
        dom = xml.dom.minidom.parse(xml_path)        #打开xml文档    
        root = dom.documentElement                   #得到xml文档对象
        # 标定框坐标
        xmin_nodelist = root.getElementsByTagName('xmin')
        xmin = xmin_nodelist[0].firstChild.data
        ymin_nodelist = root.getElementsByTagName('ymin')
        ymin = ymin_nodelist[0].firstChild.data
        xmax_nodelist = root.getElementsByTagName('xmax')
        xmax = xmax_nodelist[0].firstChild.data
        ymax_nodelist = root.getElementsByTagName('ymax')
        ymax = ymax_nodelist[0].firstChild.data
        # xml中filename、path
        xml_filename_nodelist = root.getElementsByTagName('filename')
        xml_filename_nodelist[0].firstChild.data = img_augname
        xml_path_nodelist = root.getElementsByTagName('path')
        xml_path_nodelist[0].firstChild.data = aug_img_path
        # 图片路径
        image_path = father_path + 'image\\' + key_ImgBb
        # 读取图片
        image = cv2.imread(image_path)
        # 读取groundtruth坐标
        x1 = int(xmin)
        y1 = int(ymin)
        x2 = int(xmax)
        y2 = int(ymax)
        # 变换Image、BoundingBox
        bbs = ia.BoundingBoxesOnImage([ia.BoundingBox(x1, y1, x2, y2)], shape=image.shape)
        seq = iaa.Sequential([iaa.Multiply((1.2, 1.8)),
                              iaa.Add((-40, 40), per_channel=0.5),
                             ])
        # Make our sequence deterministic. We can now apply it to the image and then to the BBs and it will lead to the same augmentations.
        # IMPORTANT: Call this once PER BATCH, otherwise you will always get the exactly same augmentations for every batch!
        seq_det = seq.to_deterministic()
        # Augment BBs and images. As we only have one image and list of BBs, we use [image] and [bbs] to turn both into lists (batches) for the
        # functions and then [0] to reverse that. In a real experiment, your variables would likely already be lists.
        image_aug = seq_det.augment_images([image])[0]
        bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]

        # print coordinates before/after augmentation (see below) use .x1_int, .y_int, ... to get integer coordinates
        for i in range(len(bbs.bounding_boxes)):
            before = bbs.bounding_boxes[i]
            after = bbs_aug.bounding_boxes[i]
            xmin_nodelist[0].firstChild.data = after.x1
            ymin_nodelist[0].firstChild.data = after.y1
            xmax_nodelist[0].firstChild.data = after.x2
            ymax_nodelist[0].firstChild.data = after.y2
            # 将xml的改变保存并写入
            dom.writexml(augxmlfile, addindent = ' ',encoding = 'utf-8' )
            cv2.imwrite(aug_img_path,image_aug)
            # 关闭文件    
            augxmlfile.close()
            '''
            print("%s BoundingBox %d: (%d, %d, %d, %d) -> (%.4f, %.4f, %.4f, %.4f)" %
                  (img_augname,
                   i,
                   before.x1, before.y1, before.x2, before.y2,
                   after.x1, after.y1, after.x2, after.y2))
            '''

    '''
        # image with BBs before/after augmentation (shown below)
        image_before = bbs.draw_on_image(image, thickness=2)
        image_after = bbs_aug.draw_on_image(image_aug, thickness=2, color=[0, 0, 255])

        image_forShow_image_before = cv2.resize(image_before,(int(image_before.shape[1] * 0.5), int(image_before.shape[0] * 0.5)))
        image_forShow_image_after = cv2.resize(image_after,(int(image_after.shape[1] * 0.5), int(image_after.shape[0] * 0.5)))
        cv2.imshow('image_forShow_image_before',image_forShow_image_before)
        cv2.imshow('image_forShow_image_after',image_forShow_image_after)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    '''
