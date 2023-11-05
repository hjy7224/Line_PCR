import torch.utils.data as data
import os
import torchvision.transforms as transforms
import cv2
import numpy as np
import xml.etree.ElementTree as ET
import random
import gc
import matplotlib.pyplot as plt

import torchvision

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
hm_stride = 1/4
train_size = 640

class pointer_dataset(data.Dataset):
    def __init__(self,srcImgDir,hmImgDir):
        self.srcImgDir = srcImgDir
        self.hmImgDir = hmImgDir
        #self.inputImgSize = inputImgSize
        #self.outputImgSize = outputImgSize
        #self.data_list = os.listdir(data_path)
        self.Normalize = transforms.Normalize(mean=mean,std = std)##
        self.imgs = []
        self.hms = []
        self.class_label=[]
        self.process()
    def gammatrans(self,img,gamma):
        return (np.power((img / float(np.max(img))), gamma) * 255).astype(np.uint8)

    def __getitem__(self, item):
        img = self.imgs[item]
        hm = self.hms[item]

        # sigma = np.random.normal(1,0.7)
        # if sigma<1: sigma = 1+(1-sigma)
        # flag = np.random.randint(0,2)
        # if flag :sigma = 1./sigma
        # img = self.gammatrans(img,sigma)
        # rot_angle = round(np.random.normal(0,2))
        # img = self.rotate_bound(img,rot_angle)
        # hm = self.rotate_bound(hm,rot_angle)
        # hm = cv2.resize(hm, (128, 128))
        # img = cv2.resize(img,(512,512))
        return img,hm,self.class_label[item]

    def __len__(self):
        assert len(self.imgs)==len(self.hms)
        return len(self.hms)

    def reloadimgs(self):
        self.imgs.clear()
        self.hms.clear()
        self.class_label.clear()
        gc.collect()
        self.process()

    #多条边界线
    def process(self):
        c = 0
        srcImgList = os.listdir(self.srcImgDir)
        #hmImgList = os.listdir(self.hmImgDir)
        for imgName in srcImgList:
            if imgName[-4:]!=".bmp" or imgName[-7:] == '_hm.bmp': continue

            imgPath = os.path.join(self.srcImgDir, imgName)
            #print(img_path)
            hm_1left_path = os.path.join(self.hmImgDir, imgName[:-4] + '_1left_hm.bmp')
            hm_1right_path = os.path.join(self.hmImgDir, imgName[:-4] + '_1right_hm.bmp')
            hm_2left_path = os.path.join(self.hmImgDir, imgName[:-4] + '_2left_hm.bmp')
            hm_2right_path = os.path.join(self.hmImgDir, imgName[:-4] + '_2right_hm.bmp')

            if not(os.path.exists(imgPath) and os.path.exists(hm_1left_path) and os.path.exists(hm_1right_path) and os.path.exists(hm_2left_path) and os.path.exists(hm_2right_path)):continue
            if random.randint(0,99)<0 : continue
            print(c)
            c += 1

            #print(hm_path)
            img = cv2.imread(imgPath, 0).astype(np.float32)/255.
            img = cv2.resize(img, (512, 512))
            hm_1left = cv2.imread(hm_1left_path, 0).astype(np.float32)/255.
            hm_1left = cv2.resize(hm_1left, (128, 128))
            hm_1right = cv2.imread(hm_1right_path, 0).astype(np.float32)/255.
            hm_1right = cv2.resize(hm_1right, (128, 128))
            hm_2left = cv2.imread(hm_2left_path, 0).astype(np.float32)/255.
            hm_2left = cv2.resize(hm_2left, (128, 128))
            hm_2right = cv2.imread(hm_2right_path, 0).astype(np.float32)/255.
            hm_2right = cv2.resize(hm_2right, (128, 128))

            #img = np.array([img]).transpose((1,2,0))
            inImgs = np.array([img]).transpose(1,2,0)
            #img = np.expand_dims(img,axis=2)
            inHms = np.array([hm_1left, hm_1right, hm_2left, hm_2right]).transpose((1, 2, 0))
            #hm = np.expand_dims(hm_fan, axis=-1)

            # print("数组元素总数：", hm.size)  # 打印数组尺寸，即数组元素总数
            # print("数组形状：", hm.shape)  # 打印数组形状
            # print("数组的维度数目", hm.ndim)'

            self.hms.append(inHms)
            self.imgs.append(inImgs)
            self.class_label.append(0)##class_label有什么用？这里注释掉？

            ############flip -1#######################
            img_flip = cv2.flip(img, -1)
            hm_1left_flip = cv2.flip(hm_1left, -1)
            hm_1right_flip = cv2.flip(hm_1right, -1)
            hm_2left_flip = cv2.flip(hm_2left, -1)
            hm_2right_flip = cv2.flip(hm_2right, -1)
            # cv2.imshow("hm_l0l",hm_l0l)
            # cv2.imshow("hm_l0l_flip",hm_l0l_flip)
            # cv2.waitKey(0)

            # img = np.array([img]).transpose((1,2,0))
            inImgs = np.array([img_flip]).transpose(1, 2, 0)
            # img = np.expand_dims(img,axis=2)
            inHms = np.array([hm_1left_flip, hm_1right_flip, hm_2left_flip, hm_2right_flip]).transpose((1, 2, 0))
            # hm = np.expand_dims(hm_fan, axis=-1)

            self.hms.append(inHms)
            self.imgs.append(inImgs)
            self.class_label.append(0)  ##class_label有什么用？这里注释掉？

    # #单条边界线
    # def process(self):
    #     c = 0
    #     for imgName in self.data_list:
    #         self.class_label.append(0)##class_label有什么用？这里注释掉？
    #         if imgName[-4:]!=".bmp" or imgName[-7:] == '_hm.bmp': continue
    #         print(c)
    #         c+=1
    #         imgPath = os.path.join(self.data_path, imgName)
    #         #print(img_path)
    #         hmImgPath = os.path.join(self.data_path, imgName[:-4] + '_fan_hm.bmp')
    #         #print(hm_path)
    #         img = cv2.imread(imgPath, 1).astype(np.float32)/255.
    #         hm = cv2.imread(hmImgPath, 0).astype(np.float32)/255.
    #
    #         #我不需要resize,大小都固定的
    #         #fx = train_size / img.shape[1]
    #         #fy = train_size / img.shape[0]
    #         #img = cv2.resize(img,(0,0),fx = fx,fy = fy)
    #         #img = cv2.resize(img, (train_size, train_size))
    #         #keypoints,pointers = self.get_xml_data(label_path,fx,fy)
    #         #hm = self.creat_heatmap(img.shape[0],img.shape[1],keypoints,pointers, 10)
    #
    #         hm = np.expand_dims(hm, axis=-1)
    #         self.hms.append(hm)
    #         self.imgs.append(img)

    # def process(self):
    #     c = 0
    #     for xml in self.data_list:
    #         self.class_label.append(int(xml[0])-1)##class_label有什么用？
    #         if xml[-3:]!='xml':continue
    #         print(c)
    #         c+=1
    #         img_path = os.path.join(self.data_path, xml[:-3] + 'jpg')
    #         label_path = os.path.join(self.data_path, xml)
    #         img = cv2.imread(img_path, 1).astype(np.float32)/255.
    #         fx = train_size / img.shape[1]
    #         fy = train_size / img.shape[0]
    #         #img = cv2.resize(img,(0,0),fx = fx,fy = fy)
    #         img = cv2.resize(img, (train_size, train_size))
    #         keypoints,pointers = self.get_xml_data(label_path,fx,fy)
    #         hm = self.creat_heatmap(img.shape[0],img.shape[1],keypoints,pointers, 10)
    #         self.hms.append(hm)
    #         self.imgs.append(img)


    def get_xml_data(self,label_path,fx,fy):
        obj = ET.parse(label_path)

        keypoint_str = obj.find('keypoints').text
        keypoints = []
        pointers = []
        if keypoint_str != '[]':
            keypoint_str = str.split(keypoint_str[2:-2],'), (')
            keypoints = np.array([np.array(str.split(point,','),dtype=np.float32) for point in keypoint_str])
        pointers_str = obj.findall('pointer')
        for pointer_str in pointers_str:
            pointer = str.split(pointer_str.text[2:-2],'), (')
            pointer = np.array([np.array(str.split(point,','),dtype=np.float32) for point in pointer])
            pointers.append(pointer)
        pointers = np.array(pointers)
        if len(keypoints):
            keypoints[:, 0] *= fx
            keypoints[:, 1] *= fy
        pointers[:, :, 0] *= fx
        pointers[:, :, 1] *= fy
        return keypoints,pointers

    def guassian_kernel(self,size_w, size_h, center_x, center_y, sigma):

        gridy, gridx = np.mgrid[0:size_h, 0:size_w]
        D2 = (gridx - center_x) ** 2 + (gridy - center_y) ** 2
        return np.exp(-D2 / 2.0 / sigma / sigma)

    def creat_heatmap(self,h,w,keypoints,pointers, sigma):
        heatmap = []
        start = np.zeros((h,w),dtype=np.float32)
        end = np.zeros((w,h),dtype=np.float32)
        if len(keypoints):
            for point in keypoints:
                heatmap.append(self.guassian_kernel(h, w, point[0], point[1], sigma))
        else:
            for i in range(4):
                heatmap.append(np.zeros((512,512),dtype=np.float32))
        for i in range(pointers.shape[0]):
            start_tmp = self.guassian_kernel(w, h, pointers[i ,0, 0], pointers[i, 0, 1], sigma)
            start = np.where(start > start_tmp, start, start_tmp)
            end_tmp = self.guassian_kernel(w, h, pointers[i, 1, 0], pointers[i, 1, 1], sigma)
            end = np.where(end > end_tmp, end, end_tmp)
        heatmap.append(start)
        heatmap.append(end)
        return np.array(heatmap).transpose((1,2,0))



    def rotate_bound(self, image, angle):
        (h, w) = image.shape[:2]
        (cX, cY) = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY
        img = cv2.warpAffine(image, M, (nW, nH))
        tl = (img.shape[0] // 2 - train_size//2, img.shape[1] // 2 - train_size//2)
        img = img[tl[0]:tl[0] + train_size, tl[1]:tl[1] + train_size]
        return img



