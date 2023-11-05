import cv2
import numpy as np
from model_api import model_init
from model_api import cnn_detect
import torch
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


if __name__ == '__main__':

    # 参数信息
    # detectInfoList = [{"bboxId": 0, "deviceName": "I_BLQB_2", "deviceNum": 0,
    #                    "permutation": "vertical", "x": 9, "y": 303, "w": 663, "h": 663}]
    # detectInfoList = [{"bboxId": 0, "deviceName": "I_BLQB_1", "deviceNum": 0,
    #                    "permutation": "vertical", "x": 125, "y": 303, "w": 333, "h": 333}]
    # detectInfoList = [{"bboxId": 0, "deviceName": "I_YLJDQB_0", "deviceNum": 0,
    #                    "permutation": "vertical", "x": 187, "y": 561, "w": 260, "h": 260}]
    # detectInfoList = [{"bboxId": 0, "deviceName": "I_YZTYB_0", "deviceNum": 0,
    #                    "permutation": "vertical", "x": 360, "y": 523, "w": 155, "h": 191}]
    # detectInfoList = [{"bboxId": 0, "deviceName": "I_YWenB_0", "deviceNum": 0,
    #                    "permutation": "vertical", "x": 301, "y": 505, "w": 80, "h": 60},
    #                   {"bboxId": 1, "deviceName": "I_YWenB_0", "deviceNum": 0,
    #                    "permutation": "vertical", "x": 397, "y": 505, "w": 80, "h": 60},
    #                   {"bboxId": 2, "deviceName": "I_YWenB_0", "deviceNum": 0,
    #                    "permutation": "vertical", "x": 299, "y": 655, "w": 80, "h": 60}
    #                   ]


    image = np.array(cv2.imread('test_image/I_YWenB_0.jpg'))

    # 1. 加载模型接口
    model = model_init('model.pkl',device)

    # 2. 检测接口
    detect_results = cnn_detect(model, image, detectInfoList,device)
    print(detect_results)


