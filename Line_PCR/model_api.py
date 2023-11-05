import torch
from PostProcess import data_transform,meter1_reading,meter2_reading,meter3_reading,meter4_reading,meter5_reading,meter6_reading
import numpy as np
from network import Net
def model_init(checkpoint_file,device):
    """
    :param checkpoint_file: 模型文件
    :return: 返回初始化的模型
    """
    model = Net().to(device)
    model.load_state_dict(torch.load(checkpoint_file))
    return model
def cnn_detect(model, image, detectInfo, device):
    """
    :param model: 初始化的模型
    :param image: 传入的图像信息
    :param detectInfo: 检测参数信息
    :return: 返回检测结果
    """
    code = 0
    message = []
    results_all = []
    for id,info in enumerate(detectInfo):
        y = info['y']
        x = info['x']
        w = info['w']
        h = info['h']
        meterName = info['deviceName']
        if y+h>image.shape[0] or x+w>image.shape[1]:
            results = {
                'bboxID': id,
                'deviceName': meterName,
                'parameter': None
            }
            code = 1
            message.append('BBoxId:'+str(id)+'-cross border')

        else:
            roi = image[y:y+h,x:x+w]
            org_roi = np.array(roi)
            fy, fx, tensor_data = data_transform(roi, device)
            hm = model(tensor_data)
            if meterName == 'I_BLQB_1':
                org_roi, reading = meter4_reading(org_roi, hm, fy, fx)
                kedu = ['count','mA']
            elif meterName == 'I_BLQB_2':
                org_roi, reading = meter2_reading(org_roi, hm, fy, fx)
                kedu = ['mA']
            elif meterName == 'I_YLJDQB_0':
                org_roi, reading = meter3_reading(org_roi, hm, fy, fx)
                kedu = ['flow']
            elif meterName == 'I_XXDYB_0':
                org_roi, reading = meter1_reading(org_roi, hm, fy, fx)
                kedu = ['kv']
            elif meterName == 'I_YWenB_0':
                org_roi, reading = meter6_reading(org_roi, hm, fy, fx)
                kedu = ['YWen1','YWen2']
            elif meterName == 'I_YZTYB_0':
                org_roi, reading = meter5_reading(org_roi, hm, fy, fx)
                kedu  = ['value1','value2','value3']
            if reading==None:
                code = 1
                message.append('BBoxId:' + str(id) + '-Detection failed')
                parameter = None
            else:
                message.append('BBoxId:' + str(id) + '-Success')
                parameter =dict(zip(kedu,reading))
            results = {
                    'bboxID':id,
                    'deviceName':meterName,
                    'parameter': parameter
            }
        results_all.append(results)
    out = {
        'code':code,
        'message':message,
        'results':results_all
    }
    return out
