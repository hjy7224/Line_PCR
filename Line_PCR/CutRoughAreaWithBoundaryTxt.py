import os
import numpy as np
import cv2

image_path = r'F:\modle_data\split\image\6thFactory#23-20200528\6_big'
left_txt_path = r'F:\modle_data\split\image\6thFactory#23-20200528\line\1'
right_txt_path = r'F:\modle_data\split\image\6thFactory#23-20200528\line\4'
output_path = r'F:\modle_data\split\image\6thFactory#23-20200528\6'

image_postfix = '.bmp'
offset = 30

def loadBoundary(txt_path):
    if not os.path.exists(txt_path):
        return []
    with open(txt_path,'r') as file:
        lines = file.readlines()
    lines = [int(line.rstrip('\n').split(' ')[1]) for line in lines]
    return lines

if __name__ == '__main__':
    if not os.path.exists(image_path) or not os.path.exists(left_txt_path) or not os.path.exists(right_txt_path):
        exit(-1)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    image_names = os.listdir(image_path)
    image_names = [os.path.splitext(image_name)[0] for image_name in image_names if os.path.splitext(image_name)[1] == image_postfix]
    left_txt_names = os.listdir(left_txt_path)
    left_txt_names = [os.path.splitext(left_txt_name)[0] for left_txt_name in left_txt_names if os.path.splitext(left_txt_name)[1] == '.txt' and left_txt_name != 'listall.txt']
    right_txt_names = os.listdir(right_txt_path)
    right_txt_names = [os.path.splitext(right_txt_name)[0] for right_txt_name in right_txt_names if os.path.splitext(right_txt_name)[1] == '.txt' and right_txt_name != 'listall.txt']

    image_names = list(set(image_names)&set(left_txt_names)&set(right_txt_names))

    for image_name in image_names:
        image = cv2.imdecode(np.fromfile(os.path.join(image_path,image_name+image_postfix),dtype=np.uint8),0)
        left_boundary = loadBoundary(os.path.join(left_txt_path,image_name+'.txt'))
        right_boundary = loadBoundary(os.path.join(right_txt_path,image_name+'.txt'))
        left_bound = max([min(left_boundary)-offset,0])
        right_bound = min([max(right_boundary)+offset,image.shape[1]])
        area = image[:,left_bound:right_bound]
        # cv2.imencode(image_postfix,area)[1].tofile(os.path.join(output_path,'{}_{}_{}{}'.format(image_name,left_bound,right_bound,image_postfix)))
        # cv2.imencode(image_postfix,area)[1].tofile(os.path.join(output_path,image_name+image_postfix))
        cv2.imencode(image_postfix,area)[1].tofile(os.path.join(output_path,'{}___{}_0{}'.format(image_name,left_bound,image_postfix)))
