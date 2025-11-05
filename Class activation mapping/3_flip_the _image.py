import pathlib
import os
import pandas as pd
import numpy as np
import regex as re
import numpy as np
import random
import cv2


path = pathlib.Path('1')


def sort_by_number_in_filename(filename):#自定义文件排序函数
    # 使用正则表达式从文件名中提取数字
    match_numbers = re.findall(r'\d+', os.path.basename(filename))
    match_numbers = [int(num) for num in match_numbers]
    #print(match_numbers[7])
    return match_numbers[1]

png_list = [str(path) for path in sorted(path.glob('*.jpg'),key=sort_by_number_in_filename)]
print(png_list)

i = 0
for file in png_list:
	img = cv2.imread(file,1)
	print(img.shape)
	image_1 = img[0:32]
	image_2 = cv2.flip(img[32:452], 0)
	image=np.concatenate((image_1, image_2), axis=0)
	cv2.imwrite("1_{}.jpg".format(i),image_2)#保存图像
	i=i+1
