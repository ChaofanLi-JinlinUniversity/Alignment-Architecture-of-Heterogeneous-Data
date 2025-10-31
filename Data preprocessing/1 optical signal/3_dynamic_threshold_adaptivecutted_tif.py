import pathlib
import os
import pandas as pd
import numpy as np
import regex as re
import numpy as np
import random
import cv2

if not os.path.exists('dynamic_threshold_adaptivecutted_tif'):
    os.makedirs('dynamic_threshold_adaptivecutted_tif')

SFT_png = pathlib.Path('cutted_tif')


def sort_by_number_in_filename(filename):#自定义文件排序函数
    # 使用正则表达式从文件名中提取数字
    match_numbers = re.findall(r'\d+', os.path.basename(filename))
    match_numbers = [int(num) for num in match_numbers]
    print(match_numbers[1])
    return match_numbers[1]

png_list = [str(path) for path in sorted(SFT_png.glob('*.tif'),key=sort_by_number_in_filename)]
print(png_list)


i = 0
for file in png_list:
    image = cv2.imread(file, 1)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    adaptive_thresh_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV, 51, 3.9)
    cv2.imwrite(os.path.join('dynamic_threshold_adaptivecutted_tif', "33tif_{}.tif".format(i)), adaptive_thresh_image)  # 保存图像
    i += 1  # 确保 i 自增
    