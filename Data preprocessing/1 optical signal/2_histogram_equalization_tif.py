import pathlib
import os
import pandas as pd
import numpy as np
import regex as re
import numpy as np
import random
import cv2

if not os.path.exists('histogram_equalization_tif'):
    os.makedirs('histogram_equalization_tif')

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
    # 确保 image 数据类型为 uint8
    image = np.clip(image, 0, 255).astype(np.uint8)
    # 将彩色图像转换为YUV色彩空间
    yuv_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    # 对Y通道（亮度通道）进行直方图均衡
    yuv_image[:, :, 0] = cv2.equalizeHist(yuv_image[:, :, 0])
    # 将图像从YUV色彩空间转换回BGR色彩空间
    equalized_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR)
    cv2.imwrite(os.path.join('histogram_equalization_tif', "33tif_{}.tif".format(i)), equalized_image)  # 保存图像
    i += 1  # 确保 i 自增
    