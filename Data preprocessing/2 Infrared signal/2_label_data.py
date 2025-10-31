import pathlib
import os
import regex as re
import cv2
import numpy as np
import pandas as pd

seted_parameter = {'setted_input_path':'cutted_png',
                   'setted_output_file_name':33
                  }


def sort_by_number_in_filename(filename):#自定义文件排序函数
    # 使用正则表达式从文件名中提取数字
    match_numbers = re.findall(r'\d+', os.path.basename(filename))
    match_numbers = [int(num) for num in match_numbers]
    #print(match_numbers[0])
    return match_numbers[1]


SFT_png = pathlib.Path(seted_parameter['setted_input_path'])
png_list = [str(path) for path in sorted(SFT_png.glob('*.png'),key=sort_by_number_in_filename)]
#print(f'len_png_list:',len(png_list))
#print(f'png_list:',png_list)

result = np.zeros([len(png_list),3])

for index,img_name in enumerate(png_list):
    image = cv2.imread(img_name,1)
    #print(image)
    #print(f'image_shape:',image.shape)
    number = 0
    for i, row in enumerate(image):
        for j, element in enumerate(row):
            if image[i,j,0]==0 and image[i,j,1]==255 and image[i,j,2]==0:
                number = number+1
    print(f'index:',index)
    #print(f'number',number)
    
    result[index,0] = index
    if number >= 1:
        result[index,1] = 1
    result[index,2] =number

#print(f'result:\n',result)


df_result = pd.DataFrame(result)
df_result.to_excel('{}_result.xlsx'.format(seted_parameter['setted_output_file_name']), index=False, header=['index','result','number'])