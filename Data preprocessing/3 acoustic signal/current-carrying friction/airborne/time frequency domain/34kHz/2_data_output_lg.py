import pathlib
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
from multiprocessing import Process
import xlrd
import xlwt  #对xls文件进行改写
#from xlutils.copy import copyasf 
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import glob
import revised_thinkdsp
import matplotlib
import csv  
import re


#0类数据,处理原始波形数据的函数，把原始的波形数据进行删除表头和转化为数组操作（同时适用于0类数据和1类数据）
def read_csv_from_wave_data(file):  
    with open(file, 'r') as f:  
        reader = csv.reader(f, delimiter='\t')  #使用tab作为分隔符(关键)  
        origin_data = [row for row in reader]          #读取所有行  
        del origin_data[:12]
        data_output = [row[0].rstrip(',').strip().split(',') for row in origin_data]       
        data_output = [[float(num.strip()) for num in row] for row in data_output]
        data_output = np.array(data_output)
        #print(data_output.dtype)
    return data_output

def save_excel_of_pinyu(ls2_01,number):
        #ls2为二维数组
        workbook = xlwt.Workbook()
        sheet = workbook.add_sheet("Sheet")
        for i in range(len(ls2_01)):
                for j in range(len(ls2_01[i])):
                        sheet.write(i, j, ls2_01[i][j])
        workbook.save("X_pinyu_{}.xls".format(number))

def save_excel_of_shiyu(ls2_01,number):
        #ls2为二维数组
        workbook = xlwt.Workbook()
        sheet = workbook.add_sheet("Sheet")
        for i in range(len(ls2_01)):
                for j in range(len(ls2_01[i])):
                        sheet.write(i, j, ls2_01[i][j])
        workbook.save("X_shiyu_{}.xls".format(number))


#自定义文件排序函数
def sort_by_number_in_filename(filename):
    # 使用正则表达式从文件名中提取数字
    match_numbers = re.findall(r'\d+', os.path.basename(filename))
    match_numbers = [int(num) for num in match_numbers]
    print(match_numbers[4])
    return match_numbers[4]


def exists(variable):
    return variable is not None

number = 0
i_file = 0
csv_file_0 = glob.glob('origin_data/*.csv')
remain_data = np.zeros((50001,2))
for file_0 in sorted(csv_file_0, key = sort_by_number_in_filename):   #把路径下的文件按照时间顺序排序并依次读取 key = os.path.getctime
    origin_data = read_csv_from_wave_data(file_0)
    print(file_0,f'start')
    if remain_data.shape[0] < int(50000):
        data = np.vstack((remain_data, origin_data))
    else:
        data = origin_data
    
    cycle_number = data.shape[0]//50000
    remain_number = data.shape[0]%50000
    remain_data = data[50000*cycle_number:,:]
    for i in range(cycle_number):
        selected_data = data[i*50000:(i+1)*50000,:]
        #print(i)
        #print(selected_data.shape)
        ts_0 = np.linspace(0,49999,50000)
        ts_0 = 1/2000000*ts_0
        ys_0 = selected_data[:,1].ravel()
        
        #输出时域信号
        '''
        shiyu_shuchu = np.transpose(np.vstack((ts_0, ys_0)))
        save_excel_of_shiyu(shiyu_shuchu,number)
        plt.clf() #清楚图里面的内容
        plt.plot(ts_0, ys_0)
        plt.savefig('P_shiyu_{}.png'.format(number))
        plt.clf() #清楚图里面的内容
        '''

        #输出频域信号
        singnal_0 = revised_thinkdsp.Wave(ys=ys_0, ts=ts_0, framerate=2000000)
        #singnal_0.plot()
        #plt.show()
        
        spectrum_0 = singnal_0.make_spectrum(full=False)
        fs_0, amps_0 = spectrum_0.plot()
        '''
        #print(ys_0.shape)
        #print(ts_0.shape)
        #amps_0 = np.log10(amps_0)
        amps_0_max = np.max(amps_0[:851])
        amps_0_min = np.min(amps_0[:851])
        amps_0 = (amps_0 - amps_0_min)/(amps_0_max - amps_0_min )
        pinyu_shuchu = np.transpose(np.vstack((fs_0, amps_0)))
        save_excel_of_pinyu(pinyu_shuchu[:851,:],number)
        plt.clf() #清楚图里面的内容
        plt.plot(fs_0[:851], amps_0[:851])
        plt.savefig('P_pinyu_{}.png'.format(number))
        plt.clf() #清楚图里面的内容 
        '''

 
        #输出时频图
        spectrogram_0 = singnal_0.make_spectrogram(seg_length=2382) #这里可以调整，调节两个分辨率的制衡
        array_0 = spectrogram_0.plot(high=34000)  #这里可以设置时频图里面显示的最高频率，然后把时频图的数据读取给array_0
        #print(array_0.shape)
        #array_0 = np.log10(array_0)
        array_max = np.max(array_0)
        array_min = np.min(array_0)
        array_0 = (array_0 - array_min)/(array_max - array_min)
        matplotlib.image.imsave('SFT_0/P_shipintu_{}.png'.format(number), array_0)

        number = number+1
    print(i_file)
    i_file = i_file+1

print('finsh')#ABC