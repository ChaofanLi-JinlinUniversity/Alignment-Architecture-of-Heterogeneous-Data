import pathlib
import os
import pandas as pd
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from torch.nn import functional as F
import regex as re
import matplotlib.pyplot as plt
import numpy as np
import random
import time
from scipy.stats import spearmanr
from scipy.stats import kendalltau



seted_parameter = {'setted_FFT_path':['FFT_log_0_time_shift_0.001_0\\PKL', 'FFT_log_0_time_shift_0.001_5\\PKL', 'FFT_log_0_time_shift_0.001_10\\PKL', 'FFT_log_0_time_shift_0.001_15\\PKL', 'FFT_log_0_time_shift_0.001_20\\PKL'],  
                   'output_file':'Time_shift_and_correlation_coefficient_5000.xlsx',
                   'setted_start_cycle_number':-60,'setted_end_cycle_number':-52,'seted_sample_number/2':2500,
                  }
label_path =  r'2-3label&OS.xls'

#建立输出文件路径
#if not os.path.exists(seted_parameter['Time_shift_and_similarity_data_file']):
#    os.makedirs(seted_parameter['Time_shift_and_similarity_data_file'])

def sort_by_number_in_filename(filename):#自定义文件排序函数
    # 使用正则表达式从文件名中提取数字
    match_numbers = re.findall(r'\d+', os.path.basename(filename))
    match_numbers = [int(num) for num in match_numbers]
    #print(match_numbers[0])
    return match_numbers[0]

def calculate_kendall_correlation(df, sheet_name=0, x_col=1, y_col=2):
    try:
        x = df.iloc[:, x_col] if isinstance(x_col, int) else df[x_col]
        y = df.iloc[:, y_col] if isinstance(y_col, int) else df[y_col]
    except KeyError as e:
        print(f"列不存在: {e}")
        return None
    
    # 删除包含缺失值的行
    data = pd.DataFrame({'x': x, 'y': y}).dropna()
    if len(data) < 3:
        print("错误: 有效数据点不足(至少需要3个非缺失值)")
        return None
    
    # 计算肯德尔相关系数
    correlation, p_value = kendalltau(data['x'], data['y'])
    return correlation, p_value

def calculate_spearman_correlation(file_path, sheet_name=0, x_col=1, y_col=2):
	try:
		x = df.iloc[:, x_col] if isinstance(x_col, int) else df[x_col]
		y = df.iloc[:, y_col] if isinstance(y_col, int) else df[y_col]
	except KeyError as e:
		print(f"列不存在: {e}")
		return None
	# 删除包含缺失值的行
	data = pd.DataFrame({'x': x, 'y': y}).dropna()
	if len(data) < 3:
		print("错误: 有效数据点不足(至少需要3个非缺失值)")
		return None
	# 计算斯皮尔曼相关系数
	correlation, p_value = spearmanr(data['x'], data['y'])
	return correlation, p_value

# 预加载所有FFT数据到内存
def preload_fft_data(fft_paths):
    fft_data = {}
    for path_idx, fft_path in enumerate(fft_paths):
        fft_dir = pathlib.Path(fft_path)
        fft_files = sorted(fft_dir.glob('*.xlsx'), key=sort_by_number_in_filename)
        fft_data[path_idx] = []
        
        for file_path in fft_files:
            df = pd.read_excel(file_path)
            fft_data[path_idx].append(df.iloc[:, 1].values)  # 只存储第二列数据
    
    return fft_data




#加载标签和文件列表
fft_path = [pathlib.Path(file_path) for file_path in seted_parameter['setted_FFT_path']]
print(fft_path)
fft_list = [[str(path) for path in sorted(fft_path.glob('*.pkl'),key=sort_by_number_in_filename)] for fft_path in fft_path]


label_file = pd.read_excel(label_path)
label_data = np.array(label_file)
label_data = label_data[:,1]
#print(label_data)



#定义循环
cyclic_array = [i for i in range(seted_parameter['setted_start_cycle_number'],seted_parameter['setted_end_cycle_number']+1)]

#print(cyclic_array)
index = 0
s_correlation = []
k_correlation = []

for cyclic_number in cyclic_array:
	print(cyclic_number)
	
	for file_number,fft_list_5ms in enumerate(fft_list):
		print(file_number)
		processed_label_data = label_data[2599:36199]
		#print(2599+cyclic_number)
		label_0_index = []
		label_1_index = []
		start_time = time.time()

		for index, value in enumerate(processed_label_data):
			if value == 1:
				label_1_index.append(index)
				#print(f'1')
			else:
				label_0_index.append(index)
				#print(f'0')
		selected_label_1_index = random.sample(label_1_index, seted_parameter['seted_sample_number/2'])
		selected_label_0_index = random.sample(label_0_index, seted_parameter['seted_sample_number/2'])
		n_half = seted_parameter['seted_sample_number/2']  # 计算一次并复用

		label_contrast1 = torch.cat((torch.zeros(n_half), torch.ones(n_half))).numpy()######################################
		label_contrast1 = label_contrast1.tolist()

		selected_label_index = selected_label_0_index+selected_label_1_index

		fft_contrast1 = []
		for index in selected_label_index:
			fft_contrast1.append(fft_list_5ms[2599+cyclic_number+index])
		cyclic_array2 = [i for i in range(-8,8+1)]
		match_percentage_list = []
		simlirity_list = []
		for cyclic_number2 in cyclic_array2:
			label_contrast2 = []
			fft_contrast2 = []
			for index in selected_label_index:
				label_contrast2.append(label_data[2599+index+cyclic_number2])
				fft_contrast2.append(fft_list_5ms[2599+index+cyclic_number+cyclic_number2])
			#print(len(label_contrast2))
			#print(len(fft_contrast2))
			#得到label_contrast1, label_contrast2, excel_contrast1, excel_contrast2

			matching_elements = sum(1 for a, b in zip(label_contrast1, label_contrast2) if a == b)
			match_percentage = matching_elements / len(label_contrast1)
			#print(match_percentage)
			match_percentage_list.append(match_percentage)


			print(f'################################################')
			simlirity = 0
			a_time = time.time()
			print(f'len_contrast1',len(fft_contrast1))
			print(f'len_contrast2',len(fft_contrast2))
			for a, b in zip(fft_contrast1, fft_contrast2):
				a_data = pd.read_pickle(a)
				a_data = np.array(a_data)
				#a_data = a_data[:,1]
				b_data = pd.read_pickle(b)
				b_data = np.array(b_data)
				#b_data = b_data[:,1]
				simlirity = simlirity+np.dot(a_data,b_data)
			b_time = time.time()
			print(f'read_fft_time',b_time-a_time)
			simlirity_list.append(simlirity)

		data = {'time_shift': cyclic_array2,'match_percentage': match_percentage_list,'simlirity': simlirity_list}
		df = pd.DataFrame(data)
		a_time = time.time()
		s,_ = calculate_spearman_correlation(df)
		k,_ = calculate_kendall_correlation(df)
		s_correlation.append(s)
		k_correlation.append(k)
		end_time = time.time()
		print(end_time-start_time,f's')
		print(f'caculate_correlation_time',end_time-a_time)





cyclic_index = range(seted_parameter['setted_start_cycle_number']*10,seted_parameter['setted_end_cycle_number']*10+10,2)
#print(cyclic_index)
cyclic_index = np.array(cyclic_index)/10
#print(cyclic_index)
data = {'cyclic_index': cyclic_index,'s_correlation': s_correlation,'k_correlation': k_correlation}
df = pd.DataFrame(data)
df.to_excel(seted_parameter['output_file'], index=False, header=['Timeshift','s_correlation','k_correlation'])

