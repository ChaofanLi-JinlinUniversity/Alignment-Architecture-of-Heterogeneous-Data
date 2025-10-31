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


if not os.path.exists('PKL'):
    os.makedirs('PKL')

def sort_by_number_in_filename(filename):#自定义文件排序函数
    # 使用正则表达式从文件名中提取数字
    match_numbers = re.findall(r'\d+', os.path.basename(filename))
    match_numbers = [int(num) for num in match_numbers]
    #print(match_numbers[0])
    return match_numbers[0]







#加载标签和文件列表
E_path =pathlib.Path('Excel')
E_list = [str(path) for path in sorted(E_path.glob('*.xlsx'),key=sort_by_number_in_filename)]
print(E_list)

for index, path in enumerate(E_list):
	print(index)
	#print(path)
	df = pd.read_excel(path)
	df = df.iloc[:, 1]
	#print(df)
	df.to_pickle('PKL\\fft{}.pkl'.format(index))
print(f'finsh')

'''
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
				a_data = pd.read_excel(a)
				a_data = np.array(a_data)
				a_data = a_data[:,1]
				b_data = pd.read_excel(b)
				b_data = np.array(b_data)
				b_data = b_data[:,1]
				simlirity = simlirity+np.dot(a_data,b_data)
			b_time = time.time()
			print(f'read_fft_time',b_time-a_time)
			simlirity_list.append(simlirity)

		data = {'time_shift': cyclic_array2,'match_percentage': match_percentage_list,'simlirity': simlirity_list}
		df = pd.DataFrame(data)
		a_time = time.time()
		s = calculate_spearman_correlation(df)
		k = calculate_kendall_correlation(df)
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
df_highest_accuracy.to_excel(seted_parameter['output_file'], index=False, header=['Timeshift','Highest_accuracy'])

'''