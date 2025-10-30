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
import torchvision.models as models
import random
from sklearn.preprocessing import StandardScaler
import time



seted_parameter = {'setted_data_path':'2-3//cutted_tif',
                   'setted_img_transform_height':224,'setted_img_transform_width':224,
                   'setted_batch_size':128,'seted_num_epochs':50, 'seted_lr':0.0001,
                   'setted_loss_and_accuracy_file_name':'Loss_and_MSE_OS_and_feature',
                   'setted_start_cycle_number':-8,'setted_end_cycle_number':8,'seted_sample_number/2':4500,
                  }

#feature_vector_path =  r'2-3//D_2.csv'
feature_vector_path =  r'2-3feature&OS.xlsx'

label_path =  r'2-3label&OS.xls'
if not os.path.exists(seted_parameter['setted_loss_and_accuracy_file_name']):
    os.makedirs(seted_parameter['setted_loss_and_accuracy_file_name'])



def sort_by_number_in_filename(filename):#自定义文件排序函数
    # 使用正则表达式从文件名中提取数字
    match_numbers = re.findall(r'\d+', os.path.basename(filename))
    match_numbers = [int(num) for num in match_numbers]
    #print(match_numbers[1])
    return match_numbers[1]

tif_data = pathlib.Path(seted_parameter['setted_data_path'])
tif_list = [str(path) for path in sorted(tif_data.glob('*.tif'),key=sort_by_number_in_filename)]
print(tif_list)
#for ele in png_list:
#	print(ele)


feature_vector_df = pd.read_excel(feature_vector_path)
print(feature_vector_df.shape)
feature_vector_col = feature_vector_df.iloc[:, :]
#feature_vector_col = pd.to_numeric(feature_vector_col, errors='coerce')
scaler = StandardScaler()          # 1. 创建标准化器对象
feature_vector_data = scaler.fit_transform(feature_vector_col.values)  # 2. 拟合数据并转换
feature_vector_data = torch.tensor(feature_vector_data, dtype=torch.float)

#label_file = np.array(label_file)
#label_file = torch.from_numpy(label_file[:,1])
#label_data = torch.tensor(label_file)
#print(label_data)
#print(f'label_shape:',label_data.shape)
#print(label_data)
label_df = pd.read_excel(label_path)
label_col = label_df.iloc[:, 1]
label_col = pd.to_numeric(label_col, errors='coerce')
label_data = torch.tensor(label_col.values, dtype=torch.long)


img_height, img_width = seted_parameter['setted_img_transform_height'], seted_parameter['setted_img_transform_width']
transform = transforms.Compose([
	transforms.ToTensor(),
	transforms.Resize((img_height,img_width))
])


class Self_Dataset(Dataset):
	def __init__(self,file_list, feature_vectors,labels):
		self.file_list = file_list
		self.feature_vectors = feature_vectors
		self.labels = labels
		self.transform = transform

	def __getitem__(self, index):
		img = Image.open(self.file_list[index])
		if self.transform is not None:
			img = self.transform(img)
		feature_vector = self.feature_vectors[index]
		label = self.labels[index]
		return img, feature_vector, label

	def __len__(self):
		return len(self.labels)

#self_dataset = Self_Dataset(tif_list, feature_vector_data,label_data)
#for tif,vec,labels in self_dataset:
#	print(tif.shape)
#	print(vec.shape)
#	print(labels)



#定义训练

def train(model, num_epochs, train_d1, valid_d1):
	loss_hist_train = [0] * num_epochs
	loss_hist_valid = [0] * num_epochs
	for epoch in range(num_epochs):
		
		model.train()
		for x_batch, y_batch, labels in train_d1:
			x_batch = x_batch[:,:3,:,:].to(DEVICE)
			#print(x_batch.shape)
			#print(y_batch)
			y_batch = y_batch.to(DEVICE)
			pred = model(x_batch)
            #pred_0, pred_1, pred_2 = model(x_batch)
			#y_batch = y_batch.to(torch.long)
			#y_batch = y_batch.squeeze()
			#pred = pred.to(torch.float32)
			#print(pred.shape)
			#print(y_batch.shape)
			loss = loss_fn(pred, y_batch)
            #loss_0 = loss_fn(pred_0, y_batch)
			#loss_1 = loss_fn(pred_1, y_batch)
			#loss_2 = loss_fn(pred_2, y_batch)
			#loss = loss_0 + loss_1 + loss_2
			loss.backward()
			optimizer.step()
			optimizer.zero_grad()
			loss_hist_train[epoch] += loss.item()*y_batch.size(0)
			#print(loss)
		loss_hist_train[epoch] /= len(train_d1.dataset)
		

		t_s = time.time()
		model.eval()
		with torch.no_grad():
			for x_batch, y_batch, labels in valid_d1:
				x_batch = x_batch.to(DEVICE)
				y_batch = y_batch.to(DEVICE)
				pred = model(x_batch)
				loss = loss_fn(pred, y_batch)
				loss_hist_valid[epoch] += \
				    loss.item()*y_batch.size(0)
			loss_hist_valid[epoch] /= len(valid_d1.dataset)

			#print(f'Epoch {epoch+1}loss: '
				  #f'{loss_hist_train[epoch]:}.4f val_loss: ' 
				  #f'{loss_hist_valid[epoch]:}')
		t_e = time.time()
		print(t_e-t_s)
	return loss_hist_train, loss_hist_valid

###############################################################################################
###############################################################################################
########################################定义循环################################################
###############################################################################################
###############################################################################################


#参数设定
cyclic_array = [i for i in range(seted_parameter['setted_start_cycle_number'],seted_parameter['setted_end_cycle_number']+1)]
lowest_loss = np.zeros(len(cyclic_array))
num_epochs = seted_parameter['seted_num_epochs']
#print(cyclic_array)
lowest_loss_index = 0

for cyclic_number in cyclic_array:
	print(cyclic_number)


	#处理时移数据和采样数据建立数据集
	prcessed_png_list = tif_list[900+cyclic_number:37800+cyclic_number]
	#print(f'prcessed_png_list_len:',len(prcessed_png_list))
	#print(prcessed_png_list)

	processed_label_data = label_data[900:37800]
	print(900+cyclic_number)
	label_0_index = []
	label_1_index = []

	number_label_1 = 0
	for index, value in enumerate(processed_label_data):
		if value.numpy() == 1:
			label_1_index.append(index)
			number_label_1=number_label_1+1
			#print(value.numpy())
			#print(index)
		else:
			label_0_index.append(index)
			
	#print(f'number_label_1:',number_label_1)
	#print(len(label_0_index))
	#print(len(label_1_index))
	#selected_label_1_index = label_1_index
	selected_label_1_index = random.sample(label_1_index, seted_parameter['seted_sample_number/2'])
	#selected_label_0_index = random.sample(label_0_index, number_label_1)
	selected_label_0_index = random.sample(label_0_index, seted_parameter['seted_sample_number/2'])
	#print(f'selected_label_0_index:',len(selected_label_0_index))

	#print(processed_label_data)
	#print(torch.zeros([number_label_1]))
	#print(torch.ones([number_label_1]))
	#finnal_label = torch.cat((torch.zeros([number_label_1]),torch.ones([number_label_1])),dim=0)
	finnal_label = torch.cat((torch.zeros([seted_parameter['seted_sample_number/2']]),torch.ones([seted_parameter['seted_sample_number/2']])),dim=0)
	finnal_label = finnal_label.float()
	print(finnal_label)
	#print(finnal_label)
	selected_label_index = selected_label_0_index+selected_label_1_index
	finnal_png_list = []
	finnal_vec_list = []
	for index in selected_label_index:
		finnal_png_list.append(prcessed_png_list[index])
		finnal_vec_list.append(feature_vector_data[index])
	print(f'finnal_png_list:',len(finnal_png_list))
	finnal_vec_list = np.vstack(finnal_vec_list)

	self_dataset = Self_Dataset(finnal_png_list, finnal_vec_list , finnal_label)
	train_size = int(0.7 * len(self_dataset))  # 训练集大小，占比为70%  
	val_size =  len(self_dataset) - train_size    # 测试集大小，占比为30% 
	train_dataset, valid_dataset = random_split(self_dataset, [train_size, val_size])
	torch.manual_seed(1)
	batch_size = seted_parameter['setted_batch_size']
	train_d1_for_CNN = DataLoader(train_dataset,batch_size,shuffle=True)
	valid_d1_for_CNN = DataLoader(valid_dataset,val_size,shuffle=False)
	
  

    #定义模型训练
	model = models.alexnet(pretrained=False)
	#model.features[0] = nn.Conv2d(1, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
	model.classifier[2] = nn.Identity() #Sigmoid()
	model.classifier[5] = nn.Identity() #Sigmoid()
	model.classifier[6] = nn.Linear(in_features=4096, out_features=8, bias=True)

	print(model)
	#model.features[0] = nn.Conv2d(1,64,kernel_size = (3,3),stride = (2,2))
	#model.features[3].squeeze_activation = nn.Identity()
	#model.classifier[1] = nn.Conv2d(512, 1, kernel_size=(1, 1), stride=(1, 1))
	#model.classifier[2] = nn.Identity()
	#print(model)
	DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	DEVICE = torch.device('cuda:0')
	model.to(DEVICE)
	print(torch.cuda.is_available())
	#损失函数和优化器
	loss_fn = nn.MSELoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=seted_parameter['seted_lr'])

	#训练
	print(f'cyclic',cyclic_number,f'start')
	torch.manual_seed(1)
	hist = train(model, num_epochs, train_d1_for_CNN, valid_d1_for_CNN)
	#print(type(hist))

	#绘制精确度曲线
	plt.clf() 
	x_arr = np.arange(len(hist[0])) + 1
	fig = plt.figure(figsize=(12,4))
	ax = fig.add_subplot(1, 2, 1)
	ax.plot(x_arr, hist[0], '-o', label='Train loss')
	ax.plot(x_arr, hist[1], '--<', label='Validation loss')
	ax.legend(fontsize=15)
	ax.set_xlabel('Epoch', size=15)
	ax.set_ylabel('Loss', size=15)
	ax = fig.add_subplot(1, 2, 2)
	ax.plot(x_arr, hist[0], '-o', label='Train loss')
	ax.plot(x_arr, hist[1], '--<', label='Validation loss')
	ax.legend(fontsize=15)
	ax.set_xlabel('Epoch', size=15)
	ax.set_ylabel('Accuracy', size=15)
	plt.savefig(os.path.join(seted_parameter['setted_loss_and_accuracy_file_name'], 'Loss_and_accuracy_quan_{}.png'.format(cyclic_number)))
	plt.clf() #清楚图里面的内容



	#保存精度曲线内容
	Train_loss = np.array(hist[0])
	Valid_loss = np.array(hist[1])


	hist = np.vstack((x_arr,Train_loss))
	hist = np.vstack((hist,Valid_loss))


	hist = np.transpose(hist)
	#print(hist)

	df_hist = pd.DataFrame(hist)
	df_hist.to_excel(os.path.join(seted_parameter['setted_loss_and_accuracy_file_name'], 'Hist_loss_and_accuracy_{}.xlsx'.format(cyclic_number)), index=False, header=['Epoch','Train_loss','Valid_loss'])
	lowest_loss_cycle = np.min(Valid_loss)
	lowest_loss[lowest_loss_index] = lowest_loss_cycle
	lowest_loss_index = lowest_loss_index+1            #计算下一轮循环的最高精度矩阵的保存说索引


#计算最高精度矩阵
#print(cyclic_array)
#print(highest_accuracy)
lowest_loss_array = np.vstack((cyclic_array,lowest_loss))
lowest_loss_array = np.transpose(lowest_loss_array)
#print(highest_accuracy_array)
df_loss = pd.DataFrame(lowest_loss_array)
df_loss.to_excel(os.path.join(seted_parameter['setted_loss_and_accuracy_file_name'], 'Hist_highest_val_accuracy.xlsx'), index=False, header=['Timeshift','Highest_accuracy'])
