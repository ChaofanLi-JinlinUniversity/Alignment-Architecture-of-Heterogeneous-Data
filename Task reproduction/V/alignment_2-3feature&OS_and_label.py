import pathlib
import os
import pandas as pd
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
from torch.utils.data import random_split
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch
from torch.nn import functional as F
import regex as re
import matplotlib.pyplot as plt
import torchvision.models as models
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



seted_parameter = {
                   'setted_batch_size':128*8,'seted_num_epochs':100, 'seted_lr':0.001,
                   'setted_loss_and_accuracy_file_name':'Loss_and_accuracy_feature&OS_and_label',
                   'setted_start_cycle_number':-8,'setted_end_cycle_number':8,'seted_sample_number/2':4500,
                  }
feature_path = r'2-3feature&OS.xlsx'
label_path =  r'2-3label&OS.xls'
if not os.path.exists(seted_parameter['setted_loss_and_accuracy_file_name']):
    os.makedirs(seted_parameter['setted_loss_and_accuracy_file_name'])


feature_df = pd.read_excel(feature_path)
feature_col = feature_df.iloc[:, 0:8]
feature_col = np.array(feature_col.values)
feature_data = torch.tensor(feature_col, dtype=torch.float)
#print(feature_data.shape)

label_df = pd.read_excel(label_path)
label_col = label_df.iloc[:, 1]
label_col = pd.to_numeric(label_col, errors='coerce')
label_data = torch.tensor(label_col.values, dtype=torch.long)




#定义训练

def train(model, num_epochs, train_d1, valid_d1):
	loss_hist_train = [0] * num_epochs
	accuracy_hist_train = [0] * num_epochs
	loss_hist_valid = [0] * num_epochs
	accuracy_hist_valid = [0] * num_epochs
	for epoch in range(num_epochs):
		model.train()
		for x_batch, y_batch in train_d1:
			x_batch = x_batch.to(DEVICE)
			#print(x_batch.shape)
			#print(y_batch)
			y_batch = y_batch.to(DEVICE)
			pred = model(x_batch)
            #pred_0, pred_1, pred_2 = model(x_batch)
			y_batch = y_batch.squeeze()
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
			is_correct = (
				torch.argmax(pred, dim=1) == y_batch
			).float()
			accuracy_hist_train[epoch] +=is_correct.sum().cpu()
		loss_hist_train[epoch] /= len(train_d1.dataset)
		accuracy_hist_train[epoch] /= len(train_d1.dataset)

		model.eval()
		with torch.no_grad():
			for x_batch, y_batch in valid_d1:
				x_batch = x_batch.to(DEVICE)
				y_batch = y_batch.to(DEVICE)
				pred = model(x_batch)
				y_batch = y_batch.squeeze()
				loss = loss_fn(pred, y_batch)
				loss_hist_valid[epoch] += \
				    loss.item()*y_batch.size(0)
				is_correct =  (
					torch.argmax(pred, dim=1) == y_batch
				).float()
				accuracy_hist_valid[epoch] += is_correct.sum().cpu()
			loss_hist_valid[epoch] /= len(valid_d1.dataset)
			accuracy_hist_valid[epoch] /= len(valid_d1.dataset)

			print(f'Epoch {epoch+1}accuracy: '
				  f'{accuracy_hist_train[epoch]:}.4f val_accuracy: ' 
				  f'{accuracy_hist_valid[epoch]:}')
	return loss_hist_train, loss_hist_valid, accuracy_hist_train, accuracy_hist_valid

class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 4))
    def forward(self, x):
    	return self.net(x)

###############################################################################################
###############################################################################################
######################################  Define cycle  #########################################
###############################################################################################
###############################################################################################


#参数设定
cyclic_array = [i for i in range(seted_parameter['setted_start_cycle_number'],seted_parameter['setted_end_cycle_number']+1)]
highest_accuracy = np.zeros(len(cyclic_array))
num_epochs = seted_parameter['seted_num_epochs']
highest_accuracy_index = 0

for cyclic_number in cyclic_array:
	print(cyclic_number)
	#处理时移数据和采样数据建立数据集

	prcessed_feature_data = feature_data[900+cyclic_number:37800+cyclic_number]
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
	#print(finnal_label)
	#print(finnal_label)
	selected_label_index = selected_label_0_index+selected_label_1_index
	for i,index in enumerate(selected_label_index):
		if i == 0:
			selected_feature = prcessed_feature_data[index].unsqueeze(0)
			#print(selected_feature.shape)
		else:
			selected_feature = torch.cat((selected_feature,prcessed_feature_data[index].unsqueeze(0)),dim=0)
			#print(selected_feature.shape)
	X_train, X_test, y_train, y_test = train_test_split(selected_feature, finnal_label, test_size=0.3, random_state=42, stratify= finnal_label)
	scaler = StandardScaler()
	X_train = scaler.fit_transform(X_train)
	X_test = scaler.transform(X_test)
	
	X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
	y_train_tensor = torch.tensor(y_train, dtype=torch.long).view(-1, 1)
	X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
	y_test_tensor = torch.tensor(y_test, dtype=torch.long).view(-1, 1)

	train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
	test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

	batch_size = seted_parameter['setted_batch_size']
	train_d1_for_CNN = DataLoader(train_dataset,batch_size,shuffle=True)
	test_d1_for_CNN = DataLoader(test_dataset,batch_size,shuffle=False)
	
    

    #定义模型训练
	model = MLP(8)
	DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	DEVICE = torch.device('cuda:0')
	model.to(DEVICE)
	print(torch.cuda.is_available())
	#损失函数和优化器
	loss_fn = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=seted_parameter['seted_lr'])

	#训练
	print(f'cyclic',cyclic_number,f'start')
	torch.manual_seed(1)
	hist = train(model, num_epochs, train_d1_for_CNN, test_d1_for_CNN)
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
	ax.plot(x_arr, hist[2], '-o', label='Train acc.')
	ax.plot(x_arr, hist[3], '--<', label='Validation acc.')
	ax.legend(fontsize=15)
	ax.set_xlabel('Epoch', size=15)
	ax.set_ylabel('Accuracy', size=15)
	plt.savefig(os.path.join(seted_parameter['setted_loss_and_accuracy_file_name'], 'Loss_and_accuracy_quan_{}.png'.format(cyclic_number)))
	plt.clf() #清楚图里面的内容



	#保存精度曲线内容
	Train_loss = np.array(hist[0])
	Valid_loss = np.array(hist[1])
	Train_acc = np.array(hist[2])
	Valid_acc = np.array(hist[3])

	hist = np.vstack((x_arr,Train_loss))
	hist = np.vstack((hist,Valid_loss))
	hist = np.vstack((hist,Train_acc))
	hist = np.vstack((hist,Valid_acc))

	hist = np.transpose(hist)
	#print(hist)

	df_hist = pd.DataFrame(hist)
	df_hist.to_excel(os.path.join(seted_parameter['setted_loss_and_accuracy_file_name'], 'Hist_loss_and_accuracy_{}.xlsx'.format(cyclic_number)), index=False, header=['Epoch','Train_loss','Valid_loss','Train_acc','Valid_acc'])
	print(Valid_acc)
	print(np.max(Valid_acc))
	print(highest_accuracy)
	print(highest_accuracy_index)
	highest_accuracy_cycle = np.max(Valid_acc)
	print(highest_accuracy_cycle)
	highest_accuracy[highest_accuracy_index] = highest_accuracy_cycle #计算每次循环的最高精度
	print(highest_accuracy)
	highest_accuracy_index = highest_accuracy_index+1            #计算下一轮循环的最高精度矩阵的保存说索引


#计算最高精度矩阵
#print(cyclic_array)
#print(highest_accuracy)
highest_accuracy_array = np.vstack((cyclic_array,highest_accuracy))
highest_accuracy_array = np.transpose(highest_accuracy_array)
#print(highest_accuracy_array)
df_highest_accuracy = pd.DataFrame(highest_accuracy_array)
df_highest_accuracy.to_excel(os.path.join(seted_parameter['setted_loss_and_accuracy_file_name'], 'Hist_highest_val_accuracy.xlsx'), index=False, header=['Timeshift','Highest_accuracy'])