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
from ResNet10 import ResNet10




seted_parameter = {'setted_SFT_path':['SFT_log_0_time_shift_0.001_0','SFT_log_0_time_shift_0.001_5','SFT_log_0_time_shift_0.001_10','SFT_log_0_time_shift_0.001_15','SFT_log_0_time_shift_0.001_20'],
                   'setted_img_transform_height':64,'setted_img_transform_width':64,
                   'setted_batch_size':512,'seted_num_epochs':100, 'seted_lr':0.0002,
                   'setted_loss_and_accuracy_file_name':'Loss_and_accuracy_AS_1_and_OS_5m_MNet4','seted_sample_number/2':4500,
                   'setted_start_cycle_number':-59,'setted_end_cycle_number':-57}
label_path =  r'2-3label&OS.xls'
if not os.path.exists(seted_parameter['setted_loss_and_accuracy_file_name']):
    os.makedirs(seted_parameter['setted_loss_and_accuracy_file_name'])



def sort_by_number_in_filename(filename):#自定义文件排序函数
    # 使用正则表达式从文件名中提取数字
    match_numbers = re.findall(r'\d+', os.path.basename(filename))
    match_numbers = [int(num) for num in match_numbers]
    #print(match_numbers[0])
    return match_numbers[0]

SFT_png = [pathlib.Path(file_path) for file_path in seted_parameter['setted_SFT_path']]
print(SFT_png)
png_list = [[str(path) for path in sorted(SFT_png_path.glob('*.png'),key=sort_by_number_in_filename)] for SFT_png_path in SFT_png]
#print(len(png_list))
#for ele in png_list:
#	print(ele)


label_file = pd.read_excel(label_path)
label_file = np.array(label_file)
label_file = torch.from_numpy(label_file)
label_data = torch.tensor(label_file)
#print(f'label_shape:',label_data.shape)
#print(label_data)

img_height, img_width = seted_parameter['setted_img_transform_height'], seted_parameter['setted_img_transform_width']
transform = transforms.Compose([
	transforms.ToTensor(),
	transforms.Resize((img_height,img_width))
])


class Self_Dataset(Dataset):
	def __init__(self,file_list, labels):
		self.file_list = file_list
		self.labels = labels
		self.transform = transform

	def __getitem__(self, index):
		img = Image.open(self.file_list[index])
		if self.transform is not None:
			img = self.transform(img)
		label = self.labels[index]
		return img, label

	def __len__(self):
		return len(self.labels)



class MiniCNN4(nn.Module):
    def __init__(self):
        super(MiniCNN4, self).__init__()
        self.feature = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1,bias=False),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(),
                                    nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1,bias=False),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU(),
                                    nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1,bias=False),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU(),
                                    nn.AvgPool2d(8),
                                    nn.Flatten()
                                    )
        self.fc = nn.Linear(256,2)

    def forward(self, x):
        out = self.feature(x)
        #print(out.shape)
        out = self.fc(out)
        return out



#定义训练

def train(model, num_epochs, train_d1, valid_d1):
	loss_hist_train = [0] * num_epochs
	accuracy_hist_train = [0] * num_epochs
	loss_hist_valid = [0] * num_epochs
	accuracy_hist_valid = [0] * num_epochs
	for epoch in range(num_epochs):
		model.train()
		for x_batch, y_batch in train_d1:
			x_batch = x_batch[:,:3,:,:].to(DEVICE)
			#print(y_batch)
			y_batch = y_batch.to(DEVICE)
			pred = model(x_batch)
            #pred_0, pred_1, pred_2 = model(x_batch)
			y_batch = y_batch.to(torch.long)
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
				x_batch = x_batch[:,:3,:,:].to(DEVICE)
				y_batch = y_batch.to(DEVICE)
				pred = model(x_batch)
				y_batch = y_batch.to(torch.long)
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

###############################################################################################
###############################################################################################
########################################定义循环################################################
###############################################################################################
###############################################################################################


#参数设定
cyclic_array = [i for i in range(seted_parameter['setted_start_cycle_number'],seted_parameter['setted_end_cycle_number']+1)]

print(cyclic_array)
highest_accuracy = np.zeros(len(cyclic_array)*len(png_list))
num_epochs = seted_parameter['seted_num_epochs']
highest_accuracy_index = 0

for cyclic_number in cyclic_array:
	print(cyclic_number)

	for file_number,png_list_5ms in enumerate(png_list): #处理时移数据和采样数据建立数据集
		prcessed_png_list = png_list_5ms[2599+cyclic_number:36199+cyclic_number]
		#print(f'prcessed_png_list_len:',len(prcessed_png_list))
		#print(prcessed_png_list)
		processed_label_data = label_data[2599:36199,1]
		print(2599+cyclic_number)
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
		for index in selected_label_index:
			finnal_png_list.append(prcessed_png_list[index])
		print(f'finnal_png_list:',len(finnal_png_list))

		self_dataset = Self_Dataset(finnal_png_list, finnal_label)
		train_size = int(0.7 * len(self_dataset))  # 训练集大小，占比为70%  
		val_size =  len(self_dataset) - train_size    # 测试集大小，占比为30% 
		train_dataset, valid_dataset = random_split(self_dataset, [train_size, val_size])
		torch.manual_seed(1)
		batch_size = seted_parameter['setted_batch_size']
		train_d1_for_CNN = DataLoader(train_dataset,batch_size,shuffle=True)
		valid_d1_for_CNN = DataLoader(valid_dataset,val_size,shuffle=False)

		#定义模型训练
		model = MiniCNN4()
		DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		DEVICE = torch.device('cuda:0')
		model.to(DEVICE)
		print(torch.cuda.is_available())
		#损失函数和优化器
		loss_fn = nn.CrossEntropyLoss()
		optimizer = torch.optim.Adam(model.parameters(), lr=seted_parameter['seted_lr'])

		#训练
		print(f'cyclic',cyclic_number,f'filenumber',file_number,f'start')
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
		ax.plot(x_arr, hist[2], '-o', label='Train acc.')
		ax.plot(x_arr, hist[3], '--<', label='Validation acc.')
		ax.legend(fontsize=15)
		ax.set_xlabel('Epoch', size=15)
		ax.set_ylabel('Accuracy', size=15)
		plt.savefig(os.path.join(seted_parameter['setted_loss_and_accuracy_file_name'], 'Loss_and_accuracy_quan_cycle_{}_cycle_5ms_{}.png'.format(cyclic_number,file_number)))
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
		df_hist.to_excel(os.path.join(seted_parameter['setted_loss_and_accuracy_file_name'], 'Hist_loss_and_accuracy_cycle_{}_cycle_5ms_{}.xlsx'.format(cyclic_number,file_number)), index=False, header=['Epoch','Train_loss','Valid_loss','Train_acc','Valid_acc'])
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

cyclic_index = range(seted_parameter['setted_start_cycle_number']*10,seted_parameter['setted_end_cycle_number']*10+10,2)
print(cyclic_index)
cyclic_index = np.array(cyclic_index)/10
print(cyclic_index)


highest_accuracy_array = np.vstack((cyclic_index,highest_accuracy))
highest_accuracy_array = np.transpose(highest_accuracy_array)
#print(highest_accuracy_array)
df_highest_accuracy = pd.DataFrame(highest_accuracy_array)
df_highest_accuracy.to_excel(os.path.join(seted_parameter['setted_loss_and_accuracy_file_name'], 'Hist_highest_val_accuracy.xlsx'), index=False, header=['Timeshift','Highest_accuracy'])
