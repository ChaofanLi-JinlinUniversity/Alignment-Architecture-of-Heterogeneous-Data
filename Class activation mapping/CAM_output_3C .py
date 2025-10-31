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
import torchvision.models as models
import time
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image




seted_parameter = {'setted_SFT_path':'SFT_log_0_time_shift_0.001_0',
                   'setted_img_transform_height':64,'setted_img_transform_width':64,
                   'setted_batch_size':512, 'setted_start_cycle_number':-58, 'setted_end_cycle_number':-58,
                   'setted_CAM_output':'CAM_3C',
                   'seted_sample_number/2':4500,
                  }
label_path =  r'2-3label&OS.xls'
if not os.path.exists(seted_parameter['setted_CAM_output']):
    os.makedirs(seted_parameter['setted_CAM_output'])


def sort_by_number_in_filename(filename):#自定义文件排序函数
    # 使用正则表达式从文件名中提取数字
    match_numbers = re.findall(r'\d+', os.path.basename(filename))
    match_numbers = [int(num) for num in match_numbers]
    #print(match_numbers[0])
    return match_numbers[0]

SFT_png = pathlib.Path(seted_parameter['setted_SFT_path'])
png_list = [str(path) for path in sorted(SFT_png.glob('*.png'),key=sort_by_number_in_filename)]
#print(png_list)
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
			img = self.transform(img)[:3,:,:]
		label = self.labels[index]
		return img, label

	def __len__(self):
		return len(self.labels)


#6_DenseNet
#6_标准DenseNet
class DenseLayer(nn.Module):
    """Basic unit of DenseBlock (DenseLayer) """
    def __init__(self, input_c, growth_rate, bn_size, drop_rate):
        super(DenseLayer, self).__init__()

        self.bn1 = nn.BatchNorm2d(input_c)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(input_c, bn_size * growth_rate,
                               kernel_size=1, stride=1, bias=False)

        self.bn2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.drop_rate = drop_rate

    def forward(self, inputs):
        # 1×1卷积 bottleneck瓶颈层
        output = self.bn1(inputs)
        output = self.relu1(output)
        output = self.conv1(output)
        # 3×3卷积
        output = self.bn2(output)
        output = self.relu2(output)
        output = self.conv2(output)

        if self.drop_rate > 0:
            output = F.dropout(output, p=self.drop_rate)
        return output

class DenseBlock(nn.ModuleDict):
    def __init__(self, num_layers, input_c, bn_size, growth_rate, drop_rate):
        super(DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = DenseLayer(input_c + i * growth_rate,
                                growth_rate=growth_rate,
                                bn_size=bn_size,
                                drop_rate=drop_rate)
            self.add_module("denselayer%d" % (i + 1), layer)
    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.items():
            concat_features = torch.cat(features, 1)
            # 当前DenseLayer的输出特征
            new_features = layer(concat_features)
            # 收集所有DenseLayer的输出特征
            features.append(new_features)
        return torch.cat(features, 1)

class Transition(nn.Module):
    def __init__(self, input_c, output_c):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(input_c)
        self.relu = nn.ReLU(inplace=True)
        # 1×1卷积
        self.conv = nn.Conv2d(input_c, output_c,
                              kernel_size=1, stride=1, bias=False)
        # 2×2池化
        self.pool = nn.AvgPool2d(2, stride=2)

    def forward(self, input):
        output = self.bn(input)
        output = self.relu(output)
        output = self.conv(output)
        output = self.pool(output)
        return output


class DenseNet(nn.Module):
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=64, bn_size=4,
                 compression_rate=0.5, drop_rate=0, num_classes=2):
        super(DenseNet, self).__init__()

        # 前部 conv+bn+relu+pool
        self.features = nn.Sequential(
            # 第一层
            nn.Conv2d(3, num_init_features, kernel_size=1, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_init_features),
            nn.ReLU(inplace=True),
            # 第二层
            #nn.MaxPool2d(3, stride=2, padding=1)
        )

        # 中部 DenseBlock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = DenseBlock(num_layers=num_layers,
                               input_c=num_features,
                               bn_size=bn_size,
                               growth_rate=growth_rate,
                               drop_rate=drop_rate)
            # 新增DenseBlock
            self.features.add_module("denseblock%d" % (i + 1), block)
            # 更新通道数
            num_features = num_features + num_layers * growth_rate

            # 除去最后一层DenseBlock不需要加Transition来连接两个相邻的DenseBlock
            if i != len(block_config) - 1:
                transition = Transition(input_c=num_features, output_c=int(num_features * compression_rate))
                # 添加Transition
                self.features.add_module("transition%d" % (i + 1), transition)
                # 更新通道数
                num_features = int(num_features * compression_rate)

        # 后部 bn+ReLU
        self.tail = nn.Sequential(
            nn.BatchNorm2d(num_features),
            nn.ReLU(inplace=True)
        )

        # 分类器 classification
        self.classifier = nn.Linear(num_features, num_classes)

        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        tail_output = self.tail(features)
        # 平均池化
        out = F.adaptive_avg_pool2d(tail_output, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out

def densenet_x(**kwargs):
    # Top-1 error: 25.35%
    # 'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth'
    return DenseNet(growth_rate=32,
                    block_config=(2, 4, 8, 4),
                    num_init_features=64,
                    bn_size=4,
                    compression_rate=0.5,
                    **kwargs)


#densenet_small = densenet_small()
#a =torch.ones(1,3,64,64)
#b = densenet_small(a)
#print(b)


#定义类激活映射辅助函数
def myimshows(imgs, titles=False, fname="test.jpg", size=6):
    lens = len(imgs)
    fig = plt.figure(figsize=(size * lens, size))
    if titles == False:
        titles = "0123456789"
    for i in range(1, lens + 1):
        cols = 100 + lens * 10 + i
        plt.xticks(())
        plt.yticks(())
        plt.subplot(cols)
        if len(imgs[i - 1].shape) == 2:
            plt.imshow(imgs[i - 1], cmap='Reds')
        else:
            plt.imshow(imgs[i - 1])
        plt.title(titles[i - 1])
    plt.xticks(())
    plt.yticks(())
    plt.savefig(fname, bbox_inches='tight')
    #plt.show()
    plt.clf()


def tensor2img(tensor, heatmap=False, shape=(224, 224)):
    np_arr = tensor.detach().numpy()  # [0]
    # 对数据进行归一化
    if np_arr.max() > 1 or np_arr.min() < 0:
        np_arr = np_arr - np_arr.min()
        np_arr = np_arr / np_arr.max()
    # np_arr=(np_arr*255).astype(np.uint8)
    if np_arr.shape[0] == 1:
        np_arr = np.concatenate([np_arr, np_arr, np_arr], axis=0)
    np_arr = np_arr.transpose((1, 2, 0))
    return np_arr



#定义训练
def CAM_caculate(model, valid_d1):
	loss_hist_valid = [0] * 1
	accuracy_hist_valid = [0] * 1
	model.eval()
	target_layers = [model.features.denseblock4.denselayer4]  # 如果传入多个layer，cam输出结果将会取均值
	for x_batch, y_batch in valid_d1:
		x_batch = x_batch.to(DEVICE)
		y_batch = y_batch.to(DEVICE)
		pred = model(x_batch)
		y_batch = y_batch.to(torch.long)
		y_batch = y_batch.squeeze()
		loss = loss_fn(pred, y_batch)
		loss_hist_valid[0] += \
			loss.item()*y_batch.size(0)
		is_correct =  (
			torch.argmax(pred, dim=1) == y_batch
		).float()
		pred_result = torch.argmax(pred, dim=1)
		accuracy_hist_valid[0] += is_correct.sum().cpu()
		for index, input_tensor in enumerate(x_batch):
			if is_correct[index].cpu().item() == 1:
				input_tensor = input_tensor.unsqueeze(0)
				with GradCAM(model=model, target_layers=target_layers) as cam:
					targets = None  # 选定目标类别，如果不设置，则默认为分数最高的那一类
					grayscale_cams = cam(input_tensor=input_tensor, targets=targets)  # targets=None 自动调用概率最大的类别显示
					for grayscale_cam, tensor in zip(grayscale_cams, input_tensor):
						rgb_img = tensor2img(tensor.cpu())
						visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
						#print(1-pred_result[index])
						#print(pred_result[index])
						print(f'rgb_shape',rgb_img.shape)
						myimshows([rgb_img, grayscale_cam, visualization], ["image", "cam", "image + cam"],fname=os.path.join(seted_parameter['setted_CAM_output'], "cam_{}_{}.jpg".format(1-pred_result[index].cpu().item(),index)))

		loss_hist_valid[0] /= len(valid_d1.dataset)
		accuracy_hist_valid[0] /= len(valid_d1.dataset)

		print(f'Epoch {0+1}accuracy:',f'{accuracy_hist_valid[0]:}')
	return loss_hist_valid, accuracy_hist_valid


###############################################################################################
###############################################################################################
########################################定义循环################################################
###############################################################################################
###############################################################################################


#参数设定
cyclic_array = [i for i in range(seted_parameter['setted_start_cycle_number'],seted_parameter['setted_end_cycle_number']+1)]

highest_accuracy_index = 0

for cyclic_number in cyclic_array:
	print(cyclic_number)


	#处理时移数据和采样数据建立数据集
	prcessed_png_list = png_list[1599+cyclic_number:37226+cyclic_number]
	#print(f'prcessed_png_list_len:',len(prcessed_png_list))
	#print(prcessed_png_list)

	processed_label_data = label_data[1599:37226,1]
	#print(2599+cyclic_number)
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
	finnal_png_list = []
	for index in selected_label_index:
		finnal_png_list.append(prcessed_png_list[index])
	#print(f'finnal_png_list:',len(finnal_png_list))

	self_dataset = Self_Dataset(finnal_png_list, finnal_label)
	torch.manual_seed(1)
	batch_size = seted_parameter['setted_batch_size']
	d1_for_CNN = DataLoader(self_dataset,batch_size,shuffle=True)

    #定义模型训练
	densenet_small = torch.load('model_densenet_-58_53_3C.pt')
	print(densenet_small)


	DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	DEVICE = torch.device('cuda:0')
	densenet_small.to(DEVICE)
	print(torch.cuda.is_available())
	#损失函数和优化器
	loss_fn = nn.CrossEntropyLoss()

	#训练
	print(f'cyclic',cyclic_number,f'start')
	torch.manual_seed(1)
	hist = CAM_caculate(densenet_small, d1_for_CNN)
	#print(type(hist))


'''
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
	TP_list_result = np.array(hist[4])
	TN_list_result = np.array(hist[5])
	FP_list_result = np.array(hist[6])
	FN_list_result = np.array(hist[7])


	hist = np.vstack((x_arr,Train_loss))
	hist = np.vstack((hist,Valid_loss))
	hist = np.vstack((hist,Train_acc))
	hist = np.vstack((hist,Valid_acc))
	hist = np.vstack((hist,TP_list_result))
	hist = np.vstack((hist,TN_list_result))
	hist = np.vstack((hist,FP_list_result))
	hist = np.vstack((hist,FN_list_result))

	hist = np.transpose(hist)
	#print(hist)

	df_hist = pd.DataFrame(hist)
	df_hist.to_excel(os.path.join(seted_parameter['setted_loss_and_accuracy_file_name'], 'Hist_loss_and_accuracy_{}.xlsx'.format(cyclic_number)), index=False, header=['Epoch','Train_loss','Valid_loss','Train_acc','Valid_acc','TP','TN','FP','FN'])
	#print(Valid_acc)
	#print(np.max(Valid_acc))
	#print(highest_accuracy)
	#print(highest_accuracy_index)
	highest_accuracy_cycle = np.max(Valid_acc)
	#print(highest_accuracy_cycle)
	highest_accuracy[highest_accuracy_index] = highest_accuracy_cycle #计算每次循环的最高精度
	#print(highest_accuracy)
	highest_accuracy_index = highest_accuracy_index+1            #计算下一轮循环的最高精度矩阵的保存说索引


#计算最高精度矩阵
#print(cyclic_array)
#print(highest_accuracy)
highest_accuracy_array = np.vstack((cyclic_array,highest_accuracy))
highest_accuracy_array = np.transpose(highest_accuracy_array)
#print(highest_accuracy_array)
df_highest_accuracy = pd.DataFrame(highest_accuracy_array)
df_highest_accuracy.to_excel(os.path.join(seted_parameter['setted_loss_and_accuracy_file_name'], 'Hist_highest_val_accuracy.xlsx'), index=False, header=['Timeshift','Highest_accuracy'])
'''