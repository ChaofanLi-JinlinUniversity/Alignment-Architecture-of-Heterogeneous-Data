from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from tabpfn import TabPFNClassifier
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
import numpy as np
import random
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR



seted_parameter = {
                   'setted_loss_and_accuracy_file_name':'Loss_and_accuracy_alignment1',
                   'setted_start_cycle_number':-8,'setted_end_cycle_number':8,'seted_sample_number/2':250,
                  }
data_path =  r'features_and_labels.xlsx'
if not os.path.exists(seted_parameter['setted_loss_and_accuracy_file_name']):
    os.makedirs(seted_parameter['setted_loss_and_accuracy_file_name'])

data_df = pd.read_excel(data_path)
x = data_df.iloc[:, [0, 1]].apply(pd.to_numeric, errors='coerce')  
y = data_df.iloc[:, 3].apply(pd.to_numeric, errors='coerce')       
X_array = x.values.astype(np.float32)
y_array = y.values.astype(np.float32)
#X_array = np.array(X_array)
#y_array = np.array(y_array)



#Define search
cyclic_array = [i for i in range(seted_parameter['setted_start_cycle_number'],seted_parameter['setted_end_cycle_number']+1)]
accuracy = np.zeros(len(cyclic_array))
cycle_index = 0

for cyclic_number in cyclic_array:
	print(cyclic_number)
	#Establish a dataset for processing time-lapse data and sampling data
	x = X_array[10+cyclic_number:590+cyclic_number,:]
	#print(x.shape)
	y = y_array[10:590]
	label_0_index = []
	label_1_index = []
	number_label_1 = 0
	for index, value in enumerate(y):
		if int(value) == int(1):
			label_1_index.append(index)
			number_label_1=number_label_1+1
		else:
			label_0_index.append(index)
	selected_label_1_index = random.sample(label_1_index, seted_parameter['seted_sample_number/2'])
	selected_label_0_index = random.sample(label_0_index, seted_parameter['seted_sample_number/2'])
	y_data = np.hstack((np.zeros([seted_parameter['seted_sample_number/2']]),np.ones([seted_parameter['seted_sample_number/2']])))
	selected_label_index = selected_label_0_index+selected_label_1_index
	for j,index in enumerate(selected_label_index):
		if j == 0:
			output = x[int(index),:]
		else:
			output = np.vstack((output,x[int(index),:]))
	X_data = output
	#print(f'x_data_shape:',X_data.shape)
	#print(f'y_data_shape', y_data.shape)

	X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3,random_state=42,stratify=y_data)
	scaler = StandardScaler()
	X_train_scaled = scaler.fit_transform(X_train)
	X_test_scaled = scaler.transform(X_test)
	svm_classifier = SVC(kernel='rbf',C=1.0,gamma='scale',random_state=42)
	svm_classifier.fit(X_train_scaled, y_train)
	y_pred = svm_classifier.predict(X_test_scaled)
	acc = accuracy_score(y_test, y_pred)
	print(acc)

	accuracy[cycle_index] = acc
	cycle_index = cycle_index+1            #Calculate the highest precision matrix for the next round of iteration and save it as an index


accuracy_array = np.vstack((cyclic_array,accuracy))
accuracy_array = np.transpose(accuracy_array)
#print(highest_accuracy_array)
df_accuracy = pd.DataFrame(accuracy_array)
df_accuracy.to_excel(os.path.join(seted_parameter['setted_loss_and_accuracy_file_name'], 'Hist_val_accuracy.xlsx'), index=False, header=['Timeshift','Highest_accuracy'])