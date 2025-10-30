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
from sklearn.metrics import mean_squared_error, r2_score


seted_parameter = {
                   'setted_loss_and_accuracy_file_name':'Loss_and_MSE_input_and output',
                   'setted_start_cycle_number':-8,'setted_end_cycle_number':8,'seted_sample_number/2':250,
                  }
data_path =  r'input and output.xlsx'
if not os.path.exists(seted_parameter['setted_loss_and_accuracy_file_name']):
    os.makedirs(seted_parameter['setted_loss_and_accuracy_file_name'])

data_df = pd.read_excel(data_path)
x = data_df.iloc[:, [0, 1]].apply(pd.to_numeric, errors='coerce')  
y = data_df.iloc[:, 2].apply(pd.to_numeric, errors='coerce')       
X_array = x.values.astype(np.float32)
y_array = y.values.astype(np.float32)
#X_array = np.array(X_array)
#y_array = np.array(y_array)



#Define search
cyclic_array = [i for i in range(seted_parameter['setted_start_cycle_number'],seted_parameter['setted_end_cycle_number']+1)]
mse_Tr = np.zeros(len(cyclic_array))
r2_Tr = np.zeros(len(cyclic_array))
mse_Te = np.zeros(len(cyclic_array))
r2_Te = np.zeros(len(cyclic_array))
cycle_index = 0

for cyclic_number in cyclic_array:
	print(cyclic_number)
	#Establish a dataset for processing time-lapse data and sampling data
	x = X_array[10+cyclic_number:590+cyclic_number,:]
	#print(x.shape)
	y = y_array[10:590]
	label_index = []
	for index, value in enumerate(y):
		label_index.append(index)
	selected_index = random.sample(label_index, seted_parameter['seted_sample_number/2']*2)
	selected_x = []
	selected_y = [] 
	for index in selected_index:
		selected_x.append(x[index])
		selected_y.append(y[index])
	X_data = np.vstack(selected_x)
	y_data = np.vstack(selected_y)
	X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3,random_state=42)
	scaler = StandardScaler()
	X_train_scaled = scaler.fit_transform(X_train)
	X_test_scaled = scaler.transform(X_test)
	svm_regressor = SVR(kernel='rbf', C=1.0, gamma='scale', epsilon=0.1)
	svm_regressor.fit(X_train_scaled, y_train)
	
	y_pred_Tr = svm_regressor.predict(X_train_scaled)
	mse_tr = mean_squared_error(y_train, y_pred_Tr)
	r2_tr = r2_score(y_train, y_pred_Tr)

	print(f"均方误差 (MSE): {mse_tr:.4f}")
	print(f"R² 分数: {r2_tr:.4f}")


	y_pred_Te = svm_regressor.predict(X_test_scaled)
	mse_te = mean_squared_error(y_test, y_pred_Te)
	r2_te = r2_score(y_test, y_pred_Te)

	print(f"均方误差 (MSE): {mse_te:.4f}")
	print(f"R² 分数: {r2_te:.4f}")

	mse_Tr[cycle_index] = mse_tr
	r2_Tr[cycle_index] = r2_tr
	mse_Te[cycle_index] = mse_te
	r2_Te[cycle_index] = r2_te

	cycle_index = cycle_index+1            #Calculate the highest precision matrix for the next round of iteration and save it as an index


result_array = np.vstack((cyclic_array,mse_Tr,r2_Tr,mse_Te,r2_Te))
result_array = np.transpose(result_array)
#print(highest_accuracy_array)
df_result = pd.DataFrame(result_array)
df_result.to_excel(os.path.join(seted_parameter['setted_loss_and_accuracy_file_name'], 'Hist_val_accuracy.xlsx'), index=False, header=['Timeshift','mse_Tr','r2_Tr','mse_Te','r2_Te'])
