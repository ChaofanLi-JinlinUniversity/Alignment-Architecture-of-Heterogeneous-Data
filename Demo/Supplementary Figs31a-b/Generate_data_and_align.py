import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import random
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import mean_squared_error, r2_score



# Set random seeds to ensure reproducible results (optional)
np.random.seed(42)

# Generate data
n_samples = 600
x = np.random.uniform(low=-0.5, high=0.5, size=n_samples)
y = np.random.uniform(low=-0.5, high=0.5, size=n_samples)
z = np.sin(x/20) + y**2  
mid = np.median(z)
labels = np.where(z > mid, 1, 0)
df = pd.DataFrame({
    'x': x,
    'y': y,
    'z': z,
    'label': labels
})



#Data visualization
xi = np.linspace(-0.5, 0.5, 100)
yi = np.linspace(-0.5, 0.5, 100)
Xi, Yi = np.meshgrid(xi, yi)
Zi = griddata((x, y), z, (Xi, Yi), method='cubic')
Labels_i = griddata((x, y), labels, (Xi, Yi), method='nearest')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
heatmap1 = ax1.pcolormesh(Xi, Yi, Zi, cmap='plasma', shading='auto')
plt.colorbar(heatmap1, ax=ax1, label='Z')
ax1.scatter(x, y, c='black', s=5, alpha=0.5)  
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_title('Z-value cloud map')
ax1.set_aspect('equal')

heatmap2 = ax2.pcolormesh(Xi, Yi, Labels_i, cmap='RdYlBu', shading='auto')
plt.colorbar(heatmap2, ax=ax2, label='Labels', ticks=[0, 1])
ax2.scatter(x, y, c='black', s=5, alpha=0.5) 
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_title('Labels cloud map (0/1 classification)')
ax2.set_aspect('equal')

plt.tight_layout()
plt.show()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
scatter1 = ax1.scatter(x, y, c=z, cmap='plasma', s=30)
plt.colorbar(scatter1, ax=ax1, label='Z')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_title('Z-value scatter plot')
ax1.set_aspect('equal')

scatter2 = ax2.scatter(x, y, c=labels, cmap='RdYlBu', s=30)
plt.colorbar(scatter2, ax=ax2, label='Labels', ticks=[0, 1])
ax2.set_xlabel('X')
ax1.set_ylabel('Y')
ax2.set_title('Labels scatter plot (0/1 classification)')
ax2.set_aspect('equal')

plt.tight_layout()
plt.show()


#################################################################################################
###############################align input and label#############################################
#################################################################################################
seted_parameter = {
                   'setted_loss_and_accuracy_file_name':'Loss_and_MSE_input_and_output',
                   'setted_start_cycle_number':-8,'setted_end_cycle_number':8,'seted_sample_number/2':250,
                  }

x = df.iloc[:, [0, 1]].apply(pd.to_numeric, errors='coerce')  
y = df.iloc[:, 2].apply(pd.to_numeric, errors='coerce')       
X_array = x.values.astype(np.float32)
y_array = y.values.astype(np.float32)

cyclic_array = [i for i in range(seted_parameter['setted_start_cycle_number'],seted_parameter['setted_end_cycle_number']+1)]
mse_Tr = np.zeros(len(cyclic_array))
r2_Tr = np.zeros(len(cyclic_array))
mse_Te = np.zeros(len(cyclic_array))
r2_Te = np.zeros(len(cyclic_array))
cycle_index = 0

for cyclic_number in cyclic_array:
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
    y_pred_Te = svm_regressor.predict(X_test_scaled)
    mse_te = mean_squared_error(y_test, y_pred_Te)
    r2_te = r2_score(y_test, y_pred_Te)
    mse_Tr[cycle_index] = mse_tr
    r2_Tr[cycle_index] = r2_tr
    mse_Te[cycle_index] = mse_te
    r2_Te[cycle_index] = r2_te

    cycle_index = cycle_index+1   


plt.figure(figsize=(10, 6))
plt.plot(cyclic_array, mse_Te)
plt.xlabel('Sample shift')
plt.ylabel('Test MSE')
plt.title('Sample shift-Test MSE')
plt.grid(True)
plt.show()