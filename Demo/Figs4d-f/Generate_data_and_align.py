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
                   'setted_loss_and_accuracy_file_name':'Loss_and_accuracy_alignment1',
                   'setted_start_cycle_number':-8,'setted_end_cycle_number':8,'seted_sample_number/2':250,
                  }

x = df.iloc[:, [0, 1]].apply(pd.to_numeric, errors='coerce')  
y = df.iloc[:, 3].apply(pd.to_numeric, errors='coerce')       
X_array = x.values.astype(np.float32)
y_array = y.values.astype(np.float32)

#Define search
cyclic_array = [i for i in range(seted_parameter['setted_start_cycle_number'],seted_parameter['setted_end_cycle_number']+1)]
accuracy = np.zeros(len(cyclic_array))
cycle_index = 0

for cyclic_number in cyclic_array:
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

    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3,random_state=42,stratify=y_data)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    svm_classifier = SVC(kernel='rbf',C=1.0,gamma='scale',random_state=42)
    svm_classifier.fit(X_train_scaled, y_train)
    y_pred = svm_classifier.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)

    accuracy[cycle_index] = acc
    cycle_index = cycle_index+1            #Calculate the highest precision matrix for the next round of iteration and save it as an index


plt.figure(figsize=(10, 6))
plt.plot(cyclic_array, accuracy)
plt.xlabel('Sample shift')
plt.ylabel('Test accuracy')
plt.title('Sample shift-test accuracy curve')
plt.grid(True)
plt.show()