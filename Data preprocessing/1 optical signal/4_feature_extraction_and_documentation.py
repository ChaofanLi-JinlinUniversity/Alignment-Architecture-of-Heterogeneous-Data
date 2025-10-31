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



def sort_by_number_in_filename(filename):                          #Custom sorting function
    match_numbers = re.findall(r'\d+', os.path.basename(filename))
    match_numbers = [int(num) for num in match_numbers]
    #print(match_numbers[0])
    return match_numbers[1]


TIF = pathlib.Path('3-3/cutted_tif')
tif_list = [str(path) for path in sorted(TIF.glob('*.tif'),key=sort_by_number_in_filename)]
#print(tif_list)




import numpy as np
from PIL import Image
from scipy.stats import skew, kurtosis

def calculate_entropy(image_array):
    """Calculate the entropy features of an image"""
    # Calculate the histogram and normalize it to probability
    hist, _ = np.histogram(image_array, bins=256, range=(0, 256))
    prob = hist / hist.sum()
    
    # Calculate entropy (ignoring zero probability values)
    entropy = -np.sum(prob[prob > 0] * np.log2(prob[prob > 0]))
    return entropy

def extract_image_features(image_path):
    """Extract various features of images"""
    # Read TIFF images
    img = Image.open(image_path)
    
    # Convert to NumPy array (grayscale image)
    if img.mode == 'RGB':
        img = img.convert('L')  # Convert to grayscale image
    img_array = np.array(img)
    
    # Flatten image data into a one-dimensional array
    flat_array = img_array.ravel()
    
    # Calculate basic statistical characteristics
    mean_val = np.mean(flat_array)
    max_val = np.max(flat_array)
    min_val = np.min(flat_array)
    range_val = max_val - min_val  # range
    variance_val = np.var(flat_array)
    
    # Calculate skewness and kurtosis
    skewness_val = skew(flat_array)
    kurtosis_val = kurtosis(flat_array, fisher=True)  
    
    # Calculate entropy features
    entropy_val = calculate_entropy(flat_array)
    
    # Return all features
    return {
        'mean': mean_val,
        'max': max_val,
        'min': min_val,
        'range': range_val,
        'variance': variance_val,
        'skewness': skewness_val,
        'kurtosis': kurtosis_val,
        'entropy': entropy_val
    }



def batch_export_features(pathlist):
	output_data = np.zeros((len(pathlist),8))
	i = 0
	for ele in pathlist:
		features = extract_image_features(ele)
		output_data[i,0] = features['mean']
		output_data[i,1] = features['max']
		output_data[i,2] = features['min']
		output_data[i,3] = features['range']
		output_data[i,4] = features['variance']
		output_data[i,5] = features['skewness']
		output_data[i,6] = features['kurtosis']
		output_data[i,7] = features['entropy']
		print(i,f'finish')
		i = i+1
	headers = ['mean', 'max', 'min', 'range', 'variance', 'skewness', 'kurtosis', 'entropy']
	df = pd.DataFrame(output_data, columns=headers)
	df.to_excel('Image_feature_and_label_alignment____image_features.xlsx', index=False)



batch_export_features(tif_list)