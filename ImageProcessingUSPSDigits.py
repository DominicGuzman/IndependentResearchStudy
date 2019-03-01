
import scipy.io as sio
import os
import pandas as pd
import numpy as np
import random as rand


os.chdir('C:/Users/Dominic Guzman/Desktop')
df = sio.loadmat('usps_resampled')
train_cases = df['train_patterns']
train_cases_pd = pd.DataFrame(train_cases)
train_labels = df['train_labels']
train_labels_pd = pd.DataFrame(train_labels)

test_cases = df['test_patterns']
test_cases_pd = pd.DataFrame(test_cases)
test_labels = df['test_labels']
test_labels_pd = pd.DataFrame(test_labels)

import matplotlib.pyplot as plt    

def ImageProcessing_means(integer):
    train_values = np.where(train_labels_pd.T[:][integer]==1)
    train_indices = train_values[0]
    elements_array = []
    for i in range(0, len(train_indices)):
        int_val = train_indices[i]
        train_array = train_cases_pd.T[int_val:int_val+1].iloc[0]
        elements_array.append(train_array)
    means_array = []
    for j in range(0, len(elements_array)):
        mean = sum(elements_array[j])/len(elements_array[j])
        means_array.append(mean)
    return (sum(means_array)/len(means_array))




        
    
    
