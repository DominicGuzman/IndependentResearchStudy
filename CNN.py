# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 16:34:22 2018

@author: Dominic Guzman
"""

import pandas as pd
import scipy.io as sio
df = sio.loadmat('usps_resampled.mat', squeeze_me = True)
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Reshape, Flatten, Dense
from keras.utils import to_categorical


def CNN():
    from scipy import array
    from sklearn.model_selection import train_test_split
    train_set = df['train_patterns'].T
    train_set_pd = pd.DataFrame(train_set)
    
    test_set = df['test_patterns'].T
    test_set_pd = pd.DataFrame(test_set)
    
    train_array = train_set_pd
    test_array = test_set_pd
    
    train_labels = df['train_labels'].T
    test_labels = df['test_labels'].T
    
    
    x_train_set = train_array.iloc[:,:]
    y_train_set = array([list(x).index(1) for x in train_labels])
    
    x_test_set = test_array.iloc[:,:]
    y_test_set = array([list(x).index(1) for x in test_labels])

    y_train_set = to_categorical(y_train_set)
    y_test_set = to_categorical(y_test_set)
    
    model = Sequential()
    
    model.add(Reshape((16,16,1), input_shape=(256,)))
    model.add(Conv2D(64,(3,3),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(.25))
    model.add(Conv2D(256,(3,3),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(.25))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    
    X_train, X_test, Y_train, Y_test = train_test_split(x_train_set, y_train_set, test_size = 0.33, random_state = 5)

    
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    model.fit(X_train,Y_train,epochs=10,validation_data=(X_test,Y_test))
    score = model.evaluate(x_test_set, y_test_set, verbose=0)
    print(score)

    
CNN()