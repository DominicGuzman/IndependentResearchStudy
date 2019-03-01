# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 17:08:43 2018

@author: Dominic Guzman
"""
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd

def LogisticRegression():
    from sklearn import datasets, linear_model
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.model_selection import train_test_split
    digits = datasets.load_breast_cancer()
    X_cancer = digits.data
    y_cancer = digits.target
    
    
    
    
    
    logistic = linear_model.LogisticRegression()
    X_train, X_test, Y_train, Y_test = train_test_split(X_cancer, y_cancer, test_size = 0.33, random_state = 5)
    print('LogisticRegression_digit score: %f'% logistic.fit(X_train, Y_train).score(X_test, Y_test))
    
    
    X_reduced = PCA(n_components=3).fit_transform(X_cancer)
    fig = plt.figure(1, figsize=(8, 6))
    ax = Axes3D(fig, elev=-150, azim=110)
    ax.scatter(X_reduced[:, 0],X_reduced[:, 1], X_reduced[:,2], c=y_cancer, cmap=plt.cm.Set1, edgecolor='k', s=40)
    ax.set_xlabel("1st eigenvector")
    ax.w_xaxis.set_ticklabels([])
    ax.set_ylabel("2nd eigenvector")
    ax.w_yaxis.set_ticklabels([])
    ax.set_zlabel("3rd eigenvector")
    ax.w_zaxis.set_ticklabels([])
    
    plt.show()
    


LogisticRegression()   

def SoftmaxRegression_flowers():
    from sklearn import datasets, linear_model
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.model_selection import train_test_split
    irises = datasets.load_iris()
    
    X_irises = irises.data
    y_irises = irises.target
    
    X_train, X_test, Y_train, Y_test = train_test_split(X_irises, y_irises, test_size = 0.33, random_state = 5)
    
    logistic = linear_model.LogisticRegression()
    print('LogisticRegression_iris score: %f'% logistic.fit(X_train, Y_train).score(X_test, Y_test))
    
    
    X_reduced = PCA(n_components=3).fit_transform(X_irises)
    
    fig = plt.figure(1, figsize=(8, 6))
    ax = Axes3D(fig, elev=-150, azim=110)
    ax.scatter(X_reduced[:, 0],X_reduced[:, 1], X_reduced[:,2], c=y_irises, cmap=plt.cm.Set1, edgecolor='k', s=40)
    ax.set_xlabel("1st eigenvector")
    ax.w_xaxis.set_ticklabels([])
    ax.set_ylabel("2nd eigenvector")
    ax.w_yaxis.set_ticklabels([])
    ax.set_zlabel("3rd eigenvector")
    ax.w_zaxis.set_ticklabels([])
    
    plt.show()
    
    
SoftmaxRegression_flowers()


def LinearRegression_boston():
    from sklearn import datasets, linear_model
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from mpl_toolkits.mplot3d import Axes3D
    from sklearn.metrics import mean_squared_error
    prices = datasets.load_boston()
    
    X_prices = prices.data
    y_prices = prices.target
    
    
    X_train, X_test, Y_train, Y_test = train_test_split(X_prices, y_prices, test_size = 0.33, random_state = 5)
    
    
    
    linear = linear_model.LinearRegression()
    
    print('LinearRegression_boston score: %f'% linear.fit(X_train, Y_train).score(X_test, Y_test))

    
    linear.fit(X_train,Y_train)
    y_pred = linear.predict(X_test)
    print("LinearRegression mean_squared_error: %.2f"% mean_squared_error(Y_test, y_pred))
    
    
    
    #PCA
    
    X_reduced = PCA(n_components=3).fit_transform(X_prices)
    
    fig = plt.figure(1, figsize=(8, 6))
    ax = Axes3D(fig, elev=-150, azim=110)
    ax.scatter(X_reduced[:, 0],X_reduced[:, 1], X_reduced[:,2], c=y_prices, cmap=plt.cm.Set1, edgecolor='k', s=40)
    ax.set_xlabel("1st eigenvector")
    ax.w_xaxis.set_ticklabels([])
    ax.set_ylabel("2nd eigenvector")
    ax.w_yaxis.set_ticklabels([])
    ax.set_zlabel("3rd eigenvector")
    ax.w_zaxis.set_ticklabels([])
    
    plt.show()
    
LinearRegression_boston()    
    
    
