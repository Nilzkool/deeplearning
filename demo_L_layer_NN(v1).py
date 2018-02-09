# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 14:09:01 2018

@author: u0113548
"""

## Scrip to demonstrate L_layer_NN (Version 1.0) with a toy dataset
# IMPORTANT: binary classification problems only
# First download the library L_layer_NN 


#%% Import libraries
import numpy as np

import os # to change to your current working directory
cur_dir=os.getcwd()
os.chdir(cur_dir)
import sys
lib_dest='C:/Users/u0113548/Google Drive/Deeplearning course/Scripts' # This is the folder where you save L_layer_NN 
sys.path.insert(0,lib_dest)
import L_layer_NN # Import the library
from sklearn.model_selection import train_test_split 
#%% Load breast cancer dataset (out toy dataset)

# Breast cancer dataset
from sklearn.datasets import load_breast_cancer 
data = load_breast_cancer()
X_orig=data['data']
Y_orig=data['target']


#%% Split train and test

X_train, X_test, Y_train, Y_test = train_test_split(X_orig, Y_orig, test_size=0.2, random_state=1)


X_train=X_train.T
Y_train=Y_train.reshape(1,len(Y_train))
X_test=X_test.T
Y_test=Y_test.reshape(1,len(Y_test))


#%% Standerdize
X_max=np.max(X_train,axis=1,keepdims=True)
X_min=np.min(X_train,axis=1,keepdims=True)


X_train_std=(X_train-X_min)/(X_max-X_min)
X_test_std=(X_test-X_min)/(X_max-X_min)

n_x=X_train.shape[0]

#%% NN model

## The inputs to the NN are described as follows:
# (1) X_train_std (training matrix of shape (n_x,m) where n_x is the no of input features and m is no of examples)
# (2) Y_train (labels or ground truth of shape (1,m)) currently binary claasification only
# (3) layer_dims ( a list of layer sizes e.g [3,4,3,1] meaning it is a 3 layer NN with 3 units orfeatures in 0th layer 
#     (input layer), 4 units in 1st layer, 3 units in 2nd layer and 1 unit in 3rd and final layer
# (4) no_itns (no of iterations)
# (5) learn_rate (the learning rate or step size of gradient descent)
# (5) act_hl (activation type in hidden layer, choices:'relu','sigmoid' or 'tanh'),
# (6) act_fl (activation type in final layer, choices:'relu','sigmoid' or 'tanh'),
# (7) regularization (It is either None meaning no regularization, {'L2': lambd} in case of L2 regularization or
#     {'dropout':[..keep_prob_l..]} where l is a hidden layer that starts at 1 to L-2 and where keep_prob is probabily 
#     of keeping a unit in layer l. By default keep_prob is set for the final layer



layer_dims=[n_x,3,2,1] # ideal config 
regu_para=None
#regu_para={'L2':1}
#regu_para={'dropout':[0.7,0.8]}
final_weights=L_layer_NN.NN_model(X_train_std,Y_train,layer_dims,no_itns=1000,learn_rate=0.1,act_hl="relu", act_fl="sigmoid",regularization=regu_para)

## Predictions and accuracy
if final_weights:
        pred_train=L_layer_NN.NN_model_predictions(final_weights,X_train_std,act_hl="relu", act_fl="sigmoid",regularization=regu_para)
        pred_test=L_layer_NN.NN_model_predictions(final_weights,X_test_std,act_hl="relu", act_fl="sigmoid",regularization=regu_para)
        print("Train accuracy (%) =", round(L_layer_NN.calc_accuracy(Y_train,pred_train),2))
        print("Validation accuracy (%) =", round(L_layer_NN.calc_accuracy(Y_test,pred_test),2))


#regu_para=None

#%% Gradient checking
# This is an optional checking of the back prop gradients against numerically computed gradients.
# Set regularization to None before running it
#L_layer_NN.check_gradients(final_params,X_train_std,Y_train,act_hl="relu",act_fl="sigmoid",epsilon=10e-7,regularization=None)


