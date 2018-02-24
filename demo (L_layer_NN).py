# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 14:09:01 2018

@author: u0113548
"""

## Scrip to demonstrate L_layer_NN (Version 1.2) with a toy dataset for multiclass problem
# IMPORTANT: First download the library L_layer_NN 

#%% Import libraries
import os # to change to your current working directory
cur_dir=os.getcwd()
os.chdir(cur_dir)
import sys
lib_dest='C:/Users/u0113548/Google Drive/Deeplearning course/Scripts' # This is the folder where you save L_layer_NN 
sys.path.insert(0,lib_dest)
import L_layer_NN # Import the library
from sklearn.model_selection import train_test_split 

#%% Import digits dataset available on sklearn.datasets (our toy dataset)

from sklearn.datasets import load_digits 
data = load_digits()
X_orig=data['data']
Y_orig=data['target']

# Visualize
#import matplotlib.pyplot as plt 
#plt.gray() 
#plt.matshow(data.images[9]) 
#plt.show() 

#%% Split data into train and test
X_train, X_test, Y_train, Y_test = train_test_split(X_orig, Y_orig, test_size=0.1,stratify=Y_orig,random_state=1)
X_train=X_train.T
Y_train=Y_train.reshape(1,len(Y_train))
X_test=X_test.T
Y_test=Y_test.reshape(1,len(Y_test))

#%% Standerdize
X_train_std=X_train/16
X_test_std=X_test/16

#%% NN model

## Set hidden layer dimensions
hidden_layer_dims=[10,10] 

## Set no of epochs
no_of_epochs=1000

## Set learn rate
alpha=0.01

## Set Optimizations type
optim=0  #plain vanilla gradient descent
#optim={'momentum':0.9}
#optim={'adam':[0.9,0.999]}

## Set Regularizations
regu_para=None
#regu_para={'L2':0.1}
#regu_para={'dropout':[1,0.8]} # probs are fo for hidden layers only

## Set Mini batch size
mini_batch_size=512


## Set activation for hidden layer
type_hl='relu'
#activation_fl='tanh'
#activation_fl='sigmoid'

## Set activation for final layer
type_fl='softmax'
#activation_fl='sigmoid'
#activation_fl='tanh'

## Turn on batch norm
batch_norm_flag= True


## Train the NN
final_weights,_=L_layer_NN.NN_model(X_train_std,Y_train,hidden_layer_dims,no_epochs=no_of_epochs,
                                  batch_size=mini_batch_size,learn_rate=alpha,act_hl=type_hl, act_fl=type_fl,
                                  regularization=regu_para, optimization=optim, batch_norm=batch_norm_flag)
## Predictions and accuracy
if final_weights:
    pred_train=L_layer_NN.NN_model_predictions(final_weights,X_train_std,act_hl=type_hl, act_fl=type_fl,regularization=regu_para,batch_norm=batch_norm_flag)
    pred_test=L_layer_NN.NN_model_predictions(final_weights,X_test_std,act_hl=type_hl, act_fl=type_fl,regularization=regu_para,batch_norm=batch_norm_flag)
    print("Train accuracy (%) =", round(L_layer_NN.calc_accuracy(Y_train,pred_train),2))
    print("Validation accuracy (%) =", round(L_layer_NN.calc_accuracy(Y_test,pred_test),2))



#%% Gradient checking
# This is an optional checking of the back prop gradients against numerically computed gradients.
# Set regularization to None before running it
L_layer_NN.check_gradients(final_weights,X_train_std,Y_train,act_hl=type_hl,act_fl=type_fl,epsilon=10e-7,regularization=None,batch_norm=batch_norm_flag)


