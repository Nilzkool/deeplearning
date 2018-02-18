
## Scrip to demonstrate L_layer_NN (Version 1.1) with a toy dataset for a binary class problem
# IMPORTANT: First download the library L_layer_NN 

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
#%% Standerdize data
X_max=np.max(X_train,axis=1,keepdims=True)
X_min=np.min(X_train,axis=1,keepdims=True)
X_train_std=(X_train-X_min)/(X_max-X_min)
X_test_std=(X_test-X_min)/(X_max-X_min)
n_x=X_train.shape[0]

#%% NN model

# Inputs to this model are as follows:
# (1) X_train_std (training matrix of shape (n_x,m) where n_x is the no of input features and m is no of examples)
# (2) Y_train (labels or ground truth of shape (1,m)), make sure labels are int values 
# (3) hidden_layer_dims ( a list of layer sizes of hidden layers e.g [3,1] means it is a 3 layer NN (L=3) with 3 units in 1st layer, 
#     1 unit in 2nd layer. The no of units in the final layer (n_k) is the no of classes k.The no of units in the 0th or input layer 
#     is n_x=X.shape[0]. n_x and n_k are deetermined automatically
# (4) no_itns (no of iterations)
# (5) batch size (a recommended size is a any power of 2)
# (6) learn_rate (the learning rate or step size of gradient descent)
# (7) act_hl (activation type in hidden layer, choices:'relu','sigmoid' or 'tanh'),
# (8) act_fl (activation type in final layer, choices:'relu','sigmoid' or 'tanh'),
# (7) regularization (It is either None meaning no regularization, {'L2': lambd} in case of L2 regularization or
#     {'dropout':[..keep_prob_l..]} where l is a hidden layer that starts at 1 to L-2 and where keep_prob is probability 
#     of keeping a unit in layer l. By default keep_prob is set for the final layer
# (8) optimization (choices are 0-Gradient descent (GD), 1-GD with momentum, 2-Adam optimization)
#     should be formatted as optimization=0 for GD (also default), {'momentum':beta1},  {'Adam':[beta1,beta2]}
#     Determine no of classes


# hidden layer dimensions
hidden_layer_dims=[3,2] # ideal config 


# Optimizations
optim=0
#optim={'momentum':0.9}
#optim={'adam':[0.9,0.999]}

# regularizations
regu_para=None
#regu_para={'L2':1}
#regu_para={'dropout':[1,0.8]}

final_weights,_=L_layer_NN.NN_model(X_train_std,Y_train,hidden_layer_dims,no_epochs=1000,
                                  batch_size=128,learn_rate=0.01,act_hl="relu", act_fl="sigmoid",
                                  regularization=regu_para, optimization=optim)
## Predictions and accuracy
if final_weights:
        pred_train=L_layer_NN.NN_model_predictions(final_weights,X_train_std,act_hl="relu", act_fl="sigmoid",regularization=regu_para)
        pred_test=L_layer_NN.NN_model_predictions(final_weights,X_test_std,act_hl="relu", act_fl="sigmoid",regularization=regu_para)
        print("Train accuracy (%) =", round(L_layer_NN.calc_accuracy(Y_train,pred_train),2))
        print("Validation accuracy (%) =", round(L_layer_NN.calc_accuracy(Y_test,pred_test),2))




#%% Gradient checking
# This is an optional checking of the back prop gradients against numerically computed gradients.
# Set regularization to None before running it
# regu_para=None        
#L_layer_NN.check_gradients(final_params,X_train_std,Y_train,act_hl="relu",act_fl="sigmoid",epsilon=10e-7,regularization=None)


