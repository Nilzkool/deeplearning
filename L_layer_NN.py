# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 10:56:24 2018

@author: u0113548
"""
# IMPORTANT: NN library for binary classification problems 
import numpy as np
import matplotlib.pyplot as plt
#%% A function to initialize parameters
def initialize_weights(layer_dims):
    # Description: A xavier He implementation of random initialization of weights
    # Input: layer_dims 
    # Output: a dict containg initial values of all parameters formatted for evry layer l as
    #         {...,{layerl:{'Wl':Wl, 'bl':bl},...}
    
    L=len(layer_dims) # all layers including input and output layers
    
    init_weights={}
    for l in range (1,L):
        multiplier=np.sqrt(2/layer_dims[l-1]) # He multiplier
        W_l=np.random.randn(layer_dims[l],layer_dims[l-1])*multiplier
        b_l=np.zeros((layer_dims[l],1))
        
        init_weights['layer'+str(l)]={'W'+str(l):W_l,'b'+str(l):b_l}
        
    return init_weights
#%% Activation function
    
def activation(z,act_name):
    # Description: A function to compute activations for any layer l
    # Input: z (the linear feedforward part or np.dot(Wl,A_prev)+b_l), 
    #         act_name ('relu', 'sigmoid' or 'tanh')
    # Output: activations Al 
    
    if act_name=="relu":
       z_copy=z.copy()
       z_copy[z_copy<0]=0
       g=z_copy
       
    elif act_name=="sigmoid":
       g=1/(1+np.exp(-z))
       
    elif act_name=="tanh":
       g=np.tanh(z)
       
    return g
        
def activation_prime(z,act_name):
    # Description: A function to compute activations for any layer l
    # Input: z (or zl cached during forward prop) ,
    #         act_name ('relu', 'sigmoid' or 'tanh')
    # Output: derivatives of the activation functions
    if act_name=="relu":
       z_copy=z.copy()
       z_copy[z_copy<=0]=0
       z_copy[z_copy>0]=1
       g_prime=z_copy 
       
    elif act_name=="sigmoid":
       sigmoid_z=activation(z,"sigmoid")
       g_prime=sigmoid_z*(1-sigmoid_z)
       
    elif act_name=="tanh":
       tanh_z=activation(z,"tanh")
       g_prime=1-np.square(tanh_z)

    return g_prime
#%% Forward propagation
def forward_prop(weights,X_inp,act_hl, act_fl,regularization):
    #Descrption: Implementation of forward propagation
    #Input: weights (refer the format in the function initialize_weights), 
    #       X_inp (refer function NN_model),
    #       act_hl (refer function NN_model)
    #       act_fl (refer function NN_model), 
    #       regularization (refer function NN_model)
    
    #Output:a nested dictionary of cached values for each layer l and final activation value
    #       formatted for every layer l as {..,{layerl:{'A':A_prev, 'Wl': Wl, 'bl': bl, 'Zl':Zl},..}}
    #       where A_prev are the activations of the previous layer, Wl, bl and Zl are layer weights and linear
    #       forward parameters
    
    if regularization is not None:
        if list(regularization.keys())[0]=="dropout":
            regu_type="dropout"
            keep_probs=regularization[regu_type]
            
        elif list(regularization.keys())[0]=="L2":
            regu_type="L2"
    else:
        regu_type=None
    
    L=len(weights)+1 # all layers including input and output layers
    
    cache={}
    A_prev=X_inp # Activation in 0th layer
    act=act_hl
    for l in range(1,L):
        if l==L-1:
            act=act_fl
             
        weights_l=weights['layer'+str(l)]
        W_l=weights_l['W'+str(l)]
        b_l=weights_l['b'+str(l)]
        Z_l=np.dot(W_l,A_prev)+b_l
        cache['layer'+str(l)]={'A'+str(l-1): A_prev, 'W'+str(l):W_l, 'b'+str(l):b_l, 'Z'+str(l):Z_l}
        
        A_prev=activation(Z_l,act)
        
        if regu_type=="dropout" and l<=L-2:
            keep_prob=keep_probs[l-1]
            D_l=np.abs(np.random.randn(A_prev.shape[0],A_prev.shape[1]))
            D_l=D_l<keep_prob
            A_prev=A_prev*D_l
            A_prev=A_prev/keep_prob
            cache['layer'+str(l)]['D'+str(l)]=D_l
        
#    if A_prev==0:
#    elif:
#      A_prev==0
    cache['final activation']=A_prev
     
    return cache

#%% Calculate cost of function    
    
def log_loss(Y,Y_hat):
    #Descrption: Log loss
    #Input: Y (refer function NN model)
    #       Y_hat (final activations)
    #       
    #Output:log loss
    
    L=-(Y*np.log(Y_hat)+(1-Y)*np.log(1-Y_hat))
    return L

def calc_cost(Y,Y_hat):
    #Descrption: Cost calculation
    #Input: Y (refer function NN model)
    #       Y_hat (final activations)
    m=Y.shape[1]
    J=(1/m)*np.sum(log_loss(Y,Y_hat))
    return J

def calc_frobenius_norm_of_weights(weights,lambd,m):
    #Descrption: Calculate frobenius norm for L2 regularization
    #Input: weights (refer function intialize_weights), 
    #       lambd (L2 regularization),
    #       m (no of training examples)
    #       Y_hat (final activations)
    #Output: frobenius norm
    
    sq_sum=0
    L=len(weights)+1
    for l in range(1,L):   
        weights_l=weights['layer'+str(l)]
        W_l=weights_l['W'+str(l)]
        sq_sum=sq_sum+np.sum(np.square(W_l))
        
    frob=(lambd/(2*m))* sq_sum
    return frob
#%% Back propagation
def back_prop(Y,cache,act_hl, act_fl,regularization):
    #Description: Implementation of backward propagation
    #Input: Y (refer function NN_model), 
    #       cache (refer function forward_prop )
    #       act_hl (refer function NN_model)
    #       act_fl (refer function NN_model), 
    #       regularization (refer  function NN_model)
    #Output: a dictionary grads formatted as for every layer l
    #       {...,layerl:{'dWl':dWl, 'bd':bl},...}
    if regularization is not None:
        if list(regularization.keys())[0]=="dropout":
            regu_type="dropout"
            keep_probs=regularization[regu_type]
        elif list(regularization.keys())[0]=="L2":
            regu_type="L2"
            lambd=regularization[regu_type]
    else:
        regu_type=None
    
    A_fl=cache['final activation']
    m=Y.shape[1]
    L=len(cache) 
    dA_fl=-(np.divide(Y,A_fl)-np.divide(1-Y,1-A_fl))
    
    dA_l=dA_fl
    grads={}
    act=act_fl
    for l in reversed(range(1,L)):
        layer_cache=cache['layer'+str(l)]
        A_prev=layer_cache['A'+str(l-1)]
        W_l=layer_cache['W'+str(l)]
        b_l=layer_cache['b'+str(l)]
        Z_l=layer_cache['Z'+str(l)]
        
        if l<L-1:
            act=act_hl
            
        g_prime_l=activation_prime(Z_l, act)
        
        dZ_l=dA_l*g_prime_l
        dW_l=(1/m)*np.dot(dZ_l,A_prev.T)
        
        if regu_type=="L2":
            dW_l=dW_l+(lambd/m)*W_l
        
        db_l=(1/m)*np.sum(dZ_l,axis=1,keepdims=True)
        dA_l=np.dot(W_l.T,dZ_l)
        if regu_type=="dropout" and l>1:
            keep_prob=keep_probs[l-2]
            layer_cache_prev=cache['layer'+str(l-1)]
            D_l=layer_cache_prev['D'+str(l-1)]
            dA_l=dA_l*D_l
            dA_l=dA_l/keep_prob
        
        
        
        grads['layer'+str(l)]={'dW'+str(l):dW_l, 'db'+str(l):db_l}
    
    return grads

#%% Update parameters using grad descent
def update_parameters(weights,grads,learn_rate):
    #Description: Weight updation
    #Input: weights (refer format at function initialize_weights), 
    #       grads (refer format at function back_prop)
    #       learning rate or alpha
    #Output: updated weights in the same format as weights
    L=len(weights)+1
    for l in range(1,L):
        weights_l=weights['layer'+str(l)]
        grads_l=grads['layer'+str(l)]
        W_l=weights_l['W'+str(l)]
        b_l=weights_l['b'+str(l)]
        dW_l=grads_l['dW'+str(l)]
        db_l=grads_l['db'+str(l)]
        W_l=W_l-learn_rate*dW_l
        b_l=b_l-learn_rate*db_l
        weights['layer'+str(l)]={'W'+str(l):W_l,'b'+str(l):b_l}
    
    return weights


#%% Functions to convert dictionary of parameters into a column vector required for gradient checking
def unroll_into_column(params,identifier):
    #Description: A function to unroll the parameters in to a column vector required during gradient checking
    #Inputs: Dictionary of weights or gradients,
    #        identifier="weights" if params passed is weights 
    #Output: A column vector of weights
    
    L= len(params)+1 #Total layers
        
    if identifier =="weights":
        pre_str=""
    else:
        pre_str="d"
    
    col_vec=np.array([0]).reshape((1,1))
    
    for l in range(1,L):
        params_l=params['layer'+str(l)]
        W_l=params_l[pre_str+'W'+str(l)]
        b_l=params_l[pre_str+'b'+str(l)]
        W_l_col=W_l.reshape((W_l.shape[0]*W_l.shape[1],1))
        b_l_col=b_l.reshape((b_l.shape[0]*b_l.shape[1],1))
        temp_col_vec=np.append(W_l_col,b_l_col,axis=0)
        col_vec=np.append(col_vec,temp_col_vec,axis=0)
        
    col_vec=col_vec[1:,0]
    return col_vec.reshape((len(col_vec),1))
    
def roll_into_dict(theta_pm,weights_orig,identifier):
    #Description: A function to roll back weights vector to a dictionary required for gradient checking
    #Inputs: theta_pm (a column vector of weights),
    #        identifier="weights" if params passed is weights  (not required anymore)
    #Output: weights (refer format at the function intialize_weights)
    if identifier =="weights":
        pre_str=""
    else:
        pre_str="d"
    
    params_dict={}
    L= len(weights_orig)+1 #Total layers
    
    counter=0
    for l in range(1,L):
         weights_l=weights_orig['layer'+str(l)]   
         shape_W=weights_l['W'+str(l)].shape
         shape_b=weights_l['b'+str(l)].shape
         num_of_W_params=shape_W[0]*shape_W[1]
         num_of_b_params=shape_b[0]*shape_b[1]
         
         W_params=theta_pm[counter:counter+num_of_W_params]
         counter=counter+num_of_W_params
         
         b_params=theta_pm[counter:counter+num_of_b_params]
         counter=counter+num_of_b_params  
         
         W_params=W_params.reshape(shape_W)
         b_params=b_params.reshape(shape_b)
         params_dict['layer'+str(l)]={pre_str+'W'+str(l):W_params,pre_str+'b'+str(l):b_params }

         
    return params_dict
            
#%% A function to check the gradients numerically
def check_gradients(weights,X,Y,act_hl,act_fl,epsilon,regularization):
    #Description: A function to validate gradients from back prop against numerically computed gradients
    #Inputs: weights (refer format at function initialize_weights)
    #        X (refer function NN_model),
    #        Y (refer function NN_model),
    #        act_hl (refer function NN model)
    #        act_fl (refer function NN model)
    #        epsilon (step value, recommended 10e-7)
    #        regularization (refer function NN model)
    #Output: A string indicating if gradients are ok or not
           
    
    cache=forward_prop(weights,X,act_hl, act_fl,regularization)
    grads=back_prop(Y,cache,act_hl, act_fl,regularization)
    
    weights_col= unroll_into_column(weights,"weights")
    grads_col=unroll_into_column(grads,"gradients")
    
    n_params = weights_col.shape[0]
    #J_plus = np.zeros((n_params, 1))
    #J_minus = np.zeros((n_params, 1))
    gradapprox = np.zeros((n_params, 1))
    
    for i in range(0,n_params):
        # plus epsilon
        theta_plus=np.copy(weights_col)
        theta_plus[i,0]=theta_plus[i,0]+epsilon
        
        theta_plus_dict=roll_into_dict(theta_plus,weights,"weights")
        cache_plus=forward_prop( theta_plus_dict,X,act_hl, act_fl,regularization)
        #Cost calculation
        Y_hat=cache_plus['final activation']
        Jplus_i=calc_cost(Y,Y_hat)
        
        # minus epsilon
        theta_minus=np.copy(weights_col)
        theta_minus[i,0]=theta_minus[i,0]-epsilon
        theta_minus_dict=roll_into_dict(theta_minus,weights,"weights")
        cache_minus=forward_prop(theta_minus_dict,X,act_hl, act_fl,regularization)
        #Cost calculation
        Y_hat=cache_minus['final activation']
        Jminus_i=calc_cost(Y,Y_hat)
        
        gradapprox[i,0]= (Jplus_i-Jminus_i)/(2*epsilon)
        
    numerator = np.linalg.norm(grads_col-gradapprox)                               
    denominator = np.linalg.norm(grads_col)+ np.linalg.norm(gradapprox)                            
    rel_diff = numerator/denominator  
    
    if rel_diff<2e-7:
        print("Gradients are OK, relative difference =",str(rel_diff))
    else:
        print("Gradients are not OK, relative difference =",str(rel_diff))
        
#%% Final model
def NN_model(X,Y,layer_dims,no_itns,learn_rate,act_hl, act_fl,regularization=None):
    #Descrption: A function to optimize the weights of the NN
    #Input: X (training matrix of shape (n_x,m) where n_x is the no of input features and m is no of examples) ,
    #       Y (labels or ground truth of shape (1,m)) currently binary claasification only,
    #       layer_dims ( a list of layer sizes e.g [3,4,3,1] means it is a 3 layer NN
    #                   with 3 units in 0th layer (input layer), 4 units in 1st layer
    #                   3 units in 2nd layer and 1 unit in 3rd and final layer),
    #       act_hl (activation type in hidden layer, choices:'relu','sigmoid' or 'tanh'),
    #       act_fl (activation type in final layer, choices:'relu','sigmoid' or 'tanh'),
    #       regularization (It is either None meaning no regularization, 
    #                         {'L2': lambd} in case of L2 regularization or
    #                        {'dropout':[..keep_prob_l..]} where l is a hidden layer that starts at 1 to L-2
    #                        and where keep_prob is probabily of keeping the unit, by default keep_prob is set to 1
    #                        for the final layer
    if regularization==None:
        regu_type=None
    else:
        if list(regularization.keys())[0]=="L2":
            regu_type="L2"
            lambd=regularization[regu_type]

        elif list(regularization.keys())[0]=="dropout":
            regu_type="dropout"
            keep_probs=regularization["dropout"]
            if len(keep_probs)>len(layer_dims)-2:
                print("Check dimension of keep_probs list")
                return False

    
    itn_no=0
    weights=initialize_weights(layer_dims)
    
    cost_list=[]
    itn_list=[]
    while itn_no<=no_itns:
        
        #Forward prop
        cache=forward_prop(weights,X,act_hl, act_fl,regularization)
        
        #Cost calculation
        Y_hat=cache['final activation']
        cost=calc_cost(Y,Y_hat)
        
        if regu_type=="L2":
            cost=cost+calc_frobenius_norm_of_weights(weights,lambd,m=X.shape[1])
        
        #print(cost)
        if itn_no % 100 == 0:
            print("Cost after iteration {}: {}".format(itn_no, round(cost,3)))
            cost_list.append(cost)
            itn_list.append(itn_no // 100)
        
        #Back prop
        grads=back_prop(Y,cache,act_hl, act_fl,regularization)
        
        #Update params
        weights=update_parameters(weights,grads,learn_rate)
        itn_no+=1
        
    if regu_type!="dropout":    
        plt.figure(figsize=(5,5), dpi= 100, facecolor='w', edgecolor='k')    
        #iterations=list(range(0,len(cost_list)))    
            
        plt.plot(itn_list,cost_list)
        plt.ylabel("J")
        plt.xlabel("Iterations (x 100)")
    return weights


#%% Predict
def NN_model_predictions(weights,X,act_hl, act_fl,regularization):
    #Descrption: A function calculate output or predictions
    #Input: X (refer function NN_model),
    #       act_hl (refer function NN_model),
    #       act_fl (refer function NN_model),
    #       regularization(refer function NN_model)
    #Output: Prediction matrix Y_hat (shape (1,m))
    regularization=None
    cache=forward_prop(weights,X,act_hl, act_fl,regularization)
    Y_hat=cache['final activation']
    Y_hat[Y_hat>0.5]=1
    Y_hat[Y_hat<=0.5]=0
    return Y_hat
#%% Calculate accuracy of predictions
def calc_accuracy(Y,Y_hat):
    #Descrption: A function calculate accuracy of predictions
    #Input: Y (refer function NN_model)
    #       Y_hat (refer function NN_model_predictions)
    #Output: accuracy (%)
    m=Y.shape[1]
    accuracy=100*(1/m)*np.sum(Y==Y_hat)
    return accuracy
    
 
    
