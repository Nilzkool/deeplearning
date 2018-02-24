# NN library for binary classification problems 
import numpy as np
import matplotlib.pyplot as plt
#%% A function to initialize parameters
def initialize_weights(layer_dims,batch_norm=False):
    # Description: A xavier implementation of random initialization of weights
    # Input: (1) layer_dims (dimensions of each layer of NN)
    # Output: a dict containg initial values of all parameters formatted for evry layer l as
    #         {...,{layerl:{'Wl':Wl, 'bl':bl},...}
    
    L=len(layer_dims) # all layers of NN including input and output layers
    
    init_weights={}
    
    for l in range (1,L):
        multiplier=np.sqrt(2/layer_dims[l-1]) # He multiplier
        W_l=np.random.randn(layer_dims[l],layer_dims[l-1])*multiplier
        #W_l=np.ones((layer_dims[l],layer_dims[l-1]))
        b_l=np.zeros((layer_dims[l],1))
        
        init_weights['layer'+str(l)]={'W'+str(l):W_l,'b'+str(l):b_l}
        
        if batch_norm:
            init_weights_dict=init_weights['layer'+str(l)]
            init_weights_dict['gamma'+str(l)]=np.ones((layer_dims[l],1))
            init_weights_dict['beta'+str(l)]=np.zeros((layer_dims[l],1))
            init_weights_dict['exp_avg_sigma'+str(l)]=np.zeros((layer_dims[l],1)) #exponentially averaged params
            init_weights_dict['exp_avg_mu'+str(l)]=np.zeros((layer_dims[l],1))
            
        
    return init_weights
#%% Activation function
    
def activation(z,act_name):
    # Description: A function to compute activations for any layer l
    # Input: (1) z (the linear feedforward part or np.dot(Wl,A_prev)+b_l), 
    #        (2) act_name (activation type, choices are "relu", "sigmoid" or "tanh")
    # Output: activations Al in layer l 
    
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
    # Description: A function to compute derivative of activation functions for any layer l
    # Input: (1) z (or zl cached during forward prop)
    #        (2) act_name (activation type, choices are "relu", "sigmoid" or "tanh")
    # Output: g_prime (derivatives of the activation function) 
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
def batch_norm_forward(Z_l,weights_l,l,pred_flag=False):
    #Description:  Implementation of batch norm forward pass at a layer l
    #Inputs: (1) Z_l (linear feedforward part of layer l)
    #        (2) weights_l (weights dict for layer l)
    #        (3) l (layer no, int)
    #        (4) pred_flag (to indicate if prediction functions are called, stored moving averages of mu and 
    #            std dev of Z_l is used for normalization)
    #Output: Zmu_l (mean of Z_l), Sstd_l (std dev of Z_l), gamma_l, beta_l, Znorm_l (normalized Z), 
    #        Ztilda_l (tuned Znorm_l)
    if pred_flag:
        Zmu_l=weights_l['exp_avg_mu'+str(l)]
        Zstd_l=weights_l['exp_avg_sigma'+str(l)]
    else:
        Zmu_l=np.mean(Z_l,axis=1,keepdims=True)
        Zstd_l=np.std(Z_l,axis=1,keepdims=True)
    
    gamma_l=weights_l['gamma'+str(l)]
    beta_l=weights_l['beta'+str(l)]
    Znorm_l=(Z_l-Zmu_l)/(Zstd_l+10e-8)
    Ztilda_l=gamma_l*Znorm_l+beta_l
    
    return Zmu_l,Zstd_l,gamma_l,beta_l,Znorm_l,Ztilda_l


def forward_prop(weights,X,act_hl, act_fl,regularization, batch_norm=False,pred_flag=False):
    #Descrption: Implementation of forward propagation
    #Input: (1) weights (refer the format in the function initialize_weights), 
    #       (2) X (taining matrix X or a mini batch of training matrix X),
    #       (3) act_hl (hidden layer activation type, refer function NN_model)
    #       (4) act_fl (final layer activation, refer function NN_model), 
    #       (5) regularization (refer function NN_model)
    #       (6) batch_norm (flag to indicate if batch normalization is True)
    #       (7) pred_flag (flag to indicate if prediction functions are called(to handle dropouts and batch norms))
    #Output:a nested dictionary called "cache", which contains cached values for each layer l and final activation value.
    #       It is formatted as every layer l as {..,{layerl:{'A':A_prev, 'Wl': Wl, 'bl': bl, 'Zl':Zl},..'final_activation':AL}}
    #       where A_prev are the activations of the previous layer, Wl, bl and Zl are layer weights and linear
    #       forward parameters. It also contains the final activation value AL.
    
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
    A_prev=X # Activation in 0th layer
    act=act_hl
    for l in range(1,L):
        if l==L-1:
            act=act_fl
             
        weights_l=weights['layer'+str(l)]
        W_l=weights_l['W'+str(l)]
        b_l=weights_l['b'+str(l)]
        Z_l=np.dot(W_l,A_prev)+b_l
        if batch_norm:
            Zmu_l,Zstd_l,gamma_l,beta_l,Znorm_l,Ztilda_l=batch_norm_forward(Z_l,weights_l,l,pred_flag)
            cache['layer'+str(l)]={'A'+str(l-1): A_prev, 'W'+str(l):W_l, 'b'+str(l):b_l, 
                                  'Z'+str(l):Z_l, 'gamma'+str(l):gamma_l, 'beta'+str(l):beta_l,
                                   'Zmu'+str(l): Zmu_l, 'Zstdev'+str(l): Zstd_l, 'Znorm'+str(l): Znorm_l,
                                   'Ztilda'+str(l):Ztilda_l}
            Z_l=Ztilda_l
            
            if 'exp_avg_mu'+str(l) in weights_l.keys():
                weights_l['exp_avg_mu'+str(l)]= 0.9*weights_l['exp_avg_mu'+str(l)]+0.1*Zmu_l 
                weights_l['exp_avg_sigma'+str(l)]=0.9*weights_l['exp_avg_sigma'+str(l)]+0.1*Zstd_l
            
        else:
            cache['layer'+str(l)]={'A'+str(l-1): A_prev, 'W'+str(l):W_l, 'b'+str(l):b_l, 'Z'+str(l):Z_l}
        
        if act=="softmax":
           t=np.exp(Z_l)
           A_prev=t/np.sum(t,axis=0,keepdims=True)
        else:
            A_prev=activation(Z_l,act)
        
        if regu_type=="dropout" and l<=L-2:
            if not pred_flag: # apply dropout only during training
                keep_prob=keep_probs[l-1]
                D_l=np.abs(np.random.randn(A_prev.shape[0],A_prev.shape[1]))
                D_l=D_l<keep_prob
                A_prev=A_prev*D_l
                A_prev=A_prev/keep_prob
                cache['layer'+str(l)]['D'+str(l)]=D_l
        

    cache['final activation']=A_prev
     
    return cache

#%% Calculate cost of function    
    
def log_loss(Y,Y_hat,act_fl):
    #Descrption: Log loss
    #Input: (1) Y (refer function NN model)
    #       (2) Y_hat (final activations)
    #       (3) Final layer activation
    #Output: log loss
    
    if act_fl=="softmax": 
        L=-np.sum((Y*np.log(Y_hat+10e-8)), axis=0, keepdims=True)
    else: # one vs rest logistic regression
        L=-(Y*np.log(Y_hat+10e-8)+(1-Y)*np.log(1-Y_hat+10e-8))
    return L

def calc_cost(Y,Y_hat,act_fl):
    #Descrption: Cost calculation
    #Input: (1) Y (refer function NN model)
    #       (2) Y_hat (final activations)
    #       (3) act_fl (refer func NN_model)
    #Output: Cost
    m=Y.shape[1]
    J=(1/(m))*np.sum(log_loss(Y,Y_hat,act_fl))
    return J

def calc_frobenius_norm_of_weights(weights,lambd,m):
    #Descrption: Calculate frobenius norm for L2 regularization
    #Input: (1) weights (refer function intialize_weights), 
    #       (2) lambd (L2 regularization),
    #       (3) m (no of training examples)
    #       (4) Y_hat (final activations)
    #Output: frobenius (L2) norm of the weights
    
    sq_sum=0
    L=len(weights)+1
    for l in range(1,L):   
        weights_l=weights['layer'+str(l)]
        W_l=weights_l['W'+str(l)]
        sq_sum=sq_sum+np.sum(np.square(W_l))
        
    frob=(lambd/(2*m))* sq_sum
    return frob

#%% Back propagation
def get_batchnorm_grads(dZtilda,layer_cache,layer,m):
    #Description:  Implementation of batch norm backward pass at a layer l
    #Inputs:(1) dZtilda (upstream derivative of the cost function)
    #       (2) layer_cache (cahce values for layer l)
    #       (3) layer (layer no, int)
    #       (4) m (no of examples)
    #Outputs: gradients dgamma, dbeta, dZ 
    Znorm=layer_cache['Znorm'+str(layer)]
    sigma=layer_cache['Zstdev'+str(layer)]
    gamma=layer_cache['gamma'+str(layer)]
    mu=layer_cache['Zmu'+str(layer)]
    Z=layer_cache['Z'+str(layer)]
    
    dgamma=np.sum(dZtilda*Znorm, axis=1, keepdims=True)
    dbeta=np.sum(dZtilda, axis=1, keepdims=True)
    
    dZnorm=dZtilda*gamma
    dsigma_sqr=np.sum(dZnorm*(Z-mu), axis=1, keepdims=True)*-0.5*(sigma**2+1e-8)**(-3/2)
    dmu=np.sum(dZnorm, axis=1, keepdims=True)*-(sigma**2+10e-8)**(-0.5)+dsigma_sqr*-2*np.mean(Z-mu,axis=1, keepdims=True)
    dZ=dZnorm*(sigma**2+10e-8)**(-0.5)+ dsigma_sqr*(Z-mu)*2/m+(dmu/m)
    
    return dgamma,dbeta,dZ

def back_prop(Y,cache,act_hl, act_fl,regularization,batch_norm=False):
    #Description: Implementation of backward propagation
    #Input: (1) Y (refer function NN_model), 
    #       (2) cache (refer function forward_prop )
    #       (3) act_hl (refer function NN_model)
    #       (4) act_fl (refer function NN_model), 
    #       (5) regularization (refer  function NN_model)
    #       (6) batch_norm (flag to indicate if batch norm is True)
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
        
    m=Y.shape[1]
    L=len(cache)    
    
    A_fl=cache['final activation']
    if act_fl!="softmax":
        dA_fl=-(np.divide(Y,A_fl+10e-8)-np.divide(1-Y,1-A_fl+10e-8))*(1/m)
        dA_l=dA_fl
        
    grads={}
    act=act_fl
    for l in reversed(range(1,L)):
        layer_cache=cache['layer'+str(l)]
        A_prev=layer_cache['A'+str(l-1)]
        W_l=layer_cache['W'+str(l)]
        b_l=layer_cache['b'+str(l)]
        Z_l=layer_cache['Z'+str(l)]
        if batch_norm:
            Z_l=layer_cache['Ztilda'+str(l)]
        
        if l<L-1:
            act=act_hl
            
        if act=="softmax":
            dZ_l=(A_fl-Y)*(1/m)
        else:
            g_prime_l=activation_prime(Z_l, act)
            dZ_l=dA_l*g_prime_l
       
                
        if batch_norm:
            dZtilda_l=dZ_l
            dgamma_l, dbeta_l, dZ_l=get_batchnorm_grads(dZtilda_l,layer_cache,l,m)
            
            
        #dW_l=(1/m)*np.dot(dZ_l,A_prev.T)
        dW_l=np.dot(dZ_l,A_prev.T)
        if regu_type=="L2":
            dW_l=dW_l+(lambd/m)*W_l
        
        #db_l=(1/m)*np.sum(dZ_l,axis=1,keepdims=True)
        db_l=np.sum(dZ_l,axis=1,keepdims=True)
        dA_l=np.dot(W_l.T,dZ_l)
        if regu_type=="dropout" and l>1:
            keep_prob=keep_probs[l-2]
            layer_cache_prev=cache['layer'+str(l-1)]
            D_l=layer_cache_prev['D'+str(l-1)]
            dA_l=dA_l*D_l
            dA_l=dA_l/keep_prob
        if batch_norm:
            grads['layer'+str(l)]={'dW'+str(l):dW_l, 'db'+str(l):db_l,'dgamma'+str(l):dgamma_l,'dbeta'+str(l):dbeta_l}
        else:
            grads['layer'+str(l)]={'dW'+str(l):dW_l, 'db'+str(l):db_l}
    
    return grads

#%% Update parameters using grad descent
def update_weights(weights,grads,learn_rate,batch_norm=False):
    #Description: Weight updation
    #Input: (1) weights (refer format at function initialize_weights), 
    #       (2) grads (refer format at function back_prop)
    #       (3) learning rate or alpha
    #       (4) batch_norm flag
    #Output: updated weights in the same format as weights (refer function initialize_weights)
    L=len(weights)+1
    for l in range(1,L):
        weights_l=weights['layer'+str(l)]
        grads_l=grads['layer'+str(l)]
        weights_l['W'+str(l)]=weights_l['W'+str(l)]-learn_rate*grads_l['dW'+str(l)]
        weights_l['b'+str(l)]=weights_l['b'+str(l)]-learn_rate*grads_l['db'+str(l)]
        if batch_norm:
            weights_l['gamma'+str(l)]=weights_l['gamma'+str(l)]-learn_rate*grads_l['dgamma'+str(l)]
            weights_l['beta'+str(l)]=weights_l['beta'+str(l)]-learn_rate*grads_l['dbeta'+str(l)]
        
    return weights


#%% Functions to convert dictionary of parameters into a column vector required for gradient checking
def unroll_into_column(params,identifier,batch_norm):
    #Description: A function to unroll the parameters in to a column vector required during gradient checking
    #Inputs: (1) Dictionary of weights or gradients,
    #        (2) identifier="weights" if params passed is weights 
    #        (3) batch norm flag
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
        if batch_norm:
            beta_l=params_l[pre_str+'beta'+str(l)]
            gamma_l=params_l[pre_str+'gamma'+str(l)]
            beta_l_col=beta_l.reshape((beta_l.shape[0]*beta_l.shape[1],1))
            gamma_l_col=gamma_l.reshape((gamma_l.shape[0]*gamma_l.shape[1],1))
            temp_col_vec=np.append(beta_l_col,gamma_l_col,axis=0)
            col_vec=np.append(col_vec,temp_col_vec,axis=0)
            
    col_vec=col_vec[1:,0]
    return col_vec.reshape((len(col_vec),1))
    
def roll_into_dict(theta_pm,weights_orig,identifier,batch_norm):
    #Description: A function to roll back weights vector to a dictionary required for gradient checking
    #Inputs: (1) theta_pm (a column vector of weights),
    #        (2) weights_orig (weights dict)
    #        (3) identifier="weights" if params passed is weights  (not required anymore)
    #        (4) batch norm flag
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
         if batch_norm:
             shape_beta=weights_l['beta'+str(l)].shape
             shape_gamma=weights_l['gamma'+str(l)].shape
             num_of_beta_params=shape_beta[0]*shape_beta[1]
             num_of_gamma_params=shape_gamma[0]*shape_gamma[1]
             
             beta_params=theta_pm[counter:counter+num_of_beta_params]
             counter=counter+num_of_beta_params
             
             gamma_params=theta_pm[counter:counter+num_of_gamma_params]
             counter=counter+num_of_gamma_params  
             
             beta_params=beta_params.reshape(shape_beta)
             gamma_params=gamma_params.reshape(shape_gamma)
             
             layer_dict=params_dict['layer'+str(l)]
             layer_dict[pre_str+'beta'+str(l)]=beta_params
             layer_dict[pre_str+'gamma'+str(l)]=gamma_params
         
    return params_dict
            
#%% A function to check the gradients numerically
def check_gradients(weights,X,Y,act_hl,act_fl,epsilon,regularization, batch_norm=False):
    #Description: A function to validate gradients from back prop against numerically computed gradients
    #Inputs: (1) weights (refer format at function initialize_weights),
    #        (2) X (refer function NN_model),
    #        (3) Y (refer function NN_model),
    #        (4) act_hl (refer function NN model)
    #        (5) act_fl (refer function NN model)
    #        (6) epsilon (step value, recommended 10e-7)
    #        (7) regularization (refer function NN model)
    #        (8) batch norm flag
    #Output: A string indicating if gradients are ok or not
           
    # Determine no of classes
    classes=list(set(Y[0,:]))
    n_k=len(classes)

    # one hot encoding for more than one class for more than two classes
    if act_fl=="softmax" or n_k>2: 
        Y=np.repeat(Y, n_k,axis=0)
        Y=Y==np.arange(0,n_k).reshape(n_k,1)
    else:
        n_k=1
    
    cache=forward_prop(weights,X,act_hl, act_fl,regularization, batch_norm)
    grads=back_prop(Y,cache,act_hl, act_fl,regularization,batch_norm)
    
    weights_col= unroll_into_column(weights,"weights",batch_norm)
    grads_col=unroll_into_column(grads,"gradients",batch_norm)
    
    n_params = weights_col.shape[0]
    gradapprox = np.zeros((n_params, 1))
    
    for i in range(0,n_params):
        # plus epsilon
        theta_plus=np.copy(weights_col)
        theta_plus[i,0]=theta_plus[i,0]+epsilon
        
        theta_plus_dict=roll_into_dict(theta_plus,weights,"weights",batch_norm)
        cache_plus=forward_prop( theta_plus_dict,X,act_hl, act_fl,regularization,batch_norm)
        #Cost calculation
        Y_hat_plus=cache_plus['final activation']
        Jplus_i=calc_cost(Y,Y_hat_plus,act_fl)
        
        # minus epsilon
        theta_minus=np.copy(weights_col)
        theta_minus[i,0]=theta_minus[i,0]-epsilon
        theta_minus_dict=roll_into_dict(theta_minus,weights,"weights",batch_norm)
        cache_minus=forward_prop(theta_minus_dict,X,act_hl, act_fl,regularization,batch_norm)
        #Cost calculation
        Y_hat_minus=cache_minus['final activation']
        Jminus_i=calc_cost(Y,Y_hat_minus,act_fl)
        
        gradapprox[i,0]= (Jplus_i-Jminus_i)/(2*epsilon)
      
    numerator = np.linalg.norm(grads_col-gradapprox)                               
    denominator = np.linalg.norm(grads_col)+ np.linalg.norm(gradapprox)                          
    rel_diff = numerator /denominator
    print("Relative difference for gradients =",str(rel_diff) )
    
#    if rel_diff<2e-7:
#        print("Gradients are OK, relative difference =",str(rel_diff))
#    else:
#        print("Gradients are not OK, relative difference =",str(rel_diff))
        
#%% Create mini batches   
def create_batches(X,Y,batch_size):
    #Description: A function to create mini batches
    #Inputs: (1) X (Input training matrix, refer function NN_model)
    #        (2) Y (Input labels, refere function NN_model)
    #        (3) batch_size (an integer specifying the batch size)
    # Output: batches (a list containing all mini batches)      
    
    m = X.shape[1]                  
    batches = []
        
    # Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation]
    
    num_complete_batches = int(np.floor(m/batch_size))
    
    # Get the mini batchs
    for k in range(0, num_complete_batches):
        batch_X = shuffled_X[:, k*batch_size : (k+1)*batch_size]
        batch_Y = shuffled_Y[:, k*batch_size : (k+1)*batch_size]
        batch = (batch_X, batch_Y)
        batches.append(batch)
    
    # Handle left over batch
    if m % batch_size != 0:
        batch_X = shuffled_X[:, num_complete_batches* batch_size:]
        batch_Y = shuffled_Y[:, num_complete_batches* batch_size:]
        batch = (batch_X, batch_Y)
        batches.append(batch)
    
    return batches

#%% Generate layer dims
def generate_layer_dims(n_x,hidden_layer_dims,n_k):
    #Description:  Creates the dimensions of each layer of the NN
    #Inputs: (1) n_x (no of features of the input train matrix X)
    #        (2) hidden_layers (refer function NN_model)
    #        (3) n_k (no of classes)
    #Output: a list if dimensions for each layer in the NN
    layer_dims=[n_x]+hidden_layer_dims
    layer_dims.append(n_k)
#    if n_k>2: # no of classes more than 2
#        layer_dims.append(n_k)
#    else:
#        layer_dims.append(1)
#        n_k=1
    
    return layer_dims

#%% Initialize momentum parameters
def initialize_momentum(weights):
    #Description:  Initializes the momentum values
    #Inputs: (1) weights (refer function initialize weights)
    #Output: Zero initialized momentum values in the same format as weights

    L = len(weights) +1 
    v = {}

    for l in range(1,L):
        layer_weights=weights['layer'+str(l)]
        v_dW_l = np.zeros_like(layer_weights["W" + str(l)])
        v_db_l = np.zeros_like(layer_weights["b" + str(l)])
        v['layer'+str(l)]={"v_dw"+str(l):v_dW_l,"v_db"+str(l):v_db_l}
    return v

#%% Initialize adam parameters
def initialize_adam(weights):
    #Description:  Initializes adam values
    #Inputs: (1) weights (refer function initialize weights)
    #Output: Zero initialized  momentum (v) and and RMS prop (s) in the same format as weights
    L = len(weights) +1 
    v = {}
    s={}

    for l in range(1,L):
        layer_weights=weights['layer'+str(l)]
        v_dW_l = np.zeros_like(layer_weights["W" + str(l)])
        v_db_l = np.zeros_like(layer_weights["b" + str(l)])
        s_dW_l = np.zeros_like(layer_weights["W" + str(l)])
        s_db_l = np.zeros_like(layer_weights["b" + str(l)])
        v['layer'+str(l)]={"v_dw"+str(l):v_dW_l,"v_db"+str(l):v_db_l}
        s['layer'+str(l)]={"s_dw"+str(l):s_dW_l,"s_db"+str(l):s_db_l}
    return v,s

#%% Update weights with momentum
def update_weights_momentum(weights,grads,learn_rate,v,beta1):
    #Description:  Update weights using momentum formulation
    #Inputs: (1) weights (refer function initialize weights)
    #        (2) grads (gradients, refer function back_prop)
    #        (3) learn_rate 
    #        (4) current momentum value v (refer function initialize_momentum)
    #        (5) momentum parameter beta1(exponential average over approx 1/(1-beta1) prev gradients or dW)
    #Output: updated weights, updated momentum values
    
    L = len(weights) +1 
    
    for l in range(1,L):
        # Unpack layer params
        layer_v=v['layer'+str(l)]
        layer_weights=weights['layer'+str(l)]
        layer_grads=grads['layer'+str(l)]
        
        # Update momentum
        layer_v["v_dw" + str(l)] = beta1*layer_v["v_dw" + str(l)]+(1-beta1)*layer_grads['dW' + str(l)]
        layer_v["v_db" + str(l)] = beta1*layer_v["v_db" + str(l)]+(1-beta1)*layer_grads['db' + str(l)]
        
        # Update weights
        layer_weights["W" + str(l)] = layer_weights['W' + str(l)]-learn_rate*layer_v["v_dw" + str(l)]
        layer_weights["b" + str(l)] = layer_weights['b' + str(l)]-learn_rate*layer_v["v_db" + str(l)]
        
    return weights,v

#%% update_weights_with adam
def update_weights_adam(weights,grads,learn_rate, t,v,s,beta1,beta2):
    #Description:  Update weights using adam formulation
    #Inputs: (1) weights (refer function initialize weights)
    #        (2) grads (gradients, refer function back_prop)
    #        (3) learn_rate 
    #        (4) t (current iteration number)
    #        (5) current momentum value v (refer function initialize_momentum)
    #        (6) current rms value (s)
    #        (5) momentum parameter beta1
    #        (6) rms prop parameter beta2 (exponential average over approx 1/(1-beta2) dW**2 values)
    #Output: updated weights, updated v, updated s
    
    
    L = len(weights) +1 
    for l in range(1,L):
        # Unpack layer params
        layer_v=v['layer'+str(l)]
        layer_s=s['layer'+str(l)]
        layer_weights=weights['layer'+str(l)]
        layer_grads=grads['layer'+str(l)]
        
        
        # Update momentums and rms params
        layer_v["v_dw" + str(l)] = beta1*layer_v["v_dw" + str(l)]+(1-beta1)*layer_grads['dW' + str(l)]
        layer_v["v_db" + str(l)] = beta1*layer_v["v_db" + str(l)]+(1-beta1)*layer_grads['db' + str(l)]
        layer_s["s_dw" + str(l)] = beta2*layer_s["s_dw" + str(l)]+(1-beta2)*np.square(layer_grads['dW' + str(l)])
        layer_s["s_db" + str(l)] = beta2*layer_s["s_db" + str(l)]+(1-beta2)*np.square(layer_grads['db' + str(l)])
        
        # Apply corrections
        v_dw_corr=layer_v["v_dw" + str(l)]/(1-beta1**t)
        v_db_corr=layer_v["v_db" + str(l)]/(1-beta1**t)
        s_dw_corr=layer_s["s_dw" + str(l)]/(1-beta2**t)
        s_db_corr=layer_s["s_db" + str(l)]/(1-beta2**t)
        
        layer_weights["W" + str(l)] = layer_weights['W' + str(l)]-learn_rate*v_dw_corr/np.sqrt(s_dw_corr+10e-8)
        layer_weights["b" + str(l)] = layer_weights['b' + str(l)]-learn_rate*v_db_corr/np.sqrt(s_db_corr+10e-8)
        
    return weights,v,s
        
#%% Final model
def NN_model(X,Y,hidden_layer_dims,no_epochs,batch_size,learn_rate,act_hl, act_fl,regularization=None,optimization=None, batch_norm=False):
    ## Description: This is entire NN model
    # Inputs:
    # (1) X_train_std (training matrix of shape (n_x,m) where n_x is the no of input features and m is no of examples)
    # (2) Y_train (labels or ground truth of shape (1,m)), make sure labels are int values 
    # (3) hidden_layer_dims ( a list of layer sizes of hidden layers e.g [3,1] means it is a 3 layer NN (L=3) with 3 units in 1st layer, 
    #     1 unit in 2nd layer. The no of units in the final layer is the no of classes k.The no of units in the 0th or input layer 
    #     is n_x=X.shape[0]
    # (4) no_itns (no of iterations)
    # (5) batch size (a recommended size is a any power of 2)
    # (6) learn_rate (the learning rate or step size of gradient descent)
    # (7) act_hl (activation type in hidden layer, choices:'relu','sigmoid' or 'tanh'),
    # (8) act_fl (activation type in final layer, choices:'relu','sigmoid' or 'tanh'),
    # (7) regularization (It is either None meaning no regularization, {'L2': lambd} in case of L2 regularization or
    #     {'dropout':[..keep_prob_l..]} where l is a hidden layer that starts at 1 to L-2 and where keep_prob is probability 
    #     of keeping a unit in layer l. By default keep_prob is set for the final layer
    # (9) optimization (choices are 0-Gradient descent (GD), 1-GD with momentum, 2-Adam optimization)
    #     should be formatted as optimization=0 for GD (also default), {'momentum':beta1},  {'Adam':[beta1,beta2]}
    #     Determine no of classes
    # (10) batch norm flag (to indicate if batch norm is turned on)
    # Output: optimized weights (refer format at function intialize_weights), cost (optional)
    
    classes=list(set(Y[0,:]))
    n_k=len(classes)
    
    
    # one hot encoding for more than one class for more than two classes
    if act_fl=="softmax" or n_k>2:
            Y=np.repeat(Y, n_k,axis=0)
            Y=Y==np.arange(0,n_k).reshape(n_k,1)
    else:
        n_k=1 # if final layer is sigmoid and it is a binary class problem
        
    # Determine the  dimsions of each layer of the NN
    layer_dims=generate_layer_dims(X.shape[0],hidden_layer_dims,n_k)
    
    # Initialize regularization    
     
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

    
    
    # Initialize weights
    weights=initialize_weights(layer_dims,batch_norm)
    
    # Intialize optimation
    optim_type=None
    if optimization!=None and optimization!=0:
        optim_type=list(optimization.keys())[0]
        if optim_type=='momentum':
            v=initialize_momentum(weights)
            beta1=optimization['momentum']
        elif optim_type=='adam':
            v,s=initialize_adam(weights)
            betas=optimization['adam']
            beta1=betas[0]
            beta2=betas[1]
            t_adam=1
            
    cost_list=[]
    epoch_list=[]
    
    cur_epoch_no=0
    while cur_epoch_no<=no_epochs:
        batches_list=create_batches(X,Y,batch_size)
        itn_no=0
        
        for batch in batches_list:
            # Unpack batch
            batch_X=batch[0]
            batch_Y=batch[1]
            
            #Forward prop
            cache=forward_prop(weights,batch_X,act_hl, act_fl,regularization,batch_norm,pred_flag=False)
                
            #Cost calculation
            batch_Y_hat=cache['final activation']
            cost=calc_cost(batch_Y,batch_Y_hat,act_fl)
                
            if regu_type=="L2":
                cost=cost+calc_frobenius_norm_of_weights(weights,lambd,m=batch_X.shape[1])
                
            #save cost value
            cost_list.append(cost/n_k)
            epoch_list.append(cur_epoch_no)
            if cur_epoch_no % 100 == 0 and itn_no==0:
                print("Cost after epoch {}: {}".format(cur_epoch_no, round(cost/n_k,3)))
                #cost_list.append(cost/n_k)
                #epoch_list.append(cur_epoch_no // 100)
                
            #Back prop
            grads=back_prop(batch_Y,cache,act_hl, act_fl,regularization,batch_norm)
            
            # Update parameters
            if optim_type=='momentum':
                weights,v=update_weights_momentum(weights,grads,learn_rate,v,beta1)
            elif optim_type=='adam':
               weights,v,s=update_weights_adam(weights,grads,learn_rate, t_adam ,v,s,beta1,beta2)
               t_adam+=1
            else: # plain vanilla GD
                weights=update_weights(weights,grads,learn_rate,batch_norm)
                
            itn_no+=1
                
        cur_epoch_no+=1
    # plots    
    if regu_type!="dropout":    
        plt.figure(figsize=(5,5), dpi= 100, facecolor='w', edgecolor='k')    
        #iterations=list(range(0,len(cost_list)))    
            
        plt.plot(epoch_list,cost_list)
        plt.ylabel("J")
        plt.xlabel("Epochs (x 100)")
    return weights, cost_list


#%% Predict
def NN_model_predictions(weights,X,act_hl, act_fl,regularization,batch_norm=False):
    #Descrption: A function calculate output or predictions
    #Input: (1) X (refer function NN_model),
    #       (2) act_hl (refer function NN_model),
    #       (3) act_fl (refer function NN_model),
    #       (4) regularization(refer function NN_model)
    #       (5) batch norm flag
    #Output: Prediction matrix Y_hat 
    
    regularization=None

    cache=forward_prop(weights,X,act_hl, act_fl,regularization,batch_norm,pred_flag=True)
    
    
    Y_hat=cache['final activation']
    n_k=Y_hat.shape[0]
    if n_k==1: # binary classification
        Y_hat[Y_hat>0.5]=1
        Y_hat[Y_hat<=0.5]=0
    else:
        Y_hat=np.argmax(Y_hat,axis=0).reshape((1,X.shape[1]))
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
    
 
    
