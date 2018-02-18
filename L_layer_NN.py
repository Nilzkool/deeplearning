# NN library for binary classification problems 
import numpy as np
import matplotlib.pyplot as plt
#%% A function to initialize parameters
def initialize_weights(layer_dims):
    # Description: A xavier implementation of random initialization of weights
    # Input: (1) layer_dims (dimensions of each layer of NN)
    # Output: a dict containg initial values of all parameters formatted for evry layer l as
    #         {...,{layerl:{'Wl':Wl, 'bl':bl},...}
    
    L=len(layer_dims) # all layers of NN including input and output layers
    
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
def forward_prop(weights,X_inp,act_hl, act_fl,regularization):
    #Descrption: Implementation of forward propagation
    #Input: (1) weights (refer the format in the function initialize_weights), 
    #       (2) X (taining matrix X or a mini batch of training matrix X),
    #       (3) act_hl (hidden layer activation type, refer function NN_model)
    #       (4) act_fl (final layer activation, refer function NN_model), 
    #       (5) regularization (refer function NN_model)
    
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
        

    cache['final activation']=A_prev
     
    return cache

#%% Calculate cost of function    
    
def log_loss(Y,Y_hat):
    #Descrption: Log loss
    #Input: (1) Y (refer function NN model)
    #       (2) Y_hat (final activations)
    #       
    #Output: log loss
    
    L=-(Y*np.log(Y_hat+10e-8)+(1-Y)*np.log(1-Y_hat+10e-8))
    return L

def calc_cost(Y,Y_hat):
    #Descrption: Cost calculation
    #Input: (1) Y (refer function NN model)
    #       (2) Y_hat (final activations)
    #        n_k (no of classes, for binary classification n_k is taken as one)
    m=Y.shape[1]
    J=(1/(m))*np.sum(log_loss(Y,Y_hat))
    return J

def calc_frobenius_norm_of_weights(weights,lambd,m):
    #Descrption: Calculate frobenius norm for L2 regularization
    #Input: (1) weights (refer function intialize_weights), 
    #       (2) lambd (L2 regularization),
    #       (3) m (no of training examples)
    #       (4) Y_hat (final activations)
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
    #Input: (1) Y (refer function NN_model), 
    #       (2) cache (refer function forward_prop )
    #       (3) act_hl (refer function NN_model)
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
    dA_fl=-(np.divide(Y,A_fl+10e-8)-np.divide(1-Y,1-A_fl+10e-8))
    
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
def update_weights(weights,grads,learn_rate, ):
    #Description: Weight updation
    #Input: (1) weights (refer format at function initialize_weights), 
    #       (2) grads (refer format at function back_prop)
    #       (3) learning rate or alpha
    #Output: updated weights in the same format as weights (refer function initialize_weights)
    L=len(weights)+1
    for l in range(1,L):
        weights_l=weights['layer'+str(l)]
        grads_l=grads['layer'+str(l)]
        weights_l['W'+str(l)]=weights_l['W'+str(l)]-learn_rate*grads_l['dW'+str(l)]
        weights_l['b'+str(l)]=weights_l['b'+str(l)]-learn_rate*grads_l['db'+str(l)]
    return weights


#%% Functions to convert dictionary of parameters into a column vector required for gradient checking
def unroll_into_column(params,identifier):
    #Description: A function to unroll the parameters in to a column vector required during gradient checking
    #Inputs: (1) Dictionary of weights or gradients,
    #        (2) identifier="weights" if params passed is weights 
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
    #Inputs: (1) theta_pm (a column vector of weights),
    #        (2) identifier="weights" if params passed is weights  (not required anymore)
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
    #Inputs: (1) weights (refer format at function initialize_weights),
    #        (2) X (refer function NN_model),
    #        (3) Y (refer function NN_model),
    #        (4) act_hl (refer function NN model)
    #        (5) act_fl (refer function NN model)
    #        (6) epsilon (step value, recommended 10e-7)
    #        (7) regularization (refer function NN model)
    #Output: A string indicating if gradients are ok or not
           
      # Determine no of classes
    classes=list(set(Y[0,:]))
    n_k=len(classes)

    # one hot encoding for more than one class for more than two classes
    if n_k>2: 
        Y=np.repeat(Y, n_k,axis=0)
        Y=Y==np.arange(0,n_k).reshape(n_k,1)
    else:
        n_k=1
    
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
    if n_k>2: # no of classes more than 2
        layer_dims.append(n_k)
    else:
        layer_dims.append(1)
        n_k=1
    
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
def NN_model(X,Y,hidden_layer_dims,no_epochs,batch_size,learn_rate,act_hl, act_fl,regularization=None,optimization=None):
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
    
    # Output: optimized weights, cost (optional)
    
    classes=list(set(Y[0,:]))
    n_k=len(classes)
    
    # Determine the  dimsions of each layer of the NN
    layer_dims=generate_layer_dims(X.shape[0],hidden_layer_dims,n_k)
    
    # one hot encoding for more than one class for more than two classes
    if n_k>2: 
        Y=np.repeat(Y, n_k,axis=0)
        Y=Y==np.arange(0,n_k).reshape(n_k,1)
    else:
        n_k=1
        
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
    weights=initialize_weights(layer_dims)
    
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
            cache=forward_prop(weights,batch_X,act_hl, act_fl,regularization)
                
            #Cost calculation
            batch_Y_hat=cache['final activation']
            cost=calc_cost(batch_Y,batch_Y_hat)
                
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
            grads=back_prop(batch_Y,cache,act_hl, act_fl,regularization)
            
            # Update parameters
            if optim_type=='momentum':
                weights,v=update_weights_momentum(weights,grads,learn_rate,v,beta1)
            elif optim_type=='adam':
               weights,v,s=update_weights_adam(weights,grads,learn_rate, t_adam ,v,s,beta1,beta2)
               t_adam+=1
            else: # plain vanilla GD
                weights=update_weights(weights,grads,learn_rate)
                
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
    
 
    
