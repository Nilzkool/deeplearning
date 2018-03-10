# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 18:48:03 2018

@author: u0113548
"""

# CNN library for  classification problems 
import numpy as np
import matplotlib.pyplot as plt

#%%
def get_convlayer_output_shape(layer,input_shape):
    nH_inp=input_shape[0]
    nW_inp=input_shape[1]
    nC_inp=input_shape[2]
    
    f=layer['Filter_size']
    p=layer['Padding']
    s=layer['Stride']
    nH_conv=int((nH_inp+2*p-f)/s+1)
    nW_conv=int((nW_inp+2*p-f)/s+1)
    nC_conv=layer['Num_filters']
    
    f=layer['Pooling_filter_size']
    s=layer['Pooling_stride']
    nH_op=int((nH_conv-f)/s+1)
    nW_op=int((nW_conv-f)/s+1)
    nC_op=nC_conv
    
    shape_op=(nH_op,nW_op,nC_op)
    return shape_op

#%% Weights initialization

def initialize_weights(architecture):
    weights={}
    conv_output_shapes=[]
    l_cnt=0
    for layer in architecture:
        if layer['Type']=='conv_layer':
            f=layer['Filter_size']
            n_filters=layer['Num_filters']
            if l_cnt==0:
                input_shape= layer['Input_shape']
            else:
                input_shape=conv_output_shapes[l_cnt-1]
             
            nC=input_shape[2] # num of filter channels    
            W_l= np.random.randn(f,f,nC,n_filters)*0.01
            b_l=np.zeros((1,1,1,n_filters))
            weights_layer_dict={'Type':'conv_layer','W':W_l,'b':b_l}
            weights['layer'+str(l_cnt+1)]=weights_layer_dict
            conv_output_shapes.append(get_convlayer_output_shape(layer,input_shape))
            
        elif layer['Type']=='FC' or layer['Type']=='output_layer':
            prev_layer=architecture[l_cnt-1]
            if prev_layer['Type']=="conv_layer": #first layer after conv layers
                shape_conv_out=conv_output_shapes[l_cnt-1]
                flat_inp_size=shape_conv_out[0]*shape_conv_out[1]*shape_conv_out[2]
                num_neurons=layer['Num_neurons']
                W_l= np.random.randn(num_neurons,flat_inp_size)*0.01
                b_l=np.zeros((num_neurons,1))
                weights_layer_dict={'Type':'FC','W':W_l,'b':b_l}
                weights['layer'+str(l_cnt+1)]=weights_layer_dict
                
            else:
                num_neurons_prev_layer=prev_layer['Num_neurons']
                layer_type=layer['Type']
                if layer_type=='FC':
                    num_neurons=layer['Num_neurons']
                else:
                    num_neurons=layer['Num_output_units']
                W_l= np.random.randn(num_neurons,num_neurons_prev_layer)*0.01
                b_l=np.zeros((num_neurons,1))
                
                weights_layer_dict={'Type':layer_type,'W':W_l,'b':b_l}
                weights['layer'+str(l_cnt+1)]=weights_layer_dict
        
        l_cnt+=1
        
    return weights
#%% Create mini batches   
def create_mini_batches(X,Y,batch_size):
    #Description: A function to create shuffled mini batches
    #Inputs: (1) X (Input training matrix, refer function NN_model)
    #        (2) Y (Input labels, refere function NN_model)
    #        (3) batch_size (an integer specifying the batch size)
    # Output: batches (a list containing all mini batches)      
    
    m = X.shape[0]                  
    batches = []
        
    # Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[ permutation,:,:,:]
    shuffled_Y = Y[:, permutation]
    
    num_complete_batches = int(np.floor(m/batch_size))
    
    # Get the mini batchs
    for k in range(0, num_complete_batches):
        batch_X = shuffled_X[k*batch_size : (k+1)*batch_size,:,:,:]
        batch_Y = shuffled_Y[:, k*batch_size : (k+1)*batch_size]
        batch = (batch_X, batch_Y)
        batches.append(batch)
    
    # Handle left over batch
    if m % batch_size != 0:
        batch_X = shuffled_X[num_complete_batches* batch_size:, :,:,:]
        batch_Y = shuffled_Y[:, num_complete_batches* batch_size:]
        batch = (batch_X, batch_Y)
        batches.append(batch)
    
    return batches
#%% Forward prop
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
       
    elif act_name=="softmax":
       t=np.exp(z)
       g=t/np.sum(t,axis=0,keepdims=True)
       
    return g    

def conv_one_step(a_slice_prev, W, b):
    s = np.multiply(a_slice_prev,W)
    # Sum over all entries of the volume s.
    Z = np.sum(s)
    # Add bias b to Z. Cast b to a float() so that Z results in a scalar value.
    Z = float(Z+b)
    return Z

def apply_convolution_using_for(A_prev,layer,W,b):
    #Input dims
    m=A_prev.shape[0]
    nH_prev=A_prev.shape[1]
    nW_prev=A_prev.shape[2]
    
    # Filter shape
    (f, f, nC_prev, nC) = W.shape
    
    # Unpack params
    p=layer['Padding']
    s=layer['Stride']
    
    # Apply padding
    A_prev_pad=np.pad(A_prev,((0,0),(p,p),(p,p),(0,0)),'constant')
    
    # Calc output dims
    nH=int((nH_prev+2*p-f)/s)+1
    nW=int((nW_prev+2*p-f)/s)+1
    
    # Initialize output volume
    Z = np.zeros((m, nH, nW, nC))
    
    # For loop
    for i in range(m):
        a_prev_pad = A_prev_pad[i,:,:,:]
        for h in range(nH):
            for w in range(nW):
                for c in range(nC):
                    # define the convolution region over input volume
                    vert_start = h*s
                    vert_end = vert_start+f
                    horiz_start = w*s
                    horiz_end = horiz_start+f
    
                    # Use the corners to define the (3D) slice of a_prev_pad (See Hint above the cell). (≈1 line)
                    a_slice_prev = a_prev_pad[vert_start:vert_end,horiz_start:horiz_end,:]
                    Z[i, h, w, c] = conv_one_step(a_slice_prev, W[:,:,:,c], b[:,:,:,c])
                    
    return Z             
    
    
    
def apply_convolution(A_prev,layer,W,b):
    
    #Input dims
    m=A_prev.shape[0]
    nH_prev=A_prev.shape[1]
    nW_prev=A_prev.shape[2]
    
    # Filter shape
    (f, f, nC_prev, nC) = W.shape
    
    # Unpack params
    p=layer['Padding']
    s=layer['Stride']
    
    # Apply padding
    A_prev=np.pad(A_prev,((0,0),(p,p),(p,p),(0,0)),'constant')
    
    # Calc output dims
    nH=int((nH_prev+2*p-f)/s)+1
    nW=int((nW_prev+2*p-f)/s)+1
    
    # Initialize output volume
    Z = np.zeros((m, nH, nW, nC))
    
    # Loop over each element in output volume
    #cnt=0
    for h in range(nH):
        for w in range(nW):
#            print(cnt)
#            cnt+=1
            # define the convolution region over input volume
            vert_start = h*s
            vert_end = vert_start+f
            horiz_start = w*s
            horiz_end = horiz_start+f
            
            #print(horiz_end-horiz_start)
            # slice out the region
            A_slice_prev=A_prev[:,vert_start:vert_end,horiz_start:horiz_end,:]
            
            # Braoadcasting along direction m (num examples) and nC (num of filters)
            
            # Expand the slice into multiple slices for broadcasting
     
            expand_A_slice_prev=np.zeros((m,f,f,nC_prev,1))
            expand_A_slice_prev[:,:,:,:,0]=A_slice_prev
            
            # Expand the filter weights
            expand_filter_weights=np.zeros((1,f,f,nC_prev,nC))
            expand_filter_weights[0,:,:,:,:]=W
            
            # convolve over the slice
            temp_mult= expand_A_slice_prev*expand_filter_weights
            Z[:,h,w,:]=np.sum(temp_mult,axis=(1,2,3))+b
    
    return Z



def apply_pooling_using_for(A_prev,layer):
    #Input dims
    m=A_prev.shape[0]
    nH_prev=A_prev.shape[1]
    nW_prev=A_prev.shape[2]
    
    # pooling hyperprameters
    pool_type=layer['Pooling_type']
    f = layer['Pooling_filter_size']
    s=layer['Pooling_stride']
    
    
    # Calc output dims
    nH=int((nH_prev-f)/s)+1
    nW=int((nW_prev-f)/s)+1
    nC=A_prev.shape[3]
    
    # Initialize output volume
    Z = np.zeros((m, nH, nW, nC))
    
    # for loop
    for i in range(m):
        for h in range(nH):
            for w in range(nW):
                for c in range(nC):
                    # define the convolution region over input volume
                    vert_start = h*s
                    vert_end = vert_start+f
                    horiz_start = w*s
                    horiz_end = horiz_start+f
                    
                    a_prev_slice = A_prev[i,vert_start:vert_end,horiz_start:horiz_end,c]
                    
                    # Compute the pooling operation on the slice. Use an if statment to differentiate the modes. Use np.max/np.mean.
                    if pool_type == "max":
                        Z[i, h, w, c] = np.max(a_prev_slice)
                    elif pool_type == "average":
                        Z[i, h, w, c] = np.mean(a_prev_slice)
    return Z

def apply_pooling(A_prev,layer):
    #Input dims
    m=A_prev.shape[0]
    nH_prev=A_prev.shape[1]
    nW_prev=A_prev.shape[2]
    
    # Filter shape
    f = layer['Pooling_filter_size']
    
    # Unpack params
    s=layer['Pooling_stride']
    
    
    # Calc output dims
    nH=int((nH_prev-f)/s)+1
    nW=int((nW_prev-f)/s)+1
    nC=A_prev.shape[3]
    
    # Initialize output volume
    Z = np.zeros((m, nH, nW, nC))
    
    for h in range(nH):
        for w in range(nW):

            vert_start = h*s
            vert_end = vert_start+f
            horiz_start = w*s
            horiz_end = horiz_start+f
            
            # slice out the region
            A_slice_prev=A_prev[:,vert_start:vert_end,horiz_start:horiz_end,:]
            
            if layer['Pooling_type']=="max":
                Z[:,h,w,:]=np.max(A_slice_prev,axis=(1,2))
            elif layer['Pooling_type']=="avg":
                Z[:,h,w,:]=np.mean(A_slice_prev,axis=(1,2))
    
    return Z
        

def forward_prop(X,weights,architecture):
    cache={}
    
    ## forward pass
    A_prev=X 
    l_cnt=0
    num_layers=len(architecture)
    for layer in architecture:
        layer_weights=weights['layer'+str(l_cnt+1)]
        
        if layer['Type']=='conv_layer':
            W_l=layer_weights['W']
            
            b_l=layer_weights['b']
            
            #Z_conv=apply_convolution(A_prev,layer,W_l,b_l)
            Z_conv=apply_convolution_using_for(A_prev,layer,W_l,b_l)
            Zconv_relued=activation(Z_conv+b_l,'relu')
            #Z_pool=apply_pooling(Zconv_relued,layer)
            Z_pool=apply_pooling_using_for(Zconv_relued,layer)
            layer_cache={'Type':'conv_layer','A_prev':A_prev,'W':W_l,'b': b_l,'Zconv':Z_conv,
                         'Zconv_relued':Zconv_relued,'Zpool':Z_pool}
            cache['layer'+str(l_cnt+1)]=layer_cache
            A_prev=Z_pool
            
        elif layer['Type']=='FC' or layer['Type']=='output_layer':
            prev_layer=architecture[l_cnt-1]
            
            if prev_layer['Type']=="conv_layer":
                # flatten the input
                A_prev=A_prev.reshape(A_prev.shape[0],-1)
                A_prev=A_prev.T
               
            # Unpack weights
            W_l=layer_weights['W']
            b_l=layer_weights['b']
            Z_l=np.dot(W_l,A_prev)+b_l
            
            layer_cache={'Type':layer['Type'],'Activation':layer['Activation'],'A_prev':A_prev,'W':W_l,'b': b_l,'Z':Z_l}
            cache['layer'+str(l_cnt+1)]=layer_cache
                
            if l_cnt==num_layers-1: #final layer
                A_prev=activation(Z_l,'softmax')
            else:
                A_prev=activation(Z_l,'relu')
                    
        l_cnt+=1  
    cache['final activation']=A_prev   
    return cache
                
            #else:
                
#%% Calculate cost of function    
    
def log_loss(Y,Y_hat,act_fl="softmax"):
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

#%% back propagation
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


def create_max_mask(x):
    mask = x==np.max(x)
    return mask

def pool_backward_using_for(dA, layer_cache, layer_architecture):
    # Unpack pooling input
    A_prev=layer_cache['Zconv_relued']
    
    # Unpack pooling hyperparameters
    pool_type=layer_architecture['Pooling_type']
    f=layer_architecture['Pooling_filter_size']
    s=layer_architecture['Pooling_stride']
    
    # Retrieve dimensions of input and pooled output
    m, nH_prev, nW_prev, nC_prev = A_prev.shape
    m, nH, nW, nC = dA.shape
    
    dA_prev=np.zeros_like(A_prev)
    
    for i in range(m):                       
        a_prev = A_prev[i,:,:,:]
        for h in range(nH):                   # loop on the vertical axis
            for w in range(nW):               # loop on the horizontal axis
                for c in range(nC):           # loop over the channels (depth)
                    
                    # Find the corners of the current "slice" (≈4 lines)
                    vert_start = h
                    vert_end = vert_start+f
                    horiz_start = w
                    horiz_end = horiz_start+f
                    
                    # Compute the backward propagation in both modes.
                    if pool_type == "max":
                        # Use the corners and "c" to define the current slice from a_prev (≈1 line)
                        a_prev_slice = a_prev[vert_start:vert_end,horiz_start:horiz_end,c]
                        # Create the mask from a_prev_slice (≈1 line)
                        mask = create_max_mask(a_prev_slice)
                        # Set dA_prev to be dA_prev + (the mask multiplied by the correct entry of dA) (≈1 line)
                        dA_prev[i, vert_start: vert_end, horiz_start: horiz_end, c] += mask*dA[i,h,w, c]
    
    return dA_prev                    

def pool_backward(dA, layer_cache, layer_architecture):
    
    # Unpack pooling input
    A_prev=layer_cache['Zconv_relued']
    
    # Unpack pooling hyperparameters
    pool_type=layer_architecture['Pooling_type']
    f=layer_architecture['Pooling_filter_size']
    s=layer_architecture['Pooling_stride']
    
    # Retrieve dimensions of input and pooled output
    m, nH_prev, nW_prev, nC_prev = A_prev.shape
    m, nH, nW, nC = dA.shape
    
    dA_prev=np.zeros_like(A_prev)
    for h in range(nH):
        for w in range (nW):
            
            # define subset volume borders in A_prev over which pooling was done
            vert_start = h
            vert_end = vert_start+f
            horiz_start = w
            horiz_end = horiz_start+f
            
            if pool_type=="max":
                # carve out the subset volume in A_prev
                A_slice_prev=A_prev[:,vert_start:vert_end,horiz_start:horiz_end,:]
                
                # define the mask
                maxes=np.max(A_slice_prev,axis=(1,2))
                expand_maxes=np.zeros((m,1,1,nC))
                expand_maxes[:,0,0,:]=maxes
                mask = A_slice_prev==expand_maxes
                
                # Allow dAs which affected output
                expand_dAs=np.zeros((m,1,1,nC))
                expand_dAs[:,0,0,:]=dA[:,h,w,:]
                temp_mult=mask*expand_dAs
                
                # Update gradients
                dA_prev[:,vert_start:vert_end,horiz_start:horiz_end,:]+=temp_mult
                
            #elif pool_type="avg":
    return dA_prev              


def conv_backward_using_for(dZ,layer_cache, layer_architecture):
    # Retrieve saved input to convolution operation
    A_prev=layer_cache['A_prev']
    
    # Retrieve dimensions from input's shape
    (m, nH_prev, nW_prev, nC_prev) = A_prev.shape
    
    # Get the filter weights and bias for this layer
    W=layer_cache['W']
    b=layer_cache['b']
    
    # Retrieve filter dimensions
    (f, f, nC_prev, nC) = W.shape
    
    # Retrieve dimensions from dZ's shape
    (m, nH, nW, nC) = dZ.shape
    
    # Retrieve hyperparameteres of convolutions
    s=layer_architecture['Stride']
    p=layer_architecture['Padding']
    
    # Initialize dA_prev, dW, db
    dA_prev = np.zeros_like(A_prev)                           
    dW = np.zeros_like(W)
    db = np.zeros_like(b)
    
    # Pad A_prev and dA_prev
    A_prev_pad=np.pad(A_prev,((0,0),(p,p),(p,p),(0,0)),'constant')
    dA_prev_pad=np.pad(dA_prev,((0,0),(p,p),(p,p),(0,0)),'constant')
    
    for i in range(m):
        a_prev_pad = A_prev_pad[i,:,:,:]
        da_prev_pad = dA_prev_pad[i,:,:,:]
        for h in range(nH):                   # loop over vertical axis of the output volume
            for w in range(nW):               # loop over horizontal axis of the output volume
                for c in range(nC):           # loop over the channels of the output volume
                    # Find the corners of the current "slice"
                    vert_start = h*s
                    vert_end = vert_start+f
                    horiz_start = w*s
                    horiz_end = horiz_start+f
                    
                    a_slice = a_prev_pad[vert_start:vert_end,horiz_start:horiz_end,:]

                    # Update gradients for the window and the filter's parameters using the code formulas given above
                    da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += W[:,:,:,c] * dZ[i, h, w, c]
                    dW[:,:,:,c] += a_slice * dZ[i, h, w, c]
                    db[:,:,:,c] += dZ[i, h, w, c]
                    
        dA_prev[i, :, :, :] = da_prev_pad[p:-p, p:-p, :]
    return dA_prev, dW, db
             
def conv_backward(dZ,layer_cache, layer_architecture):
    
    # Retrieve saved input to convolution operation
    A_prev=layer_cache['A_prev']
    
    # Retrieve dimensions from input's shape
    (m, nH_prev, nW_prev, nC_prev) = A_prev.shape
    
    # Get the filter weights and bias for this layer
    W=layer_cache['W']
    b=layer_cache['b']
    
    # Retrieve filter dimensions
    (f, f, nC_prev, nC) = W.shape
    
    # Retrieve dimensions from dZ's shape
    (m, nH, nW, nC) = dZ.shape
    
    # Retrieve hyperparameteres of convolutions
    s=layer_architecture['Stride']
    p=layer_architecture['Padding']
    
    # Initialize dA_prev, dW, db
    dA_prev = np.zeros_like(A_prev)                           
    dW = np.zeros_like(W)
    db = np.zeros_like(b)
    
    # Pad A_prev and dA_prev
    A_prev_pad=np.pad(A_prev,((0,0),(p,p),(p,p),(0,0)),'constant')
    dA_prev_pad=np.pad(dA_prev,((0,0),(p,p),(p,p),(0,0)),'constant')
    
    for h in range(nH):
        for w in range (nW):
            
            # define subset volume borders in A_prev_pad over which pooling was done
            vert_start = h*s
            vert_end = vert_start+f
            horiz_start = w*s
            horiz_end = horiz_start+f
    
            # carve out the subset volume in A_prev
            A_slice_prev=A_prev_pad[:,vert_start:vert_end,horiz_start:horiz_end,:]
            expand_A_slice_prev=np.zeros((m,f,f,nC_prev,1))
            expand_A_slice_prev[:,:,:,:,0]=A_slice_prev
            
            expand_dZs=np.zeros((m,1,1,1,nC))
            expand_dZs[:,0,0,0,:]=dZ[:,h,w,:]
            
            
            # Update dA
            expand_filter_weights=np.zeros((1,f,f,nC_prev,nC))
            expand_filter_weights[0,:,:,:,:]=W
            
            temp_mult=expand_filter_weights * expand_dZs
            temp_sum=np.sum(temp_mult,axis=4)
            dA_prev_pad[:,vert_start:vert_end,horiz_start:horiz_end,:] += temp_sum
            
            # Update dW
            temp_mult_dW=expand_A_slice_prev*expand_dZs
            temp_sum_dW=np.sum(temp_mult_dW,axis=0)
            dW+=temp_sum_dW
            
            # Update db
            temp_sum_db=np.sum(dZ[:,h,w,:],axis=0)
            db+=temp_sum_db    
            
    dA_prev[:, :, :, :] = dA_prev_pad[:,p:-p, p:-p, :]
    
    return  dA_prev,  dW, db
            
def back_prop(Y,cache,architecture):
    
    m=Y.shape[1]
    L=len(architecture)    
    
    A_fl=cache['final activation']
#    if act_fl!="softmax":
#        dA_fl=-(np.divide(Y,A_fl+10e-8)-np.divide(1-Y,1-A_fl+10e-8))*(1/m)
#        dA_l=dA_fl
    grads={}
    l=0
    dA_l=None
    for l in reversed(range(0,L)):
        layer_cache=cache['layer'+str(l+1)]
        layer_architecture=architecture[l]
        layer_type=layer_cache['Type']
        
        if layer_type=="FC" or layer_type=="output_layer":
            A_prev=layer_cache['A_prev']
            Z_l=layer_cache['Z']
            W_l=layer_cache['W']
            b_l=layer_cache['b']
            
            activation=layer_cache['Activation']
            if activation=="softmax":
                dZ_l=(A_fl-Y)*(1/m)
            else:
                g_prime_l=activation_prime(Z_l, activation)
                dZ_l=dA_l*g_prime_l
            
            dW_l=np.dot(dZ_l,A_prev.T)
            db_l=np.sum(dZ_l,axis=1,keepdims=True)
            
            dA_l=np.dot(W_l.T,dZ_l)
        
        else:
            layer_cache_ahead=cache['layer'+str(l+2)]
            layer_type_ahead=layer_cache_ahead['Type']
            
            if layer_type_ahead=="FC" or layer_type_ahead=="output_layer":
                #reshape into volume matrix when transitioning from FC layer
                Z_l=layer_cache['Zpool']
                shape_Z_l=Z_l.shape
                dA_l=dA_l.reshape(shape_Z_l)
            
            # backward flow of derivatives through pooling operation 
            #dA_l=pool_backward(dA_l,layer_cache, layer_architecture)
            dA_l=pool_backward_using_for(dA_l,layer_cache, layer_architecture)
            
            # backward flow of derivatives through relu operation 
            Z_l=layer_cache['Zconv']
            g_prime_l=activation_prime(Z_l,'relu')
            dZ_l=dA_l*g_prime_l
            
            # backward flow of derivatives through convolution operation 
            #dA_l,dW_l, db_l=conv_backward(dZ_l,layer_cache,layer_architecture)
            dA_l,dW_l, db_l=conv_backward_using_for(dZ_l,layer_cache,layer_architecture)
            
        grads['layer'+str(l+1)]={'Type':layer_type,'dW':dW_l, 'db':db_l}
                
    return grads

#%% Update parameters using grad descent
def update_weights(weights,grads,learn_rate):
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
        weights_l['W']=weights_l['W']-learn_rate*grads_l['dW']
        weights_l['b']=weights_l['b']-learn_rate*grads_l['db']
        
    return weights
    
#%% Main model
def CNN_model(X,Y,architecture,no_epochs,batch_size,learn_rate,regularization=None,optimization=None):
    
    #num of examples
    m=len(Y)
    
    
    # Entering input layer info in the architecture
    inp_layer=architecture[0]
    inp_layer.update({'Input_shape':(X.shape[1],X.shape[2],X.shape[3])})
    
    # Determine no of classes
    Y=Y.reshape((1,m))
    classes=list(set(Y[0,:]))
    n_classes=len(classes)
    
    
    # one hot encoding for more than one class for more than two classes
    op_layer=architecture[-1]
    Y=Y.reshape(1,m)
    act_fl= op_layer['Activation']
    if act_fl=="softmax" or n_classes>2:
        Y=np.repeat(Y, n_classes,axis=0)
        Y=Y==np.arange(0,n_classes).reshape(n_classes,1)
    else:
        n_classes=1
        
    # Entering output layer info in the architecture
    op_layer.update({'Num_output_units':n_classes})
    
    # Initialize weights
    weights=initialize_weights(architecture)
    
    # Training loop       
    cost_list=[]
    epoch_list=[]
    global_itn=0
    cur_epoch_no=0
    
    while cur_epoch_no<=no_epochs:
        batches_list=create_mini_batches(X,Y,batch_size)
        itn_no=0
        
        for batch in batches_list:
            # Unpack batch
            batch_X=batch[0]
            batch_Y=batch[1]
            
            #Forward prop
            cache=forward_prop(batch_X,weights,architecture)
            
            #Cost calculation
            batch_Y_hat=cache['final activation']
            cost=calc_cost(batch_Y,batch_Y_hat,act_fl)
            
            #save cost value
            cost_list.append(cost)
            epoch_list.append(cur_epoch_no)
            
            #if cur_epoch_no % 10== 0 and itn_no==0:
            print("Cost after iteration {}: {}".format(global_itn, round(cost,3)))
#            if cur_epoch_no % 100 == 0 and itn_no==0:
#                print("Cost after epoch {}: {}".format(cur_epoch_no, round(cost/n_classes,3)))
            
            #Back prop
            grads=back_prop(batch_Y,cache,architecture)
            
            #Update weights
            weights=update_weights(weights,grads,learn_rate)
            itn_no+=1
            global_itn+=1
        cur_epoch_no+=1
    
    # Plot cost function
    plt.figure(figsize=(5,5), dpi= 100, facecolor='w', edgecolor='k')    
    plt.plot(epoch_list,cost_list)
    plt.ylabel("J")
    plt.xlabel("Epochs (x 100)")
    return weights, cost_list       
    