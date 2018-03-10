# deeplearning
Hello,

This is a self written deep learning repository to gain a better understanding of deep learning. The first library is called  
L_layer_NN which  is a set of functions to implement a L layered neural network (L includes input and output layers). Most of the formulations of the matrices have been taken from the notes of Andrew Ng's awesome deeplearning course (https://www.deeplearning.ai/). I have also included  python scripts on how to use the library using toy datasets. I will keep on updating this library with additional features.   

L_layer_NN has now been tested and the version history is as follows:

Version history:

L_layer_NN (V1.0)
- Binary classification problems
- Options for L2 and drop-out regularization
- Numerical verification of gradients computed in back propagation

L_layer_NN (V1.1)
- Extended to multi-class problems using one-vs-all logistic regression in the output layer
- Implementation of mini batch gradient descent
- Implementation of momentum and adam optimization

L_layer_NN (V1.2)
- Implemented softmax in the final layer
- Implemented batch normalization

I am now dveloping a library to implement a simple convolutional neural network (CNN) called conv_NN of the type conv layer-> fully connected layer -> output layer. Each conv layer has the following units: convolution+relu+Pooling. There are some bugs currently and I am working on evening them out. 
