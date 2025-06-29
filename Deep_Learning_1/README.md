# Deep Learning 1

These laboratory assigements were completed as part of the *Deep Learning 1* course at college. The laboratory assigements are implemented in Python.

## First laboratory assigement: Basic models:

data.py - for generation of linearly non-separable data  

fcann2.py - implementation of fully connected neural network with 1 hidden layer using numpy  
kswm_wrap.py - class designed as a thin wrapper around the sklearn.svm module, for working with two-dimensional datasets  

pt_linreg.py - linear regression  
pt_logreg.py - logistic regression  
pt_deep.py - configurable deep model  

*Note: All of these files contain main functions that demonstrate usage.*

## Second laboratory assigement: CNN:  

layers.py - layer definitions  
check_grads.py - testing layer gradients

### train.py:  

Dataset: MNIST  

Architecture:  
conv(16,5) -> pool(2,2) -> relu() -> conv(32,5) -> pool(2,2) -> relu() -> flatten() -> fc(512) -> relu() -> fc(10)  

### train_l2reg.py:  

Dataset: MNIST  

Architecture:  
conv(16,5) -> L2Reg() -> pool(2,2) -> relu() -> conv(32,5) -> L2Reg() -> pool(2,2) -> relu() -> flatten() -> fc(512) -> L2Reg() -> relu() -> fc(10)  

### model_z3.py:  

Dataset: MNIST  

Architecture:  
conv(16,5) -> pool(2,2) -> relu() -> conv(32,5) -> pool(2,2) -> relu() -> flatten() -> fc(512) -> relu() -> fc(10)

### model_z4.py:  

Dataset: CIFAR10

Architecture:  
conv(16,5) -> relu() -> pool(3,2) -> conv(32,5) -> relu() -> pool(3,2) -> flatten() -> fc(256) -> relu() -> fc(128) -> relu() -> fc(10)

## Third laboratory assigement: RNN:

Dataset: Stanford Sentiment Treebank  

### zad2.py
Baseline model.    

Architecture:  
avg_pool() -> fc(300, 150) -> ReLU() -> fc(150, 150) -> ReLU() -> fc(150,1)

### zad3.py
Basic RNN model.  

Architecture:  
rnn(150) -> fc(150, 150) -> ReLU() -> fc(150,1)

### zad4.py
Basic RNN model with different RNN components:  
- Vanilla RNN - /
- GRU - gru
- LSTM - lstm

*Note: main function tests model with rnn_layer set to lstm*

### zad5.py
Implementation of Bahdanau attention. 

## Fourth laboratory assigement: Metric embedding:  

This assigement results in a metric embedding model trained with triplet loss. Training data, MNIST dataset, is organized into triplets. The model learns to map the input data into a feature space where similar samples are closer together, and dissimilar ones are further apart. 

zad3.py - example of model training and evaluation

zad4.py - data visualization

