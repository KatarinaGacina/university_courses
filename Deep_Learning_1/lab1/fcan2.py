import numpy as np
import matplotlib.pyplot as plt
import random

import data

import pdb
import IPython

def relu(x):
   return np.maximum(0., x)

def softmax(x):
    max_x = np.max(x, axis=1, keepdims=True)
    exp_x_shifted = np.exp(x - max_x)

    return exp_x_shifted / np.sum(exp_x_shifted, axis=1, keepdims=True)


def fcan2_train(X, Y_, hidden, param_niter, param_delta, param_lambda):
    num_classes = len(np.unique(Y_))
    num_params = X.shape[1]

    Y = np.zeros((Y_.shape[0], num_classes))
    Y[np.arange(Y_.shape[0]), Y_] = 1

    w1 = np.random.randn(num_params, hidden)
    b1 = np.zeros((1, hidden))
    w2 = np.random.randn(hidden, num_classes)
    b2 = np.zeros((1, num_classes))

    for i in range(param_niter):
        s1 = (X @ w1) + b1
        h1 = relu(s1)
        s2 = (h1 @ w2) + b2
        p = softmax(s2)

        reg = param_lambda * (np.sum(np.square(w1)) + np.sum(np.square(w2)))
        loss  = (-np.sum(np.log(p[range(X.shape[0]), Y_] + 1e-13)) / X.shape[0]) + reg

        if i % 10 == 0:
            print("iteration {}: loss {}".format(i, loss))

        delta = (p - Y) / len(X)
        dw2 = h1.T @ delta + param_lambda * 2 * w2
        db2 = np.sum(delta, axis=0)
        dL_s1 = (delta @ w2.T) * (s1 > 0)
        dw1 = X.T @ dL_s1 + param_lambda * 2 * w1
        db1 = np.sum(dL_s1, axis=0)

        w1 -= (param_delta * dw1)
        b1 -= (param_delta * db1)
        w2 -= (param_delta * dw2)
        b2 -= (param_delta * db2)

    return w1, b1, w2, b2


def fcan2_classify(X, w1, b1, w2, b2):
    s1 = (X @ w1) + b1
    h1 = relu(s1)
    s2 = (h1 @ w2) + b2
    p = softmax(s2)

    return np.argmax(p, axis=1)


if __name__=="__main__":
    np.random.seed(100)

    K=6
    C=2
    N=10

    X, Y_ = data.sample_gmm_2d(K, C, N)

    param_niter=100000
    param_delta=0.05
    param_lambda=1e-3

    hidden = 5
    w1, b1, w2, b2 = fcan2_train(X, Y_, hidden, param_niter, param_delta, param_lambda)

    rect = (np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(lambda x: fcan2_classify(x, w1, b1, w2, b2), rect, offset=0)

    Y = fcan2_classify(X, w1, b1, w2, b2)
    data.graph_data(X, Y_, Y) 
  
    plt.show()

