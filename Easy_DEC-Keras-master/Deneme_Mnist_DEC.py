# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 17:45:18 2021

@author: Kutay
"""


from keras_dec import DeepEmbeddingClustering
from keras.datasets import mnist
import numpy as np

'''
x_train, x_test: uint8 arrays of grayscale image data with shapes (num_samples, 28, 28).

y_train, y_test: uint8 arrays of digit labels (integers in range 0-9) with shapes (num_samples,).
'''

def get_mnist():
    np.random.seed(1234) # set seed for deterministic ordering
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_all = np.concatenate((x_train, x_test), axis = 0)
    Y = np.concatenate((y_train, y_test), axis = 0)
    X = x_all.reshape(-1,x_all.shape[1]*x_all.shape[2])
    
    p = np.random.permutation(X.shape[0])
    X = X[p].astype(np.float32)*0.02
    Y = Y[p]
    return X, Y


X, Y  = get_mnist() 

c = DeepEmbeddingClustering(n_clusters=10, input_dim=784)
c.initialize(X, finetune_iters=20, layerwise_pretrain_iters=40)
c.cluster(X, y=Y, iter_max=10)





acc_out = c.cluster_acc(Y,c.y_pred)
pred_value = c.y_pred
output = c.encoder.predict(X)

