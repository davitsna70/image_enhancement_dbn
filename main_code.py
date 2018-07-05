# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 14:47:20 2018

@author: Davit
"""

import os
import glob
from image_dataset_spliter import dataset_split
from run_dbn import run_DBN_dim
import theano
from theano import tensor as T
from deep_belief_network.logistic_sgd import LogisticRegression
import timeit
import numpy
import sys
import six.moves.cPickle as pickle
import numpy
import matplotlib.pyplot as plt
import math

#%%

dim = 61

datasets = dataset_split("dataset/Numpy/noise 1/data_dim_"+str(dim)+"_noise_0_save.npy")

#%%
result = run_DBN_dim(datasets=datasets, n_ins=dim*dim, n_outs=dim*dim, layers=[dim*dim], pretraining_epochs=2, batch_size=600)

with open('only_dbn.pkl', 'wb') as f:
    pickle.dump(result[0], f)
    
#%% 
with open('only_dbn.pkl', 'rb') as f:
    dbn = pickle.load(f)
#%%    
    
len(dbn.rbm_layers)

dbn.rbm_layers[0].W.get_value().shape


dbn.sigmoid_layers[-1].output


#%%

train, validation, test = datasets
        
X_train, y_train = train[0].get_value(borrow=True), train[1].get_value(borrow=True)
X_val, y_val = validation[0].get_value(borrow=True), validation[1].get_value(borrow=True)
X_test, y_test = test[0].get_value(borrow=True), test[1].get_value(borrow=True)
        
#%%
index = T.lscalar()

predict_fn = theano.function(
            inputs = [],
            outputs = dbn.output,
            givens={
                    dbn.x : X_test,
                    dbn.y : y_test
                },
            on_unused_input='ignore'
        )

#%%
result_predict = predict_fn()

#%%
n_dim = X_test.shape[0]
temp_x = numpy.reshape(X_test[0], (int(math.sqrt(n_dim))-1, int(math.sqrt(n_dim))-1))

plt.imshow(temp_x, cmap='gray')

plt.show()

temp_result = numpy.reshape(result_predict[0], (int(math.sqrt(n_dim))-1, int(math.sqrt(n_dim))-1))

plt.imshow(temp_result, cmap='gray')

plt.show()
#%%
tmp = numpy.sum(result_predict[0])

print(tmp)

tmp = numpy.mean(result_predict[0])

print(tmp)

#%%
plt.imshow(dbn.rbm_layers[-1].W.get_value())

#%%
dbn.rbm_layers[-1].W.get_value()