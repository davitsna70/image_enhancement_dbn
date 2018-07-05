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
import pandas as pd

dims = [61,51,41,31,21]

num_epoch = 100

num_batch = 500

k = 1

pretrain_lr = 0.1


for dim in dims:

    datasets = dataset_split("dataset/Numpy/noise 1/data_dim_"+str(dim)+"_noise_0_save.npy")

    result = run_DBN_dim(datasets=datasets, n_ins=dim*dim, n_outs=dim*dim, layers=[dim*dim, dim*dim], pretraining_epochs=num_epoch, batch_size=num_batch, k = k, pretrain_lr = pretrain_lr)
    
    conf = 'dim_'+str(dim)+'_layer_'+str(dim*dim)+'_'+str(dim*dim)+'_epoch_'+str(num_epoch)+'_batch_'+str(num_batch)+'_K_'+str(k)+'_pretrain_lr_'+str(pretrain_lr)
    
    #save model
    with open('models_dbn/dbn_models_'+conf+'.pkl', 'wb') as f:
        pickle.dump(result[0], f)
    
    #save history
    numpy.save('models_dbn/history_numpy_pretrain_'+conf+'.npy',result[1])
    
    df_hist_model = pd.DataFrame(result[1])
    df_hist_model.to_csv('models_dbn/history_csv_pretrain_'+conf+'.csv')
    
    
    #save range time
    file_range_time = open('models_dbn/range_time_models_'+conf+'.txt', 'w')
    
    file_range_time.write("Range time : "+str(result[2])+" Minutes")
    
    file_range_time.close()

