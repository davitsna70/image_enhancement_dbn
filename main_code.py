# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 14:47:20 2018

@author: Davit
"""

import os
import glob
from image_dataset_spliter import dataset_split
from run_dbn import run_DBN_dim
import numpy

datasets = dataset_split("data_dim_3_save.npy")

def change_dataset(dataset = None):
    train = dataset[0]
    validation = dataset[1]
    test = dataset[2]
    
    X_tr, y_tr = train
    X_val, y_val = validation
    X_test, y_test = test
    
    y_new_tr = []
    y_tr = y_tr.get_value(borrow = True)
    for i in y_tr:
        y_new_tr.append(y_tr[int(len(y_tr)/2)])
    
    y_new_tr = numpy.asarray(y_new_tr)
    
    y_new_val = []
    y_val = y_val.get_value(borrow = True)
    for i in y_val:
        y_new_val
    y_new_val = numpy.asarray(y_new_val)
    
    y_new_test = []
    y_test = y_test.get_value(borrow = True)
    for i in y_test:
        y_new_test.append(y_tr[int(len(y_test)/2)])
    
    y_new_test = numpy.asarray(y_new_test)
    

change_dataset(datasets)

result = run_DBN_dim(datasets=datasets, n_ins=3*3, n_outs=3*3, layers=[5])