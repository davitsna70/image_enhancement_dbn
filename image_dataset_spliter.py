# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 14:22:25 2018

@author: Davit
"""

import numpy as np
from sklearn.model_selection import train_test_split
import theano
import theano.tensor as T

def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, shared_y

def dataset_loader(directory):
    print("...Dataset in load process...")
    return np.load(directory)

def dataset_converter(dataset):
    data_X = []
    data_y = []
    origin_X, origin_y = dataset
    
    print("...Dataset in convert process...")
    for X in origin_X:
        data_X.append(X.flatten())
    
    for y in origin_y:
        data_y.append(y.flatten())
    print("...Dataset finish convert...")
        
    convert_dataset = [np.asarray(data_X)/1., np.asarray(data_y)/255.]
    
    return np.asarray(convert_dataset)

def dataset_split(directory, train = 0.8, test = 0.2):
    #load dataset
    dataset = dataset_loader(directory)
    #convert dataset into 1 dimension image by flatten
    dataset = dataset_converter(dataset)
    
    X, y = dataset
    
    print("...Dataset in spliting process...")
    #split dataset into train, validation, test
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = train, test_size=test, random_state=4)
    
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, train_size = 0.8, test_size=0.2, random_state=4)
    print("...Dataset has been splited...")
    
    test_set_x, test_set_y = shared_dataset([X_test, y_test])
    valid_set_x, valid_set_y = shared_dataset([X_val, y_val])
    train_set_x, train_set_y = shared_dataset([X_tr, y_tr])

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval
        