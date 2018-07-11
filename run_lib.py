# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 01:51:37 2018

@author: Davit
"""
from __future__ import print_function, division
import os
import sys
import timeit
import numpy

import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras import metrics


from deep_belief_network.DBN import DBN

def run_dbn(datasets, n_ins, n_outs, layers=[100,100,100], pretraining_epochs=100, 
            batch_size=10, pretrain_lr=0.01, k=1):

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size

    # numpy random generator
    numpy_rng = numpy.random.RandomState(123)
    print('... building the model')
    # construct the Deep Belief Network
    dbn = DBN(numpy_rng=numpy_rng, n_ins=n_ins,
              hidden_layers_sizes=layers,
              n_outs=n_outs)

    # start-snippet-2
    #########################
    # PRETRAINING THE MODEL #
    #########################
    print('... getting the pretraining functions')
    pretraining_fns = dbn.pretraining_functions(train_set_x=train_set_x,
                                                batch_size=batch_size,
                                                k=k)

    print('... pre-training the model')
    start_time = timeit.default_timer()
    # Pre-train layer-wise
    history_layer_pre_training = []
    for i in range(dbn.n_layers):
        # go through pretraining epochs
        history_epoch_pre_training = []
        for epoch in range(pretraining_epochs):
            # go through the training set
            c = []
            for batch_index in range(n_train_batches):
                c.append(pretraining_fns[i](index=batch_index,
                                            lr=pretrain_lr))
            print('Pre-training layer %i, epoch %d, cost ' % (i, epoch), end=' ')
            print(numpy.mean(c, dtype='float64'))
            history_epoch_pre_training.append({'epoch':epoch, 'cost':numpy.mean(c, dtype='float64'),'cost_minibatch':numpy.asarray(c)})
        history_layer_pre_training.append(numpy.asarray(history_epoch_pre_training))

    end_time = timeit.default_timer()
    # end-snippet-2
    print('The pretraining code for file ' + os.path.split(__file__)[1] +
          ' ran for %.2fm' % ((end_time - start_time) / 60.), file=sys.stderr)
    
    range_time_pre_training = ((end_time - start_time) / 60.)
    
    return [dbn, numpy.asarray(history_layer_pre_training), range_time_pre_training]



def run_fine(datasets, weights, input, output, layers_dbn, layers_fine, activation_fine, epoch_fine, batch_fine, lr_fine):
    
    X_train, y_train, X_val, y_val, X_test, y_test = datasets
    
    time_start = timeit.default_timer()
    model = Sequential()
    
    for i in range(len(layers_dbn)):
        if i == 0:
            model.add(Dense(layers_dbn[i], input_dim=input, activation='sigmoid'))
        else:
            model.add(Dense(layers_dbn[i], activation='sigmoid'))
    
    for i in range(len(layers_fine)):
        model.add(Dense(layers_dbn[i], activation=activation_fine[i]))
    
    model.add(Dense(output, activation='sigmoid'))
    
    #set weights from pre trained (DBN wake sleep learning)
    model.set_weights(weights)
    
    model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.SGD(lr=lr_fine), metrics=['accuracy', metrics.mse, metrics.mape, metrics.msle, metrics.mae])
    
    history = model.fit(X_train, y_train, epochs=epoch_fine, batch_size=batch_fine, validation_data=(X_val, y_val))
    
    time_end = timeit.default_timer()
    
    range_time = ((time_end-time_start)/60.)
    
    return [model, history, range_time]
    

def run_autoencoder_dnn(datasets, input, output, layers, activation, epoch, batch, lr):
    
    X_train, y_train, X_val, y_val, X_test, y_test = datasets
    
    time_start = timeit.default_timer()
    model = Sequential()
    
    for i in range(len(layers)):
        if i==0:
            model.add(Dense(layers[i], input_dim=input, activation=activation[i]))
        else:
            model.add(Dense(layers[i], activation=activation[i]))
    
    for i in reversed(range(len(layers))):
        model.add(Dense(layers[i], activation=activation[i]))
    
    model.add(Dense(output, activation='sigmoid'))
    
    model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.SGD(lr=lr), metrics=['accuracy', metrics.mse, metrics.mape, metrics.msle, metrics.mae])
    
    history = model.fit(X_train, y_train, epochs=epoch, batch_size=batch, validation_data=(X_val, y_val))
    
    time_end = timeit.default_timer()
    
    range_time = ((time_end-time_start)/60.)
    
    return [model, history, range_time]


def run_autoencoder_conv(datasets, dim, input, output, layers, activation, epoch, batch, lr):
    
    X_train, y_train, X_val, y_val, X_test, y_test = datasets
    
    if len(X_train.shape) < 4:
        X_train = numpy.reshape(X_train, (X_train.shape[0], dim, dim, 1))
        y_train = numpy.reshape(y_train, (y_train.shape[0], dim, dim, 1))
        X_val = numpy.reshape(X_val, (X_val.shape[0], dim, dim, 1))
        y_val = numpy.reshape(y_val, (y_val.shape[0], dim, dim, 1))
        X_test = numpy.reshape(X_test, (X_test.shape[0], dim, dim, 1))
        y_test = numpy.reshape(y_test, (y_test.shape[0], dim, dim, 1))
    
    time_start = timeit.default_timer()
    model = Sequential()
    
    for i in range(len(layers)):
        if i==0:
            model.add(Conv2D(layers[i], (3, 3), padding='same', activation=activation[i], input_shape=(dim,dim,1)))
            model.add(MaxPooling2D((2, 2), padding='same'))
        else:
            model.add(Conv2D(layers[i], (3, 3), padding='same', activation=activation[i]))
            model.add(MaxPooling2D((2, 2), padding='same'))
    
    for i in reversed(range(len(layers))):
        model.add(Conv2D(layers[i], (3, 3), padding='same', activation=activation[i]))
        model.add(UpSampling2D((2, 2)))
        
    model.add(Conv2D(1, (3, 3), padding='same', activation='sigmoid'))
    
    model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.SGD(lr=lr), metrics=['accuracy', metrics.mse, metrics.mape, metrics.msle, metrics.mae])
    
    history = model.fit(X_train, y_train, epochs=epoch, batch_size=batch, validation_data=(X_val, y_val))
    
    time_end = timeit.default_timer()
    
    range_time = ((time_end-time_start)/60.)
    
    return [model, history, range_time]