# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 20:28:37 2018

@author: Davit
"""

#%%
#Lib
import keras
import cv2
import numpy
import matplotlib.pyplot as plt
import pandas as pd
import theano

from sklearn.metrics import mean_squared_error

from run_lib import run_dbn, run_fine, run_autoencoder_dnn, run_autoencoder_conv
#%%

#helper function
def converter_dataset(datasets):
    if type(datasets[0][0])==theano.tensor.sharedvar.TensorSharedVariable:
        train, val, test = datasets
        
        X_train, y_train = train[0].get_value(borrow=True), train[1].get_value(borrow=True)
        X_val, y_val = val[0].get_value(borrow=True), val[1].get_value(borrow=True)
        X_test, y_test = test[0].get_value(borrow=True), test[1].get_value(borrow=True)
    else:
        train, val, test = datasets
        
        X_train, y_train = train
        X_val, y_val = val
        X_test, y_test = test
    
    return [X_train, y_train, X_val, y_val, X_test, y_test]

def saver(directory, result):
    #save model
    model = result[0]
    model.save(directory+"_model.h5")
    #save history
    history = numpy.asarray([result[1].history['acc'], result[1].history['loss'], result[1].history['val_acc'], result[1].history['val_loss']])
    numpy.save(directory+"_history.npy", history)

    df = pd.DataFrame(history)
    df.to_csv(directory+"_history.csv", sep=";")
    #save time
    time = result[2]
    file = open(directory+"_time.txt", "w")
    file.write("Range time : "+str(time)+" Minutes")
    file.close()

def saver_weight(directory, result):
    #save model
    weights = result[0]
    numpy.save(directory+"_weight.npy", weights)
    #save history
    history = numpy.asarray(result[1])
    numpy.save(directory+"_history.npy", history)
    
    df = pd.DataFrame(history)
    df.to_csv(directory+"_history.csv", sep=";")
    #save time
    time = result[2]
    file = open(directory+"_time.txt", "w")
    file.write("Range time : "+str(time)+" Minutes")
    file.close()

#%%
#function to comparator

def mse(image, new_image):
    return mean_squared_error(image, new_image)
    
###########
def essim(image, new_image):
    return mean_squared_error(image, new_image)

def messim(image, new_image):
    return mean_squared_error(image, new_image)

def psnr(image, new_image):
    return mean_squared_error(image, new_image)

def ssim(image, new_image):
    return mean_squared_error(image, new_image)


#%%
#function making model

#dataset must in theano variable
def model_dbn(directory, datasets, dim, layers=[100,100,100], epoch=10, batch=10, lr=0.01, k=1):
    
    result = run_dbn(datasets, n_ins = dim*dim, 
                     n_outs = dim*dim, layers = layers, 
                     pretraining_epochs = epoch, batch_size = batch, pretrain_lr = lr, k = k)
    
    dbn_model = result[0]
    weights = []
    for layer in dbn_model.sigmoid_layers:
        weights.append(layer.W.get_value(borrow = True))
        weights.append(layer.b.get_value(borrow = True))
    
    result = [weights, result[1], result[2]]
    
    saver_weight(directory=directory+"_dbn", result=result)
    
    return result

#from this dataset musn't in theano variable
def model_fine_tune(directory, datasets, dim, layers_dbn, 
                    epoch_dbn, batch_dbn, lr_dbn, k_dbn, 
                    layers_fine=None, activation_fine=None, 
                    epoch_fine=None, batch_fine = None, lr_fine = None):
    
    result_dbn = model_dbn(directory=directory, datasets=datasets, dim=dim, layers=layers_dbn, epoch=epoch_dbn, batch=batch_dbn, lr=lr_dbn, k=k_dbn)
    
    weights = result_dbn[0]
    
    layers_fine = layers_fine if layers_fine!=None else []
    activation_fine = activation_fine if activation_fine!=None else []
    epoch_fine = epoch_fine if epoch_fine!=None else epoch_dbn
    batch_fine = batch_fine if batch_fine!=None else batch_dbn
    lr_fine = lr_fine if batch_fine!=None else batch_dbn
    
    datasets_convert = converter_dataset(datasets)
    
    result = run_fine(datasets_convert, weights, input=dim*dim, output=dim*dim, 
                      layers_dbn=layers_dbn, layers_fine=layers_fine, 
                      activation_fine=activation_fine, epoch_fine=epoch_fine, 
                      batch_fine=batch_fine, lr_fine=lr_fine)
    
    saver(directory=directory+"_finetune", result=result)
    
    return result

def model_autoencoder_dnn(directory, datasets, dim, layers, activation=None, epoch=100, batch=100, lr=0.01):
    
    activation = activation if activation!=None else ['relu']*len(layers)
    
    datasets_convert = converter_dataset(datasets)
    
    result = run_autoencoder_dnn(datasets_convert, input=dim*dim, output=dim*dim, layers=layers, activation=activation, epoch=epoch, batch=batch, lr=lr)
    
    saver(directory=directory+"_autoencode_dnn", result=result)
    
    return result
    
def model_autoencoder_convolution(directory, datasets, dim, layers, activation=None, epoch=100, batch=100, lr=0.1):
    
    activation = activation if activation!=None else ['relu']*len(layers)
    
    datasets_convert = converter_dataset(datasets)
    
    result = run_autoencoder_conv(datasets_convert, dim=dim, input=dim*dim, output=dim*dim, layers=layers, activation=activation, epoch=epoch, batch=batch, lr=lr)
    
    saver(directory=directory+"_autoencode_conv", result=result)
    
    return result

def testing_with_image(dim, image, model):
    
    #reshape origin image
    image = image[:(image.shape[0]-(image.shape[0]%dim)), :(image.shape[1]-(image.shape[1]%dim))]
    
    new_image = numpy.zeros((image.shape[0], image.shape[1]))
    
    for idx_row in range(int(image.shape[0]/dim)):
        for idx_col in range(int(image.shape[1]/dim)):
            flt_img = numpy.ndarray.flatten(image[idx_row*dim:(idx_row+1)*dim, idx_col*dim:(idx_col+1)*dim])
            flt_img_pred = model.predic(flt_img)
            resh_img = numpy.reshape(flt_img_pred, (dim,dim))
            new_image[idx_row*dim:(idx_row+1)*dim, idx_col*dim:(idx_col+1)*dim] = resh_img
            
    n_mse = mse(image, new_image)
    n_essim = essim(image, new_image)
    n_messim = messim(image, new_image)
    n_psnr = psnr(image, new_image)
    n_ssim = ssim(image, new_image)
    
    return [new_image, n_mse, n_essim, n_messim, n_psnr, n_ssim]

