# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 20:14:26 2018

@author: Davit
"""

import gzip
from six.moves import cPickle as pickle
import numpy

from model_lib import model_autoencoder_convolution
from image_dataset_spliter import dataset_split

#%%
dims = [61,51,41,31,21]

num_epoch = 100

num_batch = 500

k = 1

lr = 0.1

dim = 61

datasets = dataset_split("dataset/Numpy/noise 1/data_dim_"+str(dim)+"_noise_0_save.npy")

name = "models_autoencoder_conv/conv_testing"

result = model_autoencoder_convolution(name, datasets, dim, 
                               layers=[60], 
                               epoch=2, batch=num_batch, lr=lr)














#%%
name_dir = 'mnist.pkl.gz'

dims = [61,51,41,31,21]

num_epoch = 100

num_batch = 100

k = 1

lr = 0.1

dim = 28

with gzip.open(name_dir, 'rb') as f:
    try:
        train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
    except:
        train_set, valid_set, test_set = pickle.load(f)
        

X_train, y_train = train_set[0], train_set[0]
X_val, y_val = valid_set[0], valid_set[0]
X_test, y_test = test_set[0], test_set[0]

train = numpy.asarray([X_train, X_train])
val = numpy.asarray([X_val, X_val])
test = numpy.asarray([X_test, X_test])

new_dataset = [train, val, test]

#%%
result = model_autoencoder_convolution(name, new_dataset, dim, 
                               layers=[14,7], 
                               epoch=2, batch=num_batch, lr=lr)

#%%

from keras.models import load_model
from matplotlib import pyplot as plt


conv_model = load_model('models_autoencoder_conv/conv_testing_autoencode_conv_model.h5')

data = numpy.reshape(X_test[0],(28,28))

plt.imshow(data, cmap='gray')
plt.show()
#%%
new_data = numpy.reshape([data], (1,28,28,1))
data_pred = conv_model.predict(new_data)

new_data_pred = numpy.reshape(data_pred, (28,28))
plt.imshow(new_data_pred, cmap='gray')
plt.show()