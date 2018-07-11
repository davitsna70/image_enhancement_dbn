# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 20:14:25 2018

@author: Davit
"""

from model_lib import model_fine_tune
from image_dataset_spliter import dataset_split, shared_dataset
#%%
dims = [61,51,41,31,21]

num_epoch = 100

num_batch = 500

k = 1

pretrain_lr = 0.1

dim = 61

datasets = dataset_split("dataset/Numpy/noise 1/data_dim_"+str(dim)+"_noise_0_save.npy")

name = "models_dbn/dbn_testing"

result = model_fine_tune(name, datasets, dim=dim, layers_dbn=[100], 
                         epoch_dbn=2, batch_dbn=num_batch, lr_dbn=0.1, k_dbn=k,
                         layers_fine=[], activation_fine=[],
                         epoch_fine=2, batch_fine=num_batch, lr_fine=0.1)









#%%
import numpy
import matplotlib.pyplot as plt
import gzip
from six.moves import cPickle as pickle

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

new_dataset = [shared_dataset(train), shared_dataset(val), shared_dataset(test)]

#%%

name = "models_dbn/dbn_testing_mnist"

result = model_fine_tune(name, new_dataset, dim=dim, layers_dbn=[dim*2, dim*3,dim*2], 
                         epoch_dbn=4, batch_dbn=num_batch, lr_dbn=0.1, k_dbn=k,
                         layers_fine=[], activation_fine=[],
                         epoch_fine=10, batch_fine=num_batch, lr_fine=0.1)


#%%

from keras.models import load_model

model_dnn = load_model('models_dbn/dbn_testing_mnist_finetune_model.h5')

data = X_test[0]

data = numpy.reshape(data, (1,data.shape[0]))
#%%
data_predict = model_dnn.predict([data])

#%%
new_data = numpy.reshape(data, (28,28))
new_data_predict = numpy.reshape(data_predict, (28,28))

#%%

plt.imshow(new_data, cmap='gray')
plt.show()
plt.imshow(new_data_predict, cmap='gray')
plt.show()