# -*- coding: utf-8 -*-
"""
Created on Sun May 20 13:49:27 2018

@author: Davit
"""
#import library
"""build in lib"""
import os, sys
import glob
import timeit

"""image processing"""
import cv2 as cv
import pandas as pd
import numpy as np
import numpy
from matplotlib import pyplot as plt
from matplotlib import image as mpimg

#from skimage import data
from skimage.util.dtype import dtype_range
from skimage.util import img_as_ubyte
from skimage import exposure
from skimage.morphology import disk
from skimage.filters import rank

"""keras lib"""
#import keras
#from keras import models
import tensorflow as tf

"""theano lib"""
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams

"""built lib"""
from deep_belief_network.logistic_sgd import LogisticRegression, load_data
from deep_belief_network.mlp import HiddenLayer
from deep_belief_network.rbm import RBM

from image_preprocessing.noise_generator_1 import noise_generator as noise_generator

#loading the image from a directory
def load_image(directory):
    return cv.imread(directory)

#save image data into a directory with a specific name
def save_image(data, directory):
    cv.imwrite(data, directory)
    
    return True

#load multiple image from a directory
def load_images(directory):
    dir_list = glob.glob(directory+'*.jpg')
    data_images = []
    for dir_file in dir_list:
        image = load_image(dir_file)
        data_images.append(image)
        
    return data_images

#save multiple image into directory
def save_images(datas, directory):
    idx = 0
    for data_image in datas:
        save_image(data_image, directory+'/'+idx+'.jpg')
        idx += 1
        
    return True

#showing an image using matplotlib
def show_image(data, mode = None):
    if(len(data.shape)==3):
        data = cv.cvtColor(data, cv.COLOR_BGR2RGB)
    else:
        mode = 'gray'
        
    if(mode == None):
        plt.imshow(data)
        plt.xticks([]), plt.yticks([])
        plt.show()
    else:
        plt.imshow(data, cmap = mode)
        plt.xticks([]), plt.yticks([])
        plt.show()
        
    return True
    
#showing multiple images using matplotlib
def show_images(datas, mode = None):
    for data in datas:
        show_image(data, mode)
    
    return True

#definition function that return a kernel list
def kernel_filtering_modes(mode):
    identity = np.array(
                [[0,0,0],
                [0,1,0],
                [0,0,0]])
    edge_detection_1 = np.array(
                        [[1,0,-1],
                        [0,0,0],
                        [-1,0,1]])
    edge_detection_2 = np.array(
                        [[0,1,0],
                        [1,-4,1],
                        [0,1,0]])
    edge_detection_3 = np.array(
                        [[-1,-1,-1],
                        [-1,8,-1],
                        [-1,-1,-1]])
    sharpen = np.array(
                [[0,-1,0],
               [-1,5,-1],
               [0,-1,0]])
    box_blur = np.array(
                [[1,1,1],
                [1,1,1],
                [1,1,1]])/9
    gaussian_blur_3_3 = np.array(
                        [[1,2,1],
                        [2,4,2],
                        [1,2,1]])/16
    gaussian_blur_5_5 = np.array(
                        [[1,4,6,4,1],
                        [4,16,24,16,4],
                        [6,24,36,24,6],
                        [4,16,24,16,4],
                        [1,4,6,4,1]])/256
    unsharp_masking = np.array(
                        [[1,4,6,4,1],
                        [4,16,24,16,4],
                        [6,24,-476,24,6],
                        [4,16,24,16,4],
                        [1,4,6,4,1]])/-256
    dict_modes = {
            'identity' : identity,
            'edge_detection_1' : edge_detection_1,
            'edge_detection_2' : edge_detection_2,
            'edge_detection_3' : edge_detection_3,
            'sharpen' : sharpen,
            'box_blur' : box_blur,
            'gaussian_blur_3_3' : gaussian_blur_3_3,
            'gaussian_blur_5_5' : gaussian_blur_5_5,
            'unsharp_masking' : unsharp_masking ,
            }
    
    return dict_modes[mode]

#change image into a mode filtering image
def convolution_image(data, mode):
    if(mode == 'gaussian_blur_ori_3_3'):
        new_image = cv.GaussianBlur(data, (3,3), 0)
    elif(mode == 'gaussian_blur_ori_5_5'):
        new_image = cv.GaussianBlur(data, (5,5), 0)
    elif(mode == 'median_blur_3_3'):
        new_image = cv.medianBlur(data,3)
    elif(mode == 'median_blur_5_5'):
        new_image = cv.medianBlur(data,5)
    elif(mode == 'bilateral'):
        new_image = cv.bilateralFilter(data,9,75,75)
    elif(mode == 'grayscale'):
        new_image = cv.cvtColor(data, cv.COLOR_BGR2GRAY)
    else:
        kernel = kernel_filtering_modes(mode)
        new_image = cv.filter2D(data, -1, kernel)
    
    return new_image

#change a image into varian filtering modes image
def convolutions_image(data, modes):
    image = data
    for mode in modes:
        image = convolution_image(image, mode)
        
    return image

#change multiple image into varian filtering modes image
def convolutions_images(datas, modes):
    images = []
    for image in datas:
        image = convolutions_image(image, modes)
        images.append(image)
    
    return images

# Global equalize
def glob_equalization(image):
    img_rescale = exposure.equalize_hist(image)
    
    return img_rescale

# Local Equalization
def local_equalization(image, radius_disk = 30):
    selem = disk(radius_disk)
    img_eq = rank.equalize(image, selem=selem)
    
    return img_eq

#separate image into new dimentional images
def separator_image_into(data, dimension = 3):
    
    windows = []
    num_idx_col = int(len(data)/dimension)
    num_idx_row = int(len(data[0])/dimension)
    # Crop out the window and calculate the histogram
    for idx_col in range(num_idx_col):
        for idx_row in range(num_idx_row):
            windows.append(data[idx_col*dimension:(idx_col+1)*dimension, idx_row*dimension:(idx_row+1)*dimension])
    
    return windows

#separate images into new dimentional images
def separator_images_into(datas, dimension = 3):
    windows_images = []
    for data in datas:
        windows_images.append(separator_image_into(data, dimension))
    
    return windows_images

"""
#load image
image = load_image("data_rocto2.jpg")
show_image(image, title='Origin')
save_image(image, 'origin.jpg')
#grayscaling
image = convolution_image(image, 'grayscale')
show_image(image, 'gray', title='Grayscale')
save_image(image, 'gray1.jpg')
#bluring
image = convolution_image(image,'gaussian_blur_3_3')
show_image(image, 'gray', title='Gaussian Blur')
save_image(image, 'blur1.jpg')
#sharpen
image = convolution_image(image,'sharpen')
show_image(image, 'gray', title='Sharpenning')
save_image(image, 'sharp1.jpg')

#unsharpening
image = convolution_image(image,'unsharp_masking')
show_image(image, 'gray', title='Unsharp')
save_image(image, 'unsharp1.jpg')

#threshold
(tresh, image) = cv.threshold(image, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
show_image(image, 'gray', title='Threshold')
save_image(image, 'thres1.jpg')

#local equalizing
image = local_equalization(image, radius_disk=100)
show_image(image, 'gray', title='Local Histogram Equalization')
save_image(image, 'local_hist1.jpg')
#bluring
image = convolution_image(image,'gaussian_blur_ori_3_3')
show_image(image, 'gray', title='Gaussian Blur')
save_image(image, 'blur2.jpg')
#sharpen
image = convolution_image(image,'sharpen')
show_image(image, 'gray', title='Sharpenning')
save_image(image, 'sharp2.jpg')
#global equalization
image = glob_equalization(image)
show_image(image, 'gray', title='Local Histogram Equalization')
save_image(image, 'glob_hist1.jpg')
"""
"""
url_image = "dataset/image/All Gambar Rontgen/01 (2).jpg"
origin_image = load_image(url_image)

noise_image = load_image(url_image)
noise_image = noise_generator("s&p", noise_image)
noise_image = noise_generator("gauss", noise_image)

grayscale_origin_image = convolution_image(origin_image, "grayscale")
grayscale_noise_image = convolution_image(noise_image, "grayscale")

show_image(origin_image)
show_image(noise_image)

glob_equalization_image = glob_equalization(grayscale_noise_image)
local_equalization_image = local_equalization(grayscale_noise_image)

show_images([glob_equalization_image, local_equalization_image], ["gray","gray"])

data_load = numpy.load("dataset/Numpy/noise 1/data_dim_61_save.npy")

X,y = data_load

X_0 = X[1]
y_0 = y[1]

show_image(X_0)
show_image(y_0)

"""