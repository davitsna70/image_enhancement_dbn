# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 20:09:57 2018

@author: Davit
"""


import numpy as np
import glob
import cv2 as cv
import matplotlib.pyplot as plt

from .noise_generator_1 import noise_generator
import image_processor

def get_and_seperate_data_X_y(directory, dimension, modes_convolution = None):
    dir_list = glob.glob(directory+'*.jpg')
    iterator = 1
    
    dataset_X = []
    dataset_y = []
    
    for dir_file in dir_list:
        origin_image = image_processor.load_image(dir_file)
        
        noise_image = noise_generator("s&p", origin_image)
        noise_image = noise_generator("gauss", noise_image)
        
        #convolution
        if modes_convolution == None:
            noise_image = image_processor.convolutions_image(noise_image,modes=["gaussian_blur_5_5", "sharpen"])
        else:
            noise_image = image_processor.convolutions_image(noise_image, modes=modes_convolution)

        gray_origin_image = cv.cvtColor(origin_image, cv.COLOR_BGR2GRAY)
        gray_noise_image = cv.cvtColor(noise_image, cv.COLOR_BGR2GRAY)

        num_idx_col = int(len(gray_origin_image)/dimension)
        num_idx_row = int(len(gray_origin_image[0])/dimension)
        
        for idx_col in range(num_idx_col):
            for idx_row in range(num_idx_row):
                X = gray_noise_image[idx_col*dimension:(idx_col+1)*dimension, idx_row*dimension:(idx_row+1)*dimension]
                y = gray_origin_image[idx_col*dimension:(idx_col+1)*dimension, idx_row*dimension:(idx_row+1)*dimension]
                dataset_X.append(X)
                dataset_y.append(y)
        
        print("...image number "+str(iterator)+" has been processing")
        iterator += 1
        
    print("***DATASET IMAGE HAS BEEN DONE PROCESSED***")
    return [dataset_X, dataset_y]

#load image and set into X, y data
dimensions = [3,5,7,9,11,13,15,21,31,41,51,61]
for dimension in dimensions:
    directory = "D:/11S14020_DAVIT SYAHPUTRA NAPITUPULU/KULIAH/Semester 8/TA/Document/Code/On Working_Ver 1/All Gambar Rontgen/"
    dataset = get_and_seperate_data_X_y(directory, dimension)
    
    
    #save data as numpy
    dataset_type_numpy = np.asarray(dataset)
    np.save("data_dim_"+str(dimension)+"_noise_1_save.npy",dataset_type_numpy)
    
    print("data has been saved as numpy dimmension "+str(dimension)+"...")