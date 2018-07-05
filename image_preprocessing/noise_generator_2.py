# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 00:52:32 2018

@author: Davit
"""
from scipy import misc
import numpy as np

def noise_generator(image):
    image = image + 3 * image.std() * np.random.random(image.shape)
    
    return image
