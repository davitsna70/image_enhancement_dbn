# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 09:02:08 2018

@author: Davit
"""
from scipy import misc
import numpy as np

def noise_generator(image):
    alot = 2 * image.max() * np.random.random(image.shape)
    image = image + alot
    
    return image