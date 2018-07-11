# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 20:45:33 2018

@author: Davit
"""

from image_processor import load_image
import pandas as pd
import numpy
import cv2

#%%
img_3d = load_image("dataset/Image/All Gambar Rontgen/03 (3).jpg")

df_3d_b = pd.DataFrame(img_3d[:,:,0])

df_3d_g = pd.DataFrame(img_3d[:,:,1])

df_3d_r = pd.DataFrame(img_3d[:,:,2])

#%%
df_3d_b.to_csv("result_report/Grayscaling/df_3d_b.csv")

df_3d_g.to_csv("result_report/Grayscaling/df_3d_g.csv")

df_3d_r.to_csv("result_report/Grayscaling/df_3d_r.csv")

#%%

#grayscale counting

file_grayscale_calculation = open("result_report/Grayscaling/grayscale_calculation_all.txt", "w")

str_grayscale_calculation = "Grayscale = (0.2126*Red + 0.7152*Green + 0.0722*Blue)/3\n\n"

list_grayscale_calculation = []

for i in range(img_3d.shape[0]):
    temp_list = []
    for j in range(img_3d.shape[1]):
        sum_value = 0.2126*img_3d[i][j][0] + 0.7152*img_3d[i][j][1] + 0.0722*img_3d[i][j][2]
        mean = sum_value/3.
        str_grayscale_calculation+="Image["+str(i)+"]"+"["+str(j)+"] = (0.2126*"+str(img_3d[i][j][2])+" + 0.7152*"+str(img_3d[i][j][1])+" + 0.0722*"+str(img_3d[i][j][0])+")/3\n"
        str_grayscale_calculation+=" = "+str(sum_value)+"/3\n"
        str_grayscale_calculation+=" = "+str(mean)+"\n"
        str_grayscale_calculation+=" = "+str(int(mean))+"\n\n"
        
        temp_list.append(int(mean-1))
    temp_list = numpy.asarray(temp_list)
    list_grayscale_calculation.append(temp_list)
#%%    
list_grayscale_calculation = numpy.asarray(list_grayscale_calculation)

df_list_grayscale = pd.DataFrame(list_grayscale_calculation)

df_list_grayscale.to_csv("result_report/Grayscaling/grayscale_calculation_all.csv")

file_grayscale_calculation.write(str_grayscale_calculation)

file_grayscale_calculation.close()

cv2.imwrite("result_report/Grayscaling/image_all.jpg", img_3d)

cv2.imwrite("result_report/Grayscaling/image_all_grayscale.jpg", list_grayscale_calculation)
#%%

#for 50 * 50

img_3d_50 = img_3d[150:200,150:200]

df_3d_b = pd.DataFrame(img_3d_50[:,:,0])

df_3d_g = pd.DataFrame(img_3d_50[:,:,1])

df_3d_r = pd.DataFrame(img_3d_50[:,:,2])

df_3d_b.to_csv("result_report/Grayscaling/df_3d_b_50.csv")

df_3d_g.to_csv("result_report/Grayscaling/df_3d_g_50.csv")

df_3d_r.to_csv("result_report/Grayscaling/df_3d_r_50.csv")

file_grayscale_calculation = open("result_report/Grayscaling/grayscale_calculation_50.txt", "w")

str_grayscale_calculation = "Grayscale = (0.2126*Red + 0.7152*Green + 0.0722*Blue)/3\n\n"

list_grayscale_calculation = []

for i in range(img_3d_50.shape[0]):
    temp_list = []
    for j in range(img_3d_50.shape[1]):
        sum_value = 0.2126*img_3d_50[i][j][0] + 0.7152*img_3d_50[i][j][1] + 0.0722*img_3d_50[i][j][2]
        mean = sum_value/3.
        str_grayscale_calculation+="Image["+str(i)+"]"+"["+str(j)+"] = (0.2126*"+str(img_3d_50[i][j][2])+" + 0.7152*"+str(img_3d_50[i][j][1])+" + 0.0722*"+str(img_3d_50[i][j][0])+")/3\n"
        str_grayscale_calculation+=" = "+str(sum_value)+"\n"
        str_grayscale_calculation+=" = "+str(mean)+"\n"
        str_grayscale_calculation+=" = "+str(int(mean))+"\n\n"
        
        temp_list.append(int(mean))
    temp_list = numpy.asarray(temp_list)
    list_grayscale_calculation.append(temp_list)
    
list_grayscale_calculation = numpy.asarray(list_grayscale_calculation)

df_list_grayscale = pd.DataFrame(list_grayscale_calculation)

df_list_grayscale.to_csv("result_report/Grayscaling/grayscale_calculation_50.csv")

file_grayscale_calculation.write(str_grayscale_calculation)

file_grayscale_calculation.close()

cv2.imwrite("result_report/Grayscaling/image_50.jpg", img_3d_50)

cv2.imwrite("result_report/Grayscaling/image_50_grayscale.jpg", list_grayscale_calculation)

#%%





#%%

#