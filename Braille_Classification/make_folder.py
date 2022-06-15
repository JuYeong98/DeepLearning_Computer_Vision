import os
import numpy as np
import pandas as pd
from shutil import copyfile
num_list  = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','opening','closing']

#os.mkdir('./images2/')

for i in range(len(num_list)): 
    os.mkdir('./images2/' + num_list[i])
   
rootdir = '/workspace/NIA/NIA_AI_DATASET_2021-ST-GCAE/JuYeong/Deeplearning_Computer_VIsion/Deeplearning_Competition/AI_Connect_Competition/Competition/Braille Dataset/Braille Dataset2/'

for file in os.listdir(rootdir):
    letter = file.split('_')[0]    
    copyfile(rootdir+file, './images2/' + letter + '/' + file)   
