


#coding:utf-8
# First import the package
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from datetime import datetime
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import torchvision.models as models
# This is for the progress bar.
from tqdm import tqdm
import seaborn as sns
from IPython.display import Image
import csv
import cv2
from datetime import datetime 
import random
import torchvision.transforms.functional as F


import sys

# Get command-line parameters
image_date_file = sys.argv[1]
result_path_base = sys.argv[2]


#**** Read the folder and get the number of files
image_date_file = image_date_file
#Get the directory name of the next level of file containing the label name
dirs_up = os.listdir(image_date_file)
#Dimensions of converted NPY file length and width
static_array=[128] 
#Dimensional size of converted NPY files
change_select_width_size =[24]

# Define the image preprocessing crop function
def custom_crop(image, top, left, height, width):
    return F.crop(image, int(top), int(left), height, width)

#Set the image to rotate and the rotation angle is fixed
class FixedRotation(object):
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, img):
        return transforms.functional.rotate(img, self.angle)
                  
for p in change_select_width_size:        
    for g in static_array:            
        for x in dirs_up:   
            #Define the number of data preprocessing repetitions within the folder for each label
            if x.strip() == '0 No':
                num_to_mlt=1
            if x.strip() == '1 Yes':
                num_to_mlt=1 
                             
            if x.strip() == '0 High':
                num_to_mlt=1
            if x.strip() == '1 Low':
                num_to_mlt=1 
            if x.strip() == '2 Suitable':
                num_to_mlt=1  

            if x.strip() == '0 As cap':
                num_to_mlt=1                   
            if x.strip() == '1 c(4×4)':
                num_to_mlt=1
            if x.strip() == '2 (2×4)':
                num_to_mlt=1
            if x.strip() == '3 Oxidation':
                num_to_mlt=1 
            if x.strip() == '4 Deoxidation':
                num_to_mlt=1  
            if x.strip() == '5 (n×6)':
                num_to_mlt=1 
           
            #Get the directory of a specific folder at the lower level
            dirs_down = os.listdir(image_date_file+'/'+str(x))     
            #Data preprocessing completion percentage count display
            num_add=0
            num = len(dirs_down)
            
            #Get a specific multi-image folder
            ##i Name of the document  
            for a in dirs_down:
                                
                key_title=str(x)       
                                
                #Every few images blend (in depth)
                select_width_size=p
                
                #Move a few images at a time in the header file
                select_stride=20  #The default value is 1, making it convenient to modify the demonstration code to 20
                
                #Pixel square selection
                width_length=g                
                
                num_add=num_add+1
                #Document selection             
                file_select = a   
                
                #**** NPY Storage Catalogue                
                result_path = result_path_base +'/' + key_title +' select_width_size ' +str(select_width_size) + ' width_length '+str(width_length)+ '//'
                                
                #num_to_mlt Number of operation repetitions
                for num_do in range(0,num_to_mlt): 
                    
                    print(num_to_mlt)
                
                    #Get timestamp
                    now = datetime.now()
                    time = str(now.strftime('%Y%m%d%H%M%S'))

                    #List of CSVs automatically read and stored
                    csv_path = result_path+file_select+' '+ time+'.csv'                

                    # The csv file already has the path to images, so here it only goes to the top level directory
                    img_path = image_date_file+'/'+str(x)+'/'+file_select

                    #Path to file after multi-channel processing
                    channel_path = result_path+file_select +' '+ time
                    if os.path.exists(channel_path):
                        pass
                    else:
                        os.makedirs(channel_path)

                    #Iterates over the filenames of the files in the entire folder and returns a list of them
                    f_n = os.listdir(img_path) 
                    #print(f_n) 
                    
                    #Check for file name rearrangement
                    fn=f_n.sort()
                    #print(f_n) 

                    #Collated filename written to csv file
                    #Write to csv
                    f=open(csv_path,"w") 
                    for line in f_n:
                        f.write(line+'\n')
                    f.close()
                    
                    #Read the number of csv file rows, corresponding to the number of files    
                    total_csv_lines = sum(1 for line in open(csv_path))

                    #Format to integer variable and save
                    total_csv_lines = (int)(total_csv_lines)                
                    select_width_size=(int)(select_width_size)                
                    select_stride=(int)(select_stride)                

                    from PIL import Image
                    from matplotlib import pyplot as plt
                    import torchvision
                    
                    #Generate random numbers for random image broadening
                    crop_x = random.random()
                    crop_y= random.random()                
                    
                    crop_a=random.random()
                    crop_b=random.random()
                    crop_c=random.random()
                    crop_d=random.random()
                    
                    blur_sigma = random.uniform(0.1, 5)  
                    blur_kernel_size = int(random.randint(3, 7)) 
                    # Ensure an odd number of convolution kernels for Gaussian fitting
                    if blur_kernel_size % 2 == 0:  
                        blur_kernel_size += 1
   
                    #Calculate the number of loop executions Multiply the reference convolution padding with the step size
                    num_total = total_csv_lines - select_width_size + select_stride + 1   
                    
                    for i in range(num_total):
                        if (total_csv_lines)>=(select_stride*(i)+select_width_size)+1:
                            
                            with open(csv_path,'r') as csvfile:
                                reader = csv.reader(csvfile)
                                num_to_ext=0
                                
                                
                                for k,rows in enumerate(reader):
                                    
                                    if k>(select_stride*(i)):
                                        if k<=(select_stride*(i)+select_width_size):
                                            row = str(*rows)                        
                                        else:
                                            break
                                        img_path_to=str(img_path+'/'+row)                                                                               

                                        # Define the image augmentation method
                                        transforms_augm = torchvision.transforms.Compose([
                                            torchvision.transforms.ColorJitter(
                                                brightness=0.3 * crop_a, 
                                                contrast=0.3 * crop_b, 
                                                saturation=0.3 * crop_c, 
                                                hue=0.3 * crop_d
                                            ),
                                            torchvision.transforms.GaussianBlur(
                                                kernel_size=(blur_kernel_size, blur_kernel_size), 
                                                sigma=(blur_sigma, blur_sigma)
                                            )
                                        ])
                                                                    
                                        #Loading images
                                        img = Image.open(img_path_to) 
                                       
                                        #Image Augmentation
                                        img = transforms_augm(img)
                                                                               
                                        #Randomly generated image crop range
                                        img = custom_crop(img,crop_x*50,crop_y*50,400,400)
                                       
                                        #Image conversion to single channel greyscale
                                        arr=np.array(img.resize((width_length, width_length)).convert('L'))
                                        
                                        #Add a new dimension
                                        to_sum = np.expand_dims(arr,0)
                                        
                                        #The dimensions are stacked and saved as a matrix with the first dimension as select_width_size
                                        if num_to_ext==0:
                                            sum_sum=to_sum
                                            num_to_ext=num_to_ext+1
                                        else:                     
                                            sum_sum=np.concatenate((sum_sum,to_sum), axis=0)
                                            num_to_ext=num_to_ext+1                                            
                                        if  num_to_ext>(select_width_size-1):                                            
                                            #Updating random numbers
                                            crop_x = random.random()
                                            crop_y= random.random()
                                            crop_a=random.random()
                                            crop_b=random.random()
                                            crop_c=random.random()
                                            crop_d=random.random()
                                            
                                            blur_sigma = random.uniform(0.1, 2.0)  
                                            blur_kernel_size = int(random.randint(1, 3))  
                                            if blur_kernel_size % 2 == 0:  
                                                blur_kernel_size += 1
                                            
                                            #print('Finished')
                                            break                                                                               
                                
                                sum_sum=sum_sum.reshape(select_width_size, width_length, width_length)                                
                                #NPY Final Storage Catalogue
                                np.save(channel_path+'//'+file_select+' '+str(select_stride)+' '+time +' '+str(i).zfill(5)+'.npy',sum_sum)
                                print(channel_path+'//'+file_select+' '+str(select_stride)+' '+time +' '+str(i).zfill(5)+'.npy')
                    print(num_add/num)            


