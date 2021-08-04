import numpy as np
import tensorflow.keras
from tensorflow.keras.utils import to_categorical
import cv2
from random import shuffle
import os
import numpy as np
import matplotlib.pyplot as plt
import numpy
import scipy
from scipy import ndimage, misc
from scipy.ndimage.interpolation import zoom
from random import shuffle,choice
from albumentations import (
    VerticalFlip,CenterCrop,Rotate,HorizontalFlip, ShiftScaleRotate,ElasticTransform,OpticalDistortion, RandomRotate90,CLAHE , Flip, OneOf, Compose)
import pickle
from natsort import natsorted
import random
from tensorflow.keras.utils import Sequence
from PIL import Image,ImageStat 
import pandas as pd
Image.MAX_IMAGE_PIXELS = 933120000

def Zoom(img, zoomfactor=0.2):
    
    out  = np.zeros_like(img)
    zoomed = cv2.resize(img, None, fx=zoomfactor, fy=zoomfactor)
    
    h, w = img.shape
    zh, zw = zoomed.shape
    
    if zoomfactor<1:  
        out[(h-zh)/2:-(h-zh)/2, (w-zw)/2:-(w-zw)/2] = zoomed
    else:              
        out = zoomed[(zh-h)/2:-(zh-h)/2, (zw-w)/2:-(zw-w)/2]

    return out

class DataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, batch_size=2, dim=(128,128), n_channels=1,
                 n_classes=10,norm="min_max",transformations = None, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.norm = norm
        self.transformations = transformations
        self.shuffle = shuffle
        self.on_epoch_end()
        if transformations is not None:
            self.trans_custom = ['Normal','flip_x','flip_y','Zoom_in']
        else:
            self.trans_custom = ['Normal']
        self.scaler = pickle.load(open('./scaler_standarizate_events_in_sea.pkl','rb'))
        self.classes_dict = {'F':0, 'G':1, 'H':2, 'I':3, 'J':4, 'K':5, 'L':6, 'M':7, 'N':8, 'O':9}


    def apply_some_transformations(self,img,mask=None):
        'Random morphological transformations'
        data = {"image": img, "mask": mask}
        augmented = self.transformations(**data)
        img_trans, mask_trans= augmented["image"], augmented["mask"]
        x=np.stack((img_trans,)*self.n_channels, axis=-1)
        if mask is None:
            return x
        else:
            y = np.stack((mask_trans,)*1, axis=-1)
            return x,y

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))-1

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        #self.global_index +=1 
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        # Generate data
        x, y = self.__data_generation(list_IDs_temp)
        return x, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        # Generate data
        x = None
        y = None
        for i, ID in enumerate(list_IDs_temp):
            #print(ID)
            
            image = cv2.imread(ID, cv2.IMREAD_ANYDEPTH)
            image = cv2.resize(image, (self.dim[0],self.dim[1]), interpolation = cv2.INTER_NEAREST )
            if self.norm=="min_max":
                min_max_norm = [0, 65535]
                image = (np.asarray(image)-np.min(min_max_norm))/(np.max(min_max_norm)-np.min(min_max_norm))
            if self.norm=="standar":
                image = np.asarray(image)                   
                image = image.reshape(-1,1)
                image = self.scaler.transform(image)
                image = np.reshape(image,(self.dim[0],self.dim[1]))
                image = (np.asarray(image)-np.min(image))/(np.max(image)-np.min(image))                

            selected_trans = random.choice(self.trans_custom)
            if selected_trans=='Normal':
                image=image
            if selected_trans=='flip_x':
                image=np.flipud(image)
            if selected_trans=='flip_y':
                image=np.fliplr(image)
            if selected_trans=='Zoom_in':
                image=Zoom(image)
            image=np.stack((image,)*self.n_channels, axis=-1)

            mask = [self.classes_dict[ID.split('//')[-1].split('/')[0]]]
  
            image = np.expand_dims(image,axis=0)

            if x is None :
                x = image
                y = mask
            else:
                x = np.concatenate((x,image),axis=0)
                y = np.concatenate((y,mask))            

        return x,to_categorical(y,self.n_classes)
