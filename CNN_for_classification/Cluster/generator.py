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

def balance_dataset_info(array,classes):
    """
    get info of balance in dataset per class
    """
    balance = []
    for class_ in range(classes):
        counter = 0
        for index_ in range(len(array)):
            if array[index_]== class_:
                counter+=1
        balance.append(counter)
    return balance

def extract_patchs(img,mask,path_image,output_shape,stride,classes,stride_minor_class):
    """
    Returns coordinates for extract cubes in generator

    Parameters:
        - data(numpy.array): Cube of seismic data.
        -output_shape(int list): expected shape on output (How cut the cube).
        -stride(int): if is expect to apply a stride when removing the cubes.
    """
    size = mask.size 
    coordinates = []
    out_paths = []
    patch_class = []
    count_pos = 0
    count_neg = 0
    class_in = classes
    stride = stride if (class_in == 0) else stride_minor_class
    tresholding = 0.08 if (class_in == 1) else 0      
    cut_dim1 = int(output_shape[0]*stride)
    cut_dim2 = int(output_shape[1]*stride)
    add1 = 0
    add2 = 0
    print('class:{}, stride:{}, thresholding:{}'.format(class_in,stride,tresholding))

    while add2 < size[1]:
        
        while add1 < size[0]:
            id_y = [0 + add1, output_shape[0] + add1]
            id_x = [0 + add2, output_shape[1] + add2]
            
            if id_x[1] > size[0]:
                dim1_start = size[0] - output_shape[0]
                dim1_end = size[0]
                id_x = [dim1_start,dim1_end]
                
            if id_y[1] > size[1]:
                dim2_start = size[1] - output_shape[1]
                dim2_end = size[1]
                id_y = [dim2_start,dim2_end]



            patch_image = img.crop((id_x[0],id_y[0],id_x[1],id_y[1]))
            patch_image_array = np.asarray(patch_image)   
            patch_mask = mask.crop((id_x[0],id_y[0],id_x[1],id_y[1]))
            patch_mask_array = np.asarray(patch_mask).astype(np.uint8)
            _, patch_max = patch_mask.getextrema()
            if patch_max == class_in:
                count_white_piexels=np.count_nonzero(patch_mask_array==1)
                count_total_pixels=output_shape[0]*output_shape[1]
                mean=1-((count_total_pixels-count_white_piexels)/(count_total_pixels))                
      
                if mean >= tresholding:
        
                    patch_class.append(class_in)
                    coordinates.append('{},{}'.format(id_x[1],id_y[1]))
                    out_paths.append(path_image)
                    
            add1 = int(add1 + cut_dim1)
        add2 = int(add2 + cut_dim2)
        add1 = 0
    return out_paths, coordinates , patch_class

def get_coordinates(paths,out_size,stride,classes,Mode = 'test',stride_minor_class = 0.1):
    """
    gets array of coordinates and class for extract each patch.
    """
    coordinates_in_patch = None
    out_path_patch = None
    class_in_patch = None
    for path in paths:
        print(path)
        out_paths = coordinates = patch_class = []
        out_path_image = path.replace('masks','images')
        img = Image.open(path)
        img = img.convert("L")
        mask = Image.open(path)
        mask = mask.convert("L")

        out_paths_pos, coordinates_pos , patch_class_pos = extract_patchs(img,mask,out_path_image,out_size,stride,1,stride_minor_class)
        print('pos',len(out_paths_pos))
        out_paths_neg, coordinates_neg , patch_class_neg = extract_patchs(img,mask,out_path_image,out_size,stride,0,stride)
        print('neg',len(out_paths_neg))
        out_paths = np.concatenate((out_paths_pos,out_paths_neg),axis=0)
        coordinates =  np.concatenate((coordinates_pos,coordinates_neg),axis=0)
        patch_class = np.concatenate((patch_class_pos,patch_class_neg),axis=0)

        if coordinates_in_patch is None:
            coordinates_in_patch = coordinates
            out_path_patch = out_paths
            class_in_patch = patch_class

        else:
            coordinates_in_patch = np.concatenate((coordinates_in_patch,coordinates),axis=0)       
            out_path_patch = np.concatenate((out_path_patch,out_paths),axis=0)
            class_in_patch = np.concatenate((class_in_patch,patch_class),axis=0)
    view_balance_in_dataset = balance_dataset_info(class_in_patch,classes)

    print(view_balance_in_dataset)

    if Mode == 'train':
        index_dataset = list(range(len(coordinates_in_patch)))
        train_size = int(len(index_dataset)*0.75)
        shuffle(index_dataset)
        train_index = index_dataset[:train_size] 
        val_index = index_dataset[train_size:]

        coordinates_in_patch_train = coordinates_in_patch[train_index] 
        out_path_patch_train =  out_path_patch[train_index] 
        class_in_patch_train = class_in_patch[train_index] 

        coordinates_in_patch_val = coordinates_in_patch[val_index]
        out_path_patch_val = out_path_patch[val_index]
        class_in_patch_val = class_in_patch[val_index]


        list_of_tuples_train  = list(zip(out_path_patch_train, coordinates_in_patch_train ,class_in_patch_train))            
        df_train = pd.DataFrame(list_of_tuples_train, columns = ['paths', 'coordinates' ,'class'])

        list_of_tuples_val  = list(zip(out_path_patch_val, coordinates_in_patch_val ,class_in_patch_val))            
        df_val = pd.DataFrame(list_of_tuples_val, columns = ['paths', 'coordinates' ,'class'])
        return df_train,df_val
    else:
        list_of_tuples  = list(zip(out_path_patch, coordinates_in_patch ,class_in_patch))            
        df_test = pd.DataFrame(list_of_tuples, columns = ['paths', 'coordinates' ,'class'])
        return df_test


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
        self.global_index = 0
        self.gets_images()
        self.on_epoch_end()
        if transformations is not None:
            self.trans_custom = ['Normal','flip_x','flip_y','Zoom_in']
        else:
            self.trans_custom = ['Normal']
        self.scaler = pickle.load(open('./scaler_standarizate_oil_data.pkl','rb'))

        self.load_data(path_image =list_IDs[0:1]['paths'].values[0])

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

    def gets_images(self):
        self.ids_images = np.unique(np.asarray(self.list_IDs['paths']))

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))-1

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        #self.global_index +=1 
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k:k+1] for k in indexes]

        # Generate data
        x, y = self.__data_generation(list_IDs_temp)
        return x, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def load_data(self,path_image):
        self.path_image = path_image
        self.images = Image.open(path_image)

    def Zoom(img, zoomfactor=0.8):
        
        out  = np.zeros_like(img)
        zoomed = cv2.resize(img, None, fx=zoomfactor, fy=zoomfactor)
        
        h, w = img.shape
        zh, zw = zoomed.shape
        
        if zoomfactor<1:  
            out[(h-zh)/2:-(h-zh)/2, (w-zw)/2:-(w-zw)/2] = zoomed
        else:              
            out = zoomed[(zh-h)/2:-(zh-h)/2, (zw-w)/2:-(zw-w)/2]

        return out

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        # Generate data
        x = None
        y = None
        for i, ID in enumerate(list_IDs_temp):
            # Store sample

            coordinates = ID['coordinates'].values[0]
            coord_in_x_final = int(coordinates.split(',')[0])
            coord_in_y_final = int(coordinates.split(',')[1])

            coord_in_x_init = coord_in_x_final - self.dim[0]
            coord_in_y_init = coord_in_y_final - self.dim[1]

            path_image = ID['paths'].values[0]
            
            if self.path_image!=path_image:
                self.load_data(path_image = path_image)  



            image = self.images.crop((coord_in_x_init,coord_in_y_init,coord_in_x_final,coord_in_y_final))
            if self.norm=="min_max":
                min_max_norm = [-40.589836 , 27.417784]
                image = (np.asarray(image)-np.min(min_max_norm))/(np.max(min_max_norm)-np.min(min_max_norm))
            if self.norm=="standar":
                image = np.asarray(image)                   
                image = image.reshape(-1,1)
                image = self.scaler.transform(image)
                image = np.reshape(image,(self.dim[0],self.dim[1]))

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

            mask = [int(ID['class'])]
            image = np.expand_dims(image,axis=0)
            if x is None :
                x = image
                y = mask
            else:
                x = np.concatenate((x,image),axis=0)
                y = np.concatenate((y,mask))            

        return x,to_categorical(y,self.n_classes)
