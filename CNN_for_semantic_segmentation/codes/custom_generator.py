# -*- coding: utf-8 -*-
import numpy as np
import tensorflow.keras
from tensorflow.keras.utils import to_categorical
import cv2
from random import shuffle
import os
from os.path import join, exists, dirname
import numpy as np
import matplotlib.pyplot as plt
import numpy
import scipy
from scipy import ndimage, misc
from scipy.ndimage.interpolation import zoom
from random import shuffle,choice
import random
from tensorflow.keras.utils import Sequence
from PIL import Image,ImageStat 
import pandas as pd
Image.MAX_IMAGE_PIXELS = 933120000
import keras
import pickle

def Zoom(img, zoomfactor=0.02):  

    
    h, w = img.shape
    out = img[:int(h-h*zoomfactor),:int(w-w*zoomfactor)]

    return out

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

def rle2mask(mask_rle, shape=(1600,256)):
    """
    mask_rle: run-length as string formated (start length)
    shape: (width,height) of array to return 
    Returns numpy array, 1 - mask, 0 - background
    """
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T

def extract_RLE(img):
    """
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    """
    pixels= img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2][:len(runs[1::2])]
    return ' '.join(str(x) for x in runs)

def extract_patchs(img,mask,RLE_,path_image,output_shape,stride,class_in,stride_minor_class):
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
    stride_in = stride if (class_in == 0) else stride_minor_class
    cut_dim1 = int(output_shape[0]*stride_in)
    cut_dim2 = int(output_shape[1]*stride_in)
    add1 = 0
    add2 = 0

    while add2 < size[1]:
        
        while add1 < size[0]:
            id_y = [0 + add1, output_shape[0] + add1]
            id_x = [0 + add2, output_shape[1] + add2]
            
            if id_x[1] > size[0]:
                dim1_start = (size[0]-5)-(output_shape[0])
                dim1_end = size[0]-5
                if dim1_end<=size[0]:
                    id_x = [dim1_start,dim1_end]
                
            if id_y[1] > size[1]:
                dim2_start = (size[1]-5) - output_shape[1]
                dim2_end = size[1]-5
                if dim2_end<=size[1]:
                    id_y = [dim2_start,dim2_end]



            patch_image = img.crop((id_x[0],id_y[0],id_x[1],id_y[1]))
            patch_image_array = np.asarray(patch_image)   
            patch_mask = mask.crop((id_x[0],id_y[0],id_x[1],id_y[1]))
            patch_mask_array = np.asarray(patch_mask).astype(np.uint8)
            if np.max(patch_mask_array) == class_in: #and np.mean(patch_image_array)>1:
                patch_class.append(RLE_)
                coordinates.append('{},{}'.format(id_x[1],id_y[1]))
                out_paths.append(path_image)
                    
            add1 = int(add1 + cut_dim1)
        add2 = int(add2 + cut_dim2)
        add1 = 0
    return out_paths, coordinates , patch_class

def get_coordinates(geodataframe,out_size,stride,classes,Mode = 'test',stride_minor_class = 0.5):
    """
    gets array of coordinates and class for extract each patch.
    """
    coordinates_in_patch = None
    out_path_patch = None
    class_in_patch = None

    for ind, row in geodataframe.T.iteritems():
        out_paths = coordinates = patch_class = []
        path_in = row['Path']
        img = cv2.imread(path_in,cv2.IMREAD_ANYDEPTH)
        rle_ = row['RLE']
        mask = rle2mask(rle_,(img.shape[1],img.shape[0]))
        img = img
        img = Image.fromarray(img)
        #img = img.convert("L")
        mask = mask
        mask = Image.fromarray(mask)
       # mask = mask.convert("L")
        out_paths_pos, coordinates_pos , patch_class_pos = extract_patchs(img,mask,rle_,path_in,out_size,stride_minor_class,1,stride_minor_class)
        out_paths_neg, coordinates_neg , patch_class_neg = extract_patchs(img,mask,rle_,path_in,out_size,stride,0,stride)
        out_paths = np.concatenate((out_paths_pos,out_paths_neg),axis=0)
        coordinates =  np.concatenate((coordinates_pos,coordinates_neg),axis=0)
        patch_class = np.concatenate((patch_class_pos,patch_class_neg),axis=0)
        print('pos:',len(out_paths_pos),'-neg:',len(out_paths_neg))
        if coordinates_in_patch is None:
            coordinates_in_patch = coordinates
            out_path_patch = out_paths
            class_in_patch = patch_class
        else:
            coordinates_in_patch = np.concatenate((coordinates_in_patch,coordinates),axis=0)       
            out_path_patch = np.concatenate((out_path_patch,out_paths),axis=0)
            class_in_patch = np.concatenate((class_in_patch,patch_class),axis=0)

    index_dataset = list(range(len(coordinates_in_patch)))
    train_size = int(len(index_dataset)*0.70)
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

class DataGenerator_deeplab(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, batch_size=2, dim=(128,128), n_channels=1,
                 n_classes=2,norm="min_max",transformations = ['Normal'], shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.norm = norm
        self.shuffle = shuffle
        self.global_index = 0
        self.gets_images()
        self.on_epoch_end()
        self.trans_custom = transformations
        self.scaler = pickle.load(open('/scratch/parceirosbr/manntisict/radar/TEST_MODELS/oil_spill_segm/scaler_standarizate_oil_data.pkl','rb'))
        self.load_data(path_image =list_IDs[0:1]['paths'].values[0],RLE=list_IDs[0:1]['class'].values[0])

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
        return  x, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def load_data(self,RLE,path_image):
        self.path_image = path_image
        self.images=cv2.imread(self.path_image, cv2.IMREAD_ANYDEPTH)


        self.size_img = self.images.shape
        self.images = Image.fromarray(self.images)
        #self.images = self.images.convert("L")  

        self.mask = rle2mask(RLE, (self.size_img[1],self.size_img[0]))
        self.mask = Image.fromarray(self.mask)
        self.mask = self.mask.convert("L")



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
            mask = ID['class'].values[0]
            
            if self.path_image!=path_image:
                self.load_data(path_image = path_image,RLE=mask) 

            image = self.images.crop((coord_in_x_init,coord_in_y_init,coord_in_x_final,coord_in_y_final))
            mask= self.mask.crop((coord_in_x_init,coord_in_y_init,coord_in_x_final,coord_in_y_final))
            mask = np.asarray(mask) 
            image = np.asarray(image)

            #if self.norm=="standar":
            #    image = np.asarray(image)
            #    image = image.reshape(-1,1)
            #    image = (image-582.59898115)/1256408.30603894
            #    image = np.reshape(image,(self.dim[0],self.dim[1]))

            selected_trans = random.choice(self.trans_custom)
            if selected_trans=='Normal':
                image=image
                mask=mask
            if selected_trans=='flip_x':
                image=np.flipud(image)
                mask=np.flipud(mask)
            if selected_trans=='Zoom':
                image=Zoom(image)
                mask=Zoom(mask)
            if selected_trans=='flip_y':
                image=np.fliplr(image)
                mask=np.fliplr(mask)
            if selected_trans=='CLAHE':     
                clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(64,64))
                image=clahe.apply(image)
                mask=mask

            if self.norm=="min_max":
                image = image/255
            if self.norm=="standar":
                image = np.asarray(image)
                image = image.reshape(-1,1)
                image = self.scaler.transform(image)
                image = np.reshape(image,(self.dim[0],self.dim[1]))

            image=np.stack((image,)*self.n_channels, axis=-1)
            mask=np.stack((mask,)*1, axis=-1)

            image = cv2.resize(image, self.dim , interpolation = cv2.INTER_AREA)            
            mask = cv2.resize(mask, self.dim , interpolation = cv2.INTER_AREA)
            image = np.expand_dims(image,axis=0)
            mask = np.expand_dims(mask,axis=0)
            if x is None :
                x = image
            else:
                x = np.concatenate((x,image),axis=0)   

            if y is None :   
                y = mask  

            else:
                y = np.concatenate((y,mask),axis=0)         

        return x,to_categorical(y,self.n_classes)