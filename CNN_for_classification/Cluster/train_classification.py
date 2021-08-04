
import argparse
import numpy as np
from models_classification import get_model
from tensorflow.keras.models import Sequential, Model
import os
from os.path import join, exists, dirname
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
import tensorflow as tf
from generator import DataGenerator
from glob import glob
from natsort import natsorted
from random import shuffle
from generator import get_coordinates,DataGenerator,balance_dataset_info
from PIL import Image 
from albumentations import (
    VerticalFlip,CenterCrop,Rotate,HorizontalFlip, ShiftScaleRotate,ElasticTransform,OpticalDistortion, RandomRotate90, Flip, OneOf, Compose,CLAHE )
from train_utils import get_min_max
import csv
import pandas as pd
import pickle

def get_paths(path_train):
  """
  split tran and val paths
  return X_train,Y_train,X_val,Y_val paths
  """ 
  path = path_train
  path_image = join(path,'images')
  path_mask = join(path,'masks')
  list_image = natsorted(glob(os.path.join(path_image,'*.tif')))
  list_image = np.asarray(list_image)
  list_mask = natsorted(glob(os.path.join(path_mask,'*.tif')))
  list_mask = np.asarray(list_mask)

  return list_image,list_mask

def create_folders(paths_results):
  """
  Create the default directories 
  """
  directory = ['models','train_graphs']
  for path_name in directory:

    if not os.path.exists(join(paths_results,path_name)):
      os.makedirs(join(paths_results,path_name))

def save_training_graph(history,path_save):
    """
    takes the history and saves the data regarding the training process
    """
    path_graph = join(path_save,'train_graphs')
    file_name = join(path_graph,'{}.png'.format(args.modelname))
    with open(join(path_graph,'{}'.format(args.modelname)), 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

    acc, val_acc, loss, val_loss,f1,val_f1 = history.history['acc'],history.history['val_acc'],history.history['loss'], history.history['val_loss'],history.history['f1'],history.history['val_f1']
    plt.rcParams['axes.facecolor']='white'
    f, axarr = plt.subplots(1 , 3)

    f.set_figwidth(10)

    # Accuracy
    axarr[0].plot(acc)
    axarr[0].plot(val_acc)
    axarr[0].set_title('model accuracy')
    axarr[0].set_ylabel('accuracy')
    axarr[0].set_xlabel('epoch')
    axarr[0].legend(['train', 'valid'], loc='upper left')

    # Loss
    axarr[1].plot(loss)
    axarr[1].plot(val_loss)
    axarr[1].set_title('model loss')
    axarr[1].set_ylabel('loss')
    axarr[1].set_xlabel('epoch')
    axarr[1].legend(['train', 'valid'], loc='upper left')

    axarr[2].plot(f1)
    axarr[2].plot(val_f1)
    axarr[2].set_title('model f1')
    axarr[2].set_ylabel('f1')
    axarr[2].set_xlabel('epoch')
    axarr[2].legend(['train', 'valid'], loc='upper left')
    

    f.savefig(file_name)



def generate_dataframe(path_mask,out_size,stride,classes,out_directory):

  stride = stride/100
  df_path_train = join(out_directory,'dataframe_train_dataset_{}_{}.csv'.format(out_size,int(stride*100)))
  df_path_val = join(out_directory,'dataframe_val_dataset_{}_{}.csv'.format(out_size,int(stride*100)))
  if os.path.exists(df_path_train):
    print('{}    loaded!'.format(df_path_train))
    df_train = pd.read_csv(df_path_train) 
    df_val = pd.read_csv(df_path_val) 
  else:
    df_train , df_val  = get_coordinates(paths = path_mask, out_size = (out_size,out_size) ,stride = stride , classes = classes,Mode ='train',stride_minor_class=0.08)  
    df_train.to_csv(df_path_train, index = False, header=True)
    df_val.to_csv(df_path_val, index = False, header=True)
  return df_train , df_val

def start_train(args,model,train_generator,validation_generator):
  """
  define callbacks and start training using fit for generator. 
  """
  print('Start the training')
  path_model = join(args.path_results,'models')
  file_name = join(path_model,'{}.h5'.format(args.modelname))

  checkpointer = ModelCheckpoint(file_name, monitor='val_loss',mode='min', save_best_only=True)
  early_stop = EarlyStopping(monitor = 'val_f1',mode='max', min_delta = 0.01, 
                             patience = 15) 
  reduce_lr = ReduceLROnPlateau(monitor='val_f1',mode='max', factor=0.001,
                              patience=7, min_lr=0.000000000000000000001)
  history= model.fit_generator(
                train_generator,
                steps_per_epoch = train_generator.__len__()//args.batch,
                validation_data = validation_generator, 
                validation_steps = validation_generator.__len__()//args.batch,
                epochs = args.epochs,callbacks=[checkpointer,reduce_lr])
  return history

def main(args):
  os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_id
  print('selecting gpu  :', args.gpu_id)
  create_folders(args.path_results)

  list_image,list_mask = get_paths(args.path_train)

  df_train , df_val = generate_dataframe(path_mask=list_mask,out_size=args.size,stride=args.stride,classes=args.classes,out_directory=args.path_train)


  train_generator = DataGenerator(list_IDs = df_train , batch_size=args.batch,dim=(args.size,args.size), n_channels=3,
                   n_classes=args.classes,norm=args.norm, transformations=True, shuffle=True)

  validation_generator = DataGenerator(list_IDs = df_val , batch_size=args.batch, dim=(args.size,args.size), n_channels=3,norm=args.norm,
                   n_classes=args.classes)

  model = get_model(args=args,initial_lr=0.0001)
  history = start_train(args,model,train_generator,validation_generator)
  if args.no_plot==True:
    save_training_graph(history,args.path_results)


if __name__ == '__main__':
      parser = argparse.ArgumentParser(description='train: oil spill dataset')
      parser.add_argument('--model_name', type=str, dest='modelname', help='Modelname to save', metavar='model')
      parser.add_argument('--path_train', type=str, dest='path_train', help='Path to datasets folder', metavar='path_dataset')
      parser.add_argument('--path_results', type=str, dest='path_results', help='Path to results folder', metavar='path_results')
      parser.add_argument('--optm', type=str, dest='optm', help='Optimizer', metavar='optm')
      parser.add_argument('--norm', type=str, dest='norm', help='Normalization', metavar='norm')
      parser.add_argument('--batch', type=int, dest='batch', help='Batchsize', metavar='BS')
      parser.add_argument('--size', type=int, dest='size', help='define size of tile', metavar='size')
      parser.add_argument('--classes', type=int, dest='classes', help='define classes', metavar='classes')
      parser.add_argument('--stride', type=int, dest='stride', help='define stride', metavar='stride')
      parser.add_argument('--epochs', type=int, dest='epochs', help='define epochs of training', metavar='epochs')
      parser.add_argument('--model', type=str, dest='model', help='define model to use', metavar='md')
      parser.add_argument('--no_plot', dest='no_plot', help='if true not save training graph', action='store_true')
      parser.add_argument('--gpu_id', dest='gpu_id',  metavar='gpu_id')
      args = parser.parse_args()
      main(args)

