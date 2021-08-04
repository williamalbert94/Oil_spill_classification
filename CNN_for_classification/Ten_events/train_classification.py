
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
from generator import *
from PIL import Image 
from train_utils import get_min_max
import csv
import pandas as pd
import pickle
from random import shuffle
def get_paths(path_train):
  """
  split tran and val paths
  return X_train,Y_train,X_val,Y_val paths
  """ 
  paths_classes_train = []
  paths_clases_val = []
  paths_clases_test = []
  paths = os.listdir(path_train)  
  print(paths)
  for classes in paths:
    print(classes)   
    list_image = glob('{}/{}/*.tiff'.format(path_train,classes))
    list_image = np.asarray(list_image)
    index_dataset = list(range(len(list_image)))
    train_size = int(len(index_dataset)*0.55)
    val_size = int(len(index_dataset)*0.70)  
    shuffle(index_dataset)
    train_index = index_dataset[:train_size] 
    val_index = index_dataset[train_size:val_size]
    test_index = index_dataset[val_size:]

    paths_classes_train = paths_classes_train + list(list_image[train_index]) 
    paths_clases_val = paths_clases_val + list(list_image[val_index])  
    paths_clases_test = paths_clases_test + list(list_image[test_index])
  return paths_classes_train, paths_clases_val , paths_clases_test

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



def start_train(args,model,train_generator,validation_generator):
  """
  define callbacks and start training using fit for generator. 
  """
  print('Start the training')
  path_model = join(args.path_results,'models')
  file_name = join(path_model,'{}.h5'.format(args.modelname))
  #if os.path.exists(file_name):
  #  print('model is loaded!')
  #  model.load_weights(join(file_name))    

  checkpointer = ModelCheckpoint(file_name, monitor='val_loss',mode='min', save_best_only=True)
  early_stop = EarlyStopping(monitor = 'val_f1',mode='max', min_delta = 0.001, 
                             patience = 15) 
  reduce_lr = ReduceLROnPlateau(monitor='val_f1',mode='max', factor=0.01,
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
  dataset_distribution_path = join(args.path_results,'dataset_distribution.csv')
  if os.path.exists(dataset_distribution_path):
    print('{}    loaded!'.format(dataset_distribution_path))
    dataset_distribution = pd.read_csv(dataset_distribution_path)
    train, val, test =  dataset_distribution['train'], dataset_distribution['val'], dataset_distribution['test']   
  else:
    train, val, test = get_paths(args.path_train)
    dataset_distribution  = list(zip(train, val ,test))            
    dataset_distribution = pd.DataFrame(dataset_distribution, columns = ['train', 'val' ,'test'])
    dataset_distribution.to_csv(dataset_distribution_path, index = False, header=True)

  train_generator = DataGenerator(list_IDs = train , batch_size=args.batch,dim=(args.size,args.size), n_channels=3,
                   n_classes=args.classes,norm=args.norm, transformations=True, shuffle=True)


  validation_generator = DataGenerator(list_IDs = val , batch_size=args.batch, dim=(args.size,args.size), n_channels=3,norm=args.norm,
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

