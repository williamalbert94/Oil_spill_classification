import argparse
from generator import get_coordinates,DataGenerator,balance_dataset_info
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.models import Sequential, Model
import os
from os.path import join, exists, dirname
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
import pandas
import sklearn.metrics as metrics
from train_classification import get_paths,get_min_max
import numpy as np
from sklearn.metrics import confusion_matrix
import sklearn.metrics as metrics
from metrics import *
from mlxtend.plotting import plot_confusion_matrix
import numpy as np
from models_classification import get_model
from glob import glob
import pandas as pd

def create_folders(args):
  directory = ['confusion_matrix','report_metrics']
  for path_name in directory:
    print(join(args.path_results,path_name))
    if not os.path.exists(join(args.path_results,path_name)):
      os.makedirs(join(args.path_results,path_name))

def fit_model(model):
  learning_rate = 0.0001
  decay_rate = 0
  momentum = 0.9
  opt_Adam = Adam(lr=0.0001) 

  model.compile(loss='binary_crossentropy', optimizer=opt_Adam,metrics=['accuracy',f1])
  return model

def report_metrics(model,test_generator):
  y_true = None
  y_predict_arr = None
  print('len test {}'.format(test_generator.__len__()))
  for index in range(test_generator.__len__()):
    x ,y = test_generator.__getitem__(index)
    y = np.argmax(y,axis=-1)
    y_predict = model.predict(x)
    y_predict = np.argmax(y_predict,axis=-1)
    if y_true is None :
        y_true = y
        y_true_arr = y_predict
    else:
        y_true = np.concatenate([y_true,y],axis=0)
        y_true_arr = np.concatenate([y_true_arr,y_predict],axis=0)

  cm = confusion_matrix(y_true, y_true_arr)      
  print(cm)
  report = metrics.classification_report(y_true, y_true_arr,output_dict=True)
  print(metrics.classification_report(y_true, y_true_arr))
  if args.save_report is True:
    path_graph = join(args.path_results,'report_metrics')
    file_name = join(path_graph,'{}.csv'.format(args.model_dir.split('/')[-1].split('.')[0]))
    df = pandas.DataFrame(report).transpose()
    df.to_csv(file_name)

    path_graph = join(args.path_results,'confusion_matrix')
    file_graph = join(path_graph,'{}.png'.format(args.model_dir.split('/')[-1].split('.')[0]))
    fig, ax = plot_confusion_matrix(conf_mat=cm,
                                show_absolute=True,
                                show_normed=True,
                                colorbar=True)
  plt.show()
  plt.savefig(file_graph)

def main(args):
  os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_id
  print('selecting gpu  :', args.gpu_id)
  create_folders(args)
  model = get_model(args=args,weights=None)
  model.load_weights(join(args.model_dir))
  print('model:',args.model_dir)
  model = fit_model(model=model)
  path = args.path_test
  path_image = join(path,'images')
  path_mask = join(path,'masks')
  list_image = glob(os.path.join(path_image,'*.tif'))
  list_mask = glob(os.path.join(path_mask,'*.tif'))
  min_dataset_values, max_dataset_values = 0,255 #get_min_max(args.path_train)
  df_path_test = join(path,'dataframe_test_dataset_{}_standarizate.csv'.format(args.size))
  if os.path.exists(df_path_test):
    print('{}    loaded!'.format(df_path_test))
    df_test = pd.read_csv(df_path_test) 
  else:
    print('{}    saved!'.format(df_path_test))
    df_test  = get_coordinates(paths = list_mask, out_size = (args.size,args.size) ,stride = 0.8 , classes = args.classes, stride_minor_class=0.1) 
    df_test.to_csv(df_path_test, index = False, header=True)
  test_generator = DataGenerator(list_IDs = df_test , batch_size=args.batch,dim=(args.size,args.size), n_channels=3,
                   n_classes=args.classes, min_max = (min_dataset_values, max_dataset_values), shuffle=True)
  report_metrics(model,test_generator)



if __name__ == '__main__':
      parser = argparse.ArgumentParser(description='test inference: oil spill dataset')
      parser.add_argument('--model_dir', type=str, dest='model_dir', help='Model full path', metavar='model')
      parser.add_argument('--path_train', type=str, dest='path_train', help='Path to datasets folder', metavar='path_dataset')      
      parser.add_argument('--path_test', type=str, dest='path_test', help='Path to datasets folder', metavar='path_dataset')
      parser.add_argument('--path_results', type=str, dest='path_results', help='Path to results folder', metavar='path_results')
      parser.add_argument('--optm', type=str, dest='optm', help='Optimizer', metavar='optm')
      parser.add_argument('--batch', type=int, dest='batch', help='Batchsize', metavar='BS')
      parser.add_argument('--size', type=int, dest='size', help='define size of tile', metavar='size')
      parser.add_argument('--classes', type=int, dest='classes', help='define classes', metavar='classes')
      parser.add_argument('--model', type=str, dest='model', help='define model to use', metavar='md')
      parser.add_argument('--save_report', dest='save_report', help='if true not save_report', action='store_true')
      parser.add_argument('--gpu_id', dest='gpu_id',  metavar='gpu_id')
      args = parser.parse_args()
      main(args)
