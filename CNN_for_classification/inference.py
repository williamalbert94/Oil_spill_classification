import argparse
from keras.preprocessing.image import ImageDataGenerator
from models import get_model
from keras.optimizers import SGD, Adam
from keras.models import Sequential, Model
import os
from os.path import join, exists, dirname
from keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import pandas
import sklearn.metrics as metrics
from keras.models import load_model
import numpy as np
from sklearn.metrics import confusion_matrix
import sklearn.metrics as metrics
from metrics import *
from mlxtend.plotting import plot_confusion_matrix
import numpy as np

def split_test_gen(args):
    datagen = ImageDataGenerator(
        fill_mode = "nearest",
        rescale=1./255,
    )    

    test_generator = datagen.flow_from_directory(
                    args.path_test,
                    target_size=(args.size,args.size),
                    batch_size=args.batch,
                    class_mode='categorical'
                    ) 

    return test_generator

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
  create_folders(args)
  model = load_model(args.model_dir,compile=False)
  print('model:',args.model_dir)
  model = fit_model(model=model)
  test_generator = split_test_gen(args)
  report_metrics(model,test_generator)



if __name__ == '__main__':
      parser = argparse.ArgumentParser(description='test inference: oil spill dataset')
      parser.add_argument('--model_dir', type=str, dest='model_dir', help='Model full path', metavar='model')
      parser.add_argument('--path_test', type=str, dest='path_test', help='Path to datasets folder', metavar='path_dataset')
      parser.add_argument('--path_results', type=str, dest='path_results', help='Path to results folder', metavar='path_results')
      parser.add_argument('--batch', type=int, dest='batch', help='Batchsize', metavar='BS')
      parser.add_argument('--size', type=int, dest='size', help='define size of tile', metavar='size')
      parser.add_argument('--save_report', dest='save_report', help='if true not save_report', action='store_true')
      args = parser.parse_args()
      main(args)
