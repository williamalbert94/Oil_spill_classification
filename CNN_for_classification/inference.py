import argparse
from keras.preprocessing.image import ImageDataGenerator
from models import get_model
from keras.optimizers import SGD, Adam
from keras.models import Sequential, Model
import os
from os.path import join, exists, dirname
from keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt

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

  model.compile(loss='binary_crossentropy', optimizer=opt_Adam,metrics=['accuracy'])
  return model

def main(args):
  create_folders(args)
  model = load_model(args.model_dir)
  model = fit_model(model=model)







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
