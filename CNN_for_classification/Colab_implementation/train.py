
import argparse
from keras.preprocessing.image import ImageDataGenerator
from models import get_model
from keras.optimizers import SGD, Adam
from keras.models import Sequential, Model
import os
from os.path import join, exists, dirname
from keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import tensorflow as tf

def split_train_val_generator(args):
    datagen = ImageDataGenerator(
        fill_mode = "nearest",
        rescale=1./255,
        rotation_range=0.1,                                
        horizontal_flip=True,                
        vertical_flip=True,
        validation_split=0.25
    )    

    train_generator = datagen.flow_from_directory(
                    args.path_train,
                    target_size=(args.size,args.size),
                    batch_size=args.batch,
                    class_mode='categorical',
                    subset="training"
                    ) 

    validation_generator = datagen.flow_from_directory(
                   args.path_train,
                   target_size=(args.size,args.size),
                   batch_size=args.batch,
                   class_mode='categorical',
                   subset="validation"
                   ) 
    return train_generator,validation_generator

def create_folders(args):
  directory = ['models','train_graphs']
  for path_name in directory:
    print(join(args.path_results,path_name))
    if not os.path.exists(join(args.path_results,path_name)):
      os.makedirs(join(args.path_results,path_name))

def save_training_graph(history):
    acc, val_acc, loss, val_loss = history.history['acc'],history.history['val_acc'],history.history['loss'], history.history['val_loss']
    plt.rcParams['axes.facecolor']='white'
    f, axarr = plt.subplots(1 , 2)
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
    
    path_graph = join(args.path_results,'train_graphs')
    file_name = join(path_graph,'{}.png'.format(args.modelname))
    f.savefig(file_name)


def start_train(args,model,train_generator,validation_generator):
  print('Start the training')
  path_model = join(args.path_results,'models')
  file_name = join(path_model,'{}.model'.format(args.modelname))
  checkpointer = ModelCheckpoint(file_name, monitor='val_f1',mode='max', save_best_only=False)
  early_stop = EarlyStopping(monitor = 'val_acc',mode='max', min_delta = 0.001, 
                             patience = 10) 
  history= model.fit_generator(
                 train_generator,
                 steps_per_epoch = train_generator.samples// args.batch,
                 validation_data = validation_generator, 
                 validation_steps = validation_generator.samples// args.batch,
                 epochs = args.epochs,callbacks=[checkpointer,early_stop])
  return history

def main(args):
  create_folders(args)
  train_generator,validation_generator = split_train_val_generator(args)
  model = get_model(args=args)
  history = start_train(args,model,train_generator,validation_generator)
  if args.no_plot==True:
    save_training_graph(history)



if __name__ == '__main__':
      parser = argparse.ArgumentParser(description='train: oil spill dataset')
      parser.add_argument('--model_name', type=str, dest='modelname', help='Modelname to save', metavar='model')
      parser.add_argument('--path_train', type=str, dest='path_train', help='Path to datasets folder', metavar='path_dataset')
      parser.add_argument('--path_results', type=str, dest='path_results', help='Path to results folder', metavar='path_results')
      parser.add_argument('--batch', type=int, dest='batch', help='Batchsize', metavar='BS')
      parser.add_argument('--size', type=int, dest='size', help='define size of tile', metavar='size')
      parser.add_argument('--epochs', type=int, dest='epochs', help='define epochs of training', metavar='epochs')
      parser.add_argument('--classes', type=int, dest='classes', help='define classes', metavar='classes')
      parser.add_argument('--model', type=str, dest='model', help='define model to use', metavar='md')
      parser.add_argument('--no_plot', dest='no_plot', help='if true not save training graph', action='store_true')
      args = parser.parse_args()
      main(args)
