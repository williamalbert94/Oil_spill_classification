from glob import glob
import pandas as pd
import os
from os.path import join, exists, dirname
import argparse
import numpy as np
import matplotlib.pyplot as plt
from keras.optimizers import adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import tensorflow as tf
from tensorflow.keras import Model
from sklearn.metrics import classification_report
from get_model import *
from utils_inf import *
from custom_generator import *
from custom_generator import DataGenerator_deeplab as DataGenerator
import pickle
import tensorflow.keras.backend as K


def get_loss(weights):
    def loss(y_true, y_pred):
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss
    return loss

def run_train(args,train_generator,validation_generator):
	"""
	Start the train process:
	args(argparse): Parameters of argparse input.
	train_gen,val_gen(object): generators.
	"""
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		model = model_define(args.size,args.size,args.classes,args.model_name,None)
		opt = adam(lr=0.0001,clipnorm=1.)
		#if args.model_name=='Unet':
		#	from tensorflow.keras.optimizers import Adam
		#	opt = Adam(lr=0.0001,clipnorm=1.)
		model.compile(optimizer =opt , loss='binary_crossentropy',metrics=['accuracy',dice_coef,iou_coef,get_mean_iou(2)])
		steps_per_epoch = train_generator.__len__() // args.batch
		print("total steps_per_epoch :{}".format(steps_per_epoch))

		earlyStopping = EarlyStopping(monitor='val_mean_iou', mode = 'max', patience=16, verbose=1)
		reduce_lr = ReduceLROnPlateau(monitor='val_loss', mode = 'min', factor=0.0001, patience=6,min_lr=0.0000000001, verbose=1)
		model_checkpoint = ModelCheckpoint(args.model_weigths_dir, monitor='val_mean_iou',
											mode='max', save_best_only=True, verbose=1, period=1)
		validation_steps = validation_generator.__len__() // args.batch
		history = model.fit_generator(train_generator,
										steps_per_epoch=steps_per_epoch,
										epochs=args.epochs,
										verbose=1,
										validation_data=validation_generator,
										validation_steps=validation_steps,
										callbacks=[earlyStopping,reduce_lr,model_checkpoint],
										shuffle=True)

		with open(args.model_weigths_dir.replace('.h5',''), 'wb') as file_pi:
			pickle.dump(history.history, file_pi)
		acc, val_acc, loss, val_loss,iou,val_iou= history.history['acc'],history.history['val_acc'],history.history['loss'], history.history['val_loss'],history.history['mean_iou'], history.history['val_mean_iou']		
		plt.rcParams['axes.facecolor']='white'
		f, axarr = plt.subplots(1 , 3)

		f.set_figwidth(20)
		f.set_figheight(7)
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

		axarr[2].plot(iou)
		axarr[2].plot(val_iou)
		axarr[2].set_title('model mean iou')
		axarr[2].set_ylabel('meaniou')
		axarr[2].set_xlabel('epoch')
		axarr[1].legend(['train', 'valid'], loc='upper left')
		plt.savefig(args.model_weigths_dir.replace('.h5','.png'),bbox_inches='tight')



def main(args):
	print('GPU:{}'.format(args.id_gpu))
	os.environ["CUDA_VISIBLE_DEVICES"]=args.id_gpu
	csv_train = '{}/2nontimepQLK_trainpatches_{}_data{}_{}.csv'.format(args.csv_path,args.split_train,args.size,args.stride)
	csv_val = '{}/2nontimepQLK_valpatches_data_{}_{}_{}.csv'.format(args.csv_path,args.split_train,args.size,args.stride)

	if os.path.exists(csv_train):
	    print('Patches distribution loaded!')
	    df_train = pd.read_csv(csv_train) 
	    df_train = df_train.dropna()
	    df_val = pd.read_csv(csv_val)
	    df_val = df_val.dropna()    

	else:
	    print('Calculating patches distribution...')
	    geodataframe = pd.read_csv('../RLE_CSV/train_oil.csv')
	    geodataframe = geodataframe.dropna()
	    geodataframe = geodataframe
	    df_train,df_val = get_coordinates(geodataframe = geodataframe, out_size = (args.size,args.size) ,stride = 0.3, classes = 2 ,Mode ='train',stride_minor_class=args.stride)    
	    df_train.to_csv(csv_train, index = False, header=True)
	    df_val.to_csv(csv_val, index = False, header=True)


	used_transf = []
	used_transf.append('Normal')
	if args.Flip_x is True:
		used_transf.append('flip_x')
	if args.Flip_y is True:
		used_transf.append('flip_y')
	if args.zoom is True:
		used_transf.append('Zoom')
	if args.CLAHE is True:
		used_transf.append('CLAHE')		
	used_transf = np.asarray(used_transf)
	print('used transformations:',used_transf)

	train_generator = DataGenerator(list_IDs = df_train , batch_size=args.batch,dim=(args.size,args.size), n_channels=3,
	                 n_classes=args.classes,norm='standar', transformations=used_transf, shuffle=True)

	validation_generator = DataGenerator(list_IDs = df_val , batch_size=args.batch, dim=(args.size,args.size), n_channels=3,norm='standar',
	                 n_classes=args.classes)

	run_train(args,train_generator,validation_generator)

if __name__ == '__main__':
      parser = argparse.ArgumentParser(description='Inference: Radar QLK images')
      parser.add_argument('--id_gpu', type=str, dest='id_gpu', help='id da gpu', metavar='id_gpu')
      parser.add_argument('--size', type=int, dest='size', help='Input size', metavar='size')
      parser.add_argument('--stride', type=float, dest='stride', help='Input stride', metavar='stride')
      parser.add_argument('--csv_path', type=str, dest='csv_path', help='csv_path', metavar='csv_path')
      parser.add_argument('--classes', type=int, dest='classes', help='Input classes', metavar='classes')
      parser.add_argument('--model_name', type=str, dest='model_name', help='Input model_name', metavar='model_name')
      parser.add_argument('--model_weigths_dir', type=str, dest='model_weigths_dir', help='Input model_weigths_dir', metavar='model_weigths_dir')
      parser.add_argument('--batch', type=int, dest='batch', help='batch', metavar='batch', default=8)
      parser.add_argument('--epochs', type=int, dest='epochs', help='epochs', metavar='epochs', default=25)
      parser.add_argument('--split_train', type=int, dest='split_train', help='split_train', metavar='split_train', default=70)
      parser.add_argument('--save_plot', type=bool, dest='save_plot', help='save_plot', metavar='save_plot', default= False)
      parser.add_argument('--Flip_x', type=bool, dest='Flip_x', help='Flip_x', metavar='Flip_x', default= False)
      parser.add_argument('--Flip_y', type=bool, dest='Flip_y', help='Flip_y', metavar='Flip_y', default= False)
      parser.add_argument('--zoom', type=bool, dest='zoom', help='zoom', metavar='zoom', default= False)
      parser.add_argument('--CLAHE', type=bool, dest='CLAHE', help='CLAHE', metavar='CLAHE', default= False)
      args = parser.parse_args()
      main(args)
