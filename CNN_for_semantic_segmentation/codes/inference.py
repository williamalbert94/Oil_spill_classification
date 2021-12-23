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
from generator import *
import pandas as pd
import numpy as np
import os
from os.path import join,basename,splitext
import ast
import scipy
import pickle


def read_img(img_path):
    img=cv2.imread(img_path, cv2.IMREAD_ANYDEPTH)
    return np.asarray(img)

def sliding_window(image, step_size, window_size):
    for y in range(0, image.shape[0], step_size):
        if (y+step_size)>image.shape[0]:
            y = image.shape[0]-step_size
        for x in range(0, image.shape[1], step_size):
            if (x+step_size)>image.shape[1]:
                x = image.shape[1]-step_size
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])

def run_inference_batch(args,geodataframe):
	"""
	Start the inference process:
	args(argparse): Parameters of argparse input.
	geodataframe(dataframe): Dataframe for test.
	return metrics(dataframe) and save images of inference
	"""
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		image_name = []
		f1_score_out = []
		recall_out = []
		precision_out = []
		model = model_define(args.size,args.size,args.classes,args.model_name,None)
		opt = adam(lr=0.0001,clipnorm=1.)
		model.compile(optimizer =opt , loss='binary_crossentropy',metrics=['accuracy',dice_coef,iou_coef])
		model.load_weights(args.model_weigths_dir)

		Path_image = '{}/images/'.format(args.out_path_inf)
		Path_mask = '{}/mask/'.format(args.out_path_inf)
		Path_Inference = '{}/Inference/'.format(args.out_path_inf)
		Path_border = '{}/Border/'.format(args.out_path_inf)
		folders_inf = [Path_image,Path_border,Path_Inference,Path_mask]

		for dirs_inference in folders_inf:

			if not os.path.exists(dirs_inference):
				os.makedirs(dirs_inference)
		
		for ind, row in geodataframe.T.iteritems():
			path = row['Path']
			out_rle = row['RLE']
			scaler = pickle.load(open('/scratch/parceirosbr/manntisict/radar/TEST_MODELS/oil_spill_segm/scaler_standarizate_oil_data.pkl','rb'))		
			image=read_img(path)
			image = np.asarray(image)                   
			size_img = image.shape
			image = image.reshape(-1,1)
			image = scaler.transform(image)
			image = np.reshape(image,(size_img[0],size_img[1]))


			mask = rle2mask(out_rle,(image.shape[1],image.shape[0]))
			window_size = (int(args.size),int(args.size))
			step_size = int(args.size*0.5)
			rois = []
			locs = []
			reconstructed_heatmp = np.zeros((image.shape[0],image.shape[1]))
			for (x, y, window) in sliding_window(image,step_size, window_size):
				if window.shape[0] != window_size[0] or window.shape[1] != window_size[1]:
					continue
				rois.append(np.stack((window,)*3, axis=-1))
				locs.append((x, y, x + window_size[0], y + window_size[0]))
			preds = model.predict(np.array(rois,dtype=np.float32))
			preds[:,:,:,1][preds[:,:,:,1]>args.limiar]=1
			preds = np.argmax(preds,axis=-1)

			result = cv2.imread(path, cv2.IMREAD_ANYDEPTH)

			for (i,label) in enumerate(preds):
				(startX,startY,endX,endY) = locs[i]
				reconstructed_heatmp[startY:endY,startX:endX] = label
			name = path.split('/')[-1].split('.')[:-1]
			print(name)
			if args.save_plot is True:

				COORD = np.where(mask==1)
				cv2.imwrite('{}/{}_inf.png'.format(Path_image,name[0]),result)
				ret, thresh = cv2.threshold(mask, 0, 1, cv2.THRESH_BINARY)
				contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
				cv2.drawContours(result, contours, -1, (255,0,0),1 )
				cv2.imwrite('{}/{}_inf.png'.format(Path_border,name[0]),result)
				cv2.imwrite('{}/{}_inf.png'.format(Path_mask,name[0]),mask*255)
				cv2.imwrite('{}/{}_inf.png'.format(Path_Inference,name[0]),reconstructed_heatmp*255)
			y_ref = reconstructed_heatmp.astype('uint8')
			print(np.unique(y_ref))
			#y_ref = cv2.medianBlur(y_ref,3)
			y_ref = y_ref.reshape(-1,1)
			y_true = mask
			y_true= y_true.reshape(-1,1) 
			target_names = ['0', '1']
			var_1 = classification_report(y_true, y_ref, target_names=target_names,output_dict=True)
			image_name.append(path)
			f1_score_out.append([var_1['0']['f1-score'],var_1['1']['f1-score'],var_1['macro avg']['f1-score'],var_1['weighted avg']['f1-score']])
			recall_out.append([var_1['0']['recall'],var_1['1']['recall'],var_1['macro avg']['recall'],var_1['weighted avg']['recall']])
			precision_out.append([var_1['0']['precision'],var_1['1']['precision'],var_1['macro avg']['precision'],var_1['weighted avg']['precision']])
			list_of_tuples_train  = list(zip(image_name,f1_score_out, recall_out ,precision_out))            
			report_df = pd.DataFrame(list_of_tuples_train, columns = ['paths', 'F1-score' ,'recall','precision'])
			report_df.to_csv('{}/report_metrics.csv'.format(args.out_path_inf), index = False, header=True)

def main(args):
	print('GPU:{}'.format(args.id_gpu))
	os.environ["CUDA_VISIBLE_DEVICES"]=args.id_gpu
	geodataframe = pd.read_csv('/scratch/parceirosbr/manntisict/radar/TEST_MODELS/oil_spill_segm/RLE_CSV/test_oil.csv')	
	geodataframe = geodataframe
	run_inference_batch(args,geodataframe)

if __name__ == '__main__':
      parser = argparse.ArgumentParser(description='Inference: Radar QLK images')
      parser.add_argument('--id_gpu', type=str, dest='id_gpu', help='id da gpu', metavar='id_gpu')
      parser.add_argument('--size', type=int, dest='size', help='Input size', metavar='size')
      parser.add_argument('--classes', type=int, dest='classes', help='Input classes', metavar='classes')
      parser.add_argument('--model_name', type=str, dest='model_name', help='Input model_name', metavar='model_name')
      parser.add_argument('--model_weigths_dir', type=str, dest='model_weigths_dir', help='Input model_weigths_dir', metavar='model_weigths_dir')
      parser.add_argument('--out_path_inf', type=str, dest='out_path_inf', help='out_path_inf', metavar='out_path_inf')
      parser.add_argument('--save_plot', type=bool, dest='save_plot', help='save_plot', metavar='save_plot', default= False)
      parser.add_argument('--split_train', type=int, dest='split_train', help='split_train', metavar='split_train', default=70)
      parser.add_argument('--limiar', type=float, dest='limiar', help='limiar', metavar='limiar', default=0.1)
      parser.add_argument('--stride', type=int, dest='stride', help='stride', metavar='stride', default=50)
      args = parser.parse_args()
      main(args)
