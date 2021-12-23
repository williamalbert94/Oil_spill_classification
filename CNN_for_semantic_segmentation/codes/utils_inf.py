import cv2
import tensorflow as tf
from tensorflow.keras import Model
import numpy as np

def read_img(img_path):
    img=cv2.imread(img_path, cv2.IMREAD_ANYDEPTH)
    return np.asarray(img)

def patch_inference(path, size='512',over='50',model=None,limiar=0.07):
    """
    Take an patch and execute inference process:
    path(str): input patch of image.
    size(int): size of patch.
    over(float): stride for inference.
    model(tensorflow object): compiled model.
    limiar(float): minimum limiar for probability matrix.
    return inference for all image, using patch inference.
    """
    over=int(over)
    size=int(size)
    image1=read_img(path)
    image1 = (np.asarray(image1)-np.min(image1))/(np.max(image1)-np.min(image1))
    reconstructed_heatmp = np.zeros((image1.shape[0],image1.shape[1],1))
    value=(over/100)*size
    stride=value*(size)
    acumulador=0
    acumulador2=0

    coordenadasx=[]
    coordenadasy=[]
       
    while acumulador2<(image1.shape[1]):

        while acumulador<(image1.shape[0]):
            inde_x=[0+acumulador ,size+acumulador ]
            inde_y=[0+acumulador2 ,size+acumulador2 ]

            if inde_x[1] > image1.shape[0]:
                
                dim1_start = image1.shape[0] - size
                dim1_end = image1.shape[0]
                inde_x = [dim1_start,dim1_end]
                
            if inde_y[1] > image1.shape[1]:
                dim2_start = image1.shape[1] - size
                dim2_end = image1.shape[1]
                inde_y = [dim2_start,dim2_end]
            
            recorte=image1[inde_x[0]:inde_x[1], inde_y[0]:inde_y[1]]
            recorte = recorte.reshape(1, recorte.shape[0], recorte.shape[1], 1)               

            if model is not None:
                x = image1[inde_x[0]:inde_x[1], inde_y[0]:inde_y[1]]
                x=np.stack((x,)*3, axis=-1)
                x = np.expand_dims(x,axis=0)

                preds = model.predict(x)
                preds[:,:,:,1][preds[:,:,:,1]>limiar]=1
                preds = np.argmax(preds,axis=-1) 
              
                preds = preds[0,:,:]
                preds = np.expand_dims(preds,axis=-1)
                reconstructed_heatmp[inde_x[0]:inde_x[1], inde_y[0]:inde_y[1]]=preds 
                        
                
            acumulador=int(acumulador+value)
        acumulador2=int(acumulador2+value)
        acumulador=0
    return reconstructed_heatmp