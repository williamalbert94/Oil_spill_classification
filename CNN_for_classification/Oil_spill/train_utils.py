import tensorflow.keras.backend as K
import tensorflow as tf
import os
from os.path import join, exists, dirname
from glob import glob
import numpy as np
import cv2
from numpy import linalg as la
from shutil import copyfile
import matplotlib.pyplot as plt
from PIL import Image 
Image.MAX_IMAGE_PIXELS = 933120000

def get_min_max(path_train):
  """
  Finds the minimum and maximum value of the data, starting from a directory.
  """
  mode_to_bpp = {'1':1, 'L':8, 'P':8, 'RGB':24, 'RGBA':32, 'CMYK':32, 'YCbCr':24, 'I':32, 'F':32 ,'I;16': 16, 
                  "I;16B": 16, "I;16L": 16, "I;16S": 16, "I;16BS": 16, "I;16LS": 16, "I;32": 32, "I;32B": 32, "I;32L": 32, "I;32S": 32, "I;32BS": 32, "I;32LS": 32}
  path = path_train
  path_image = join(path,'images')
  list_image = glob(os.path.join(path_image,'*.tif'))
  result_max = 0
  result_min = 0
  for img_dir in list_image:
    temp_data = Image.open(img_dir)
    bpp = mode_to_bpp[temp_data.mode]
    temp_max = 2**(bpp)
    if temp_max > result_max:
      result_max = temp_max
    
  return result_min, result_max

def try_to_copy_file(current_path, new_path):
  """
  Move lbl.txt for use in conversion stage.
  """
  try:
    copyfile(current_path, new_path)
  except:
    print("---- ERRO EM COPIAR labels.txt ----")
    raise

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph

def get_loss(weights=None, use_weights=True):
    """
    Get a balanced loss using array of weigths
    """
    if use_weights:
        weights = K.variable(weights)

    def loss(y_true, y_pred):
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        if use_weights:
            loss = y_true * K.log(y_pred) * weights
        else:
            loss = y_true * K.log(y_pred)
        loss = -K.sum(loss, -1)
        return loss
    return loss

def binary_crossentropy_with_logits(ground_truth, predictions):
    """
    Get a balanced loss using  balanced binary logits
    """    
    return K.mean(K.binary_crossentropy(ground_truth, predictions,from_logits=True),axis=-1)

def weighted_cross_entropy(weights):
    """
    Get a balanced loss using cross entropy logits
    """
    def loss(y_true, y_pred):
        loss = K.nn.weighted_cross_entropy_with_logits(targets=y_true, logits=y_pred, pos_weight=weights)
        return loss
    return loss

def get_mean_iou(nclasses):
    """
    Get mean of Intersection over Union, this method is used only in training stage for tf 2.x
    Update metric for each step in epoch.
    """
    def mean_iou(y_true, y_pred):
        nb_classes = K.int_shape(y_pred)[-1]
        true_pixels = K.argmax(y_true, axis=-1)
        pred_pixels = K.argmax(y_pred, axis=-1)
        iou = []
        for i in range(1, nb_classes):
            true_labels = K.equal(true_pixels, i)
            pred_labels = K.equal(pred_pixels, i)
            m=tf.keras.metrics.MeanIoU(num_classes=nclasses)
            m.update_state(y_true=true_labels, y_pred=pred_labels)
            res=m.result()
            iou.append(res)
        iou = tf.stack(iou)
        return iou
    return mean_iou

def get_weights(directory, classes, perPixel, enable):
  """
  get distribution of weigthsm based in number of pixel per class or number of images.
  parameters:
    directory(str): path of dataset.
    classes(int): total of classes of study.
    perpixel,enable (int): defin if use at pixel or image level.
  return array of weigths
  """
  if enable:
    totalClass = np.zeros(classes)
    totalPixels = np.zeros(classes)
    totalWeights = np.zeros(classes)
    # Set path to files
    masks_path = join(directory, 'masks', 'masks')
    image_pattern = join(masks_path, '*.png')
    image_lst = glob(image_pattern)
    n_images = len(image_lst)    
    print (n_images)
    j=0
    for image in image_lst:
      j += 1
      img = cv2.imread(image, 0)
      nClasses = img.max()
      for i in range(nClasses+1):
        temp = np.where(img == i)
        nPixels = len(temp[0])
        if nPixels > 0:
          totalClass[i] += 1
          totalPixels[i] += nPixels
    
    totalClassDetected = len(np.where(totalClass>=1)[0])
    totalDetected = np.where(totalClass>=1)[0]
    totalClass = np.where(totalClass>=1, totalClass, 0)
    totalPixels = np.where(totalPixels>=1, totalPixels, 0)
    temp = []
    for i in totalDetected:
      if perPixel:
        temp.append(totalPixels[i])
      else:
        temp.append(totalClass[i])
    temp = np.asarray(temp)
    normT = la.norm(temp)
    weights = normT/temp

    c = 0
    for i in totalDetected:
      totalWeights[i] = weights[c]
      c += 1

    print ('Total classes detected:', totalClassDetected)
    print ('Detected classes:', totalDetected)
    if perPixel:
      return totalWeights, totalPixels
    else:
      return totalWeights, totalClass
  else:
    totalWeights = np.ones(classes)
    return totalWeights, 0

def get_weights_v2(train_df, classes, perPixel, enable):
  """
  get distribution of weigthsm based in number of pixel per class or number of images using dataframe file.
  parameters:
    train_df(str): path of dataframe.
    classes(int): total of classes of study.
    perpixel,enable (int): defin if use at pixel or image level.
  return array of weigths
  """
  if enable:
    totalClass = np.zeros(classes)
    totalPixels = np.zeros(classes)
    totalWeights = np.zeros(classes)
    image_lst = train_df["masks_filename"]
    
    n_images = len(image_lst)
    print (n_images)
    j=0
    for image in image_lst:
      j += 1
      img = cv2.imread(image, 0)
      nClasses = img.max()
      for i in range(nClasses+1):
        temp = np.where(img == i)
        nPixels = len(temp[0])
        if nPixels > 0:
          totalClass[i] += 1
          totalPixels[i] += nPixels
    
    totalClassDetected = len(np.where(totalClass>=1)[0])
    totalDetected = np.where(totalClass>=1)[0]
    totalClass = np.where(totalClass>=1, totalClass, 0)
    totalPixels = np.where(totalPixels>=1, totalPixels, 0)
    temp = []
    for i in totalDetected:
      if perPixel:
        temp.append(totalPixels[i])
      else:
        temp.append(totalClass[i])
    temp = np.asarray(temp)
    normT = la.norm(temp)
    weights = normT/temp

    c = 0
    for i in totalDetected:
      totalWeights[i] = weights[c]
      c += 1

    print ('Total classes detected:', totalClassDetected)
    print ('Detected classes:', totalDetected)
    if perPixel:
      return totalWeights, totalPixels
    else:
      return totalWeights, totalClass
  else:
    totalWeights = np.ones(classes)
    return totalWeights, 0

def apply_conversion(data, conversion):
    """
    convert mask to class.
    parameters:
      data(array) : annotation data
      conversion(array) : array of representations for each class
    return label mask
    """    
    result = np.copy(data)
    for key, value in conversion.items():        
        result = np.where(data == key, value, result)
    return result

def approach2classes(data_type, masks, conversion):
    """
    generate global classes depending to approach.
    parameters:
      history(dict) : history of training
      no_plot(bool) : flag to save figure
      model_basename(str) : model name
    return save training graph
    """
    result = np.copy(masks)
    if data_type == 3:
        print('major')        
        result[masks==1] = conversion[1]
        result[masks==2] = conversion[2]
        result[masks==3] = conversion[3]
        result[masks==4] = conversion[4]
        result[masks==5] = conversion[5]
    return result

def save_training_plot(history,no_plot,model_basename):
  """
  Save graph of training
  parameters:
    history(dict) : history of training
    no_plot(bool) : flag to save figure
    model_basename(str) : model name
  return save training graph
  """
  if not no_plot:
    fig = plt.figure(figsize=(10,6))
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy',fontsize=20)
    plt.ylabel('Accuracy',fontsize=20)
    plt.xlabel('Epoch',fontsize=20)
    plt.legend(['train', 'val'], loc='center right',fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.tick_params(axis='both', which='minor', labelsize=18)
    plt.show()
    plt.savefig(model_basename+'_ACC.png')

    fig = plt.figure(figsize=(10,6))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss',fontsize=20)
    plt.ylabel('Loss',fontsize=20)
    plt.xlabel('Epoch',fontsize=20)
    plt.legend(['train', 'val'], loc='center right',fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.tick_params(axis='both', which='minor', labelsize=18)
    plt.show()
    plt.savefig(model_basename+'_LOSS.png')