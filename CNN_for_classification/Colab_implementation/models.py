import tensorflow as tf
from keras.applications import vgg16, resnet50, xception , MobileNetV2, InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.layers import Input, Dense, Dropout, Flatten, Activation, merge
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D, GlobalAveragePooling2D,GlobalMaxPooling2D
from keras.models import Sequential, Model
from keras.optimizers import SGD, Adam
import keras.backend as K
from metrics import *
from keras.layers import BatchNormalization


def get_model(args,loss_function='binary_crossentropy',initial_lr=0.0001):
  if args.model=='VGG16':


    pre_trained_model = vgg16.VGG16(input_shape=(args.size, args.size, 3), include_top=False, weights="imagenet")
    for layer in pre_trained_model.layers:
        layer.trainable = False
    last_layer = pre_trained_model.get_layer('block5_pool')
    last_output = last_layer.output
    x = GlobalMaxPooling2D()(last_output)
    # Add a fully connected layer with 512 hidden units and ReLU activation
    x = Dense(512, activation='relu')(x)
    # Add a dropout rate of 0.7
    x = Dropout(0.5)(x)
    # Add a final sigmoid layer for classification
    x = Dense(args.classes, activation='softmax')(x)
    # Configure and compile the model
    model = Model(pre_trained_model.input, x)
    decay_rate = 0
    momentum = 0.9
    opt_SGD = SGD(lr=initial_lr, momentum=momentum, decay=decay_rate, nesterov=False)
    opt_Adam = Adam(lr=initial_lr) 

    model.compile(loss=loss_function, optimizer=opt_Adam,metrics=['accuracy',f1])
    model.summary()
    return model

  if args.model=='ResNet50':
    pre_trained_model  = resnet50.ResNet50(weights='imagenet',include_top=False, input_shape=(args.size, args.size, 3))

    output = pre_trained_model.layers[-1].output
    output = Flatten()(output)
    pre_trained_model = Model(pre_trained_model.input, output=output)
    for layer in pre_trained_model.layers:
        layer.trainable = False

    x=pre_trained_model.output
    x=Dense(1024, activation='relu')(x)
    x=Dropout(rate=0.5)(x)
    x=Dense(512,activation='relu')(x) 
    preds=Dense(args.classes,activation='softmax')(x)
    model=Model(inputs=pre_trained_model.input, outputs=preds)
    decay_rate = 0
    momentum = 0.9

    opt_SGD = SGD(lr=initial_lr, momentum=momentum, decay=decay_rate, nesterov=False)
    opt_Adam = Adam(lr=initial_lr) 

    model.compile(loss=loss_function, optimizer=opt_Adam,metrics=['accuracy',f1])
    model.summary()    
    return model

  if args.model=='Xception':
    pre_trained_model = xception.Xception(weights="imagenet",include_top=False, input_shape=(args.size, args.size, 3))
    x=pre_trained_model.output
    x=GlobalAveragePooling2D()(x)
    x=Dense(1024, activation='relu')(x)
    x=Dropout(rate=0.5)(x)
    x=Dense(1024, activation='relu')(x)
    x=Dropout(rate=0.5)(x)
    x=Dense(512,activation='relu')(x) 
    preds=Dense(args.classes,activation='softmax')(x) 
    model=Model(inputs=pre_trained_model.input, outputs=preds)
    for layer in pre_trained_model.layers:
        layer.trainable=True
    for layer in model.layers:
        if hasattr(layer, 'moving_mean') and hasattr(layer, 'moving_variance'):
            layer.trainable = True
            K.eval(K.update(layer.moving_mean, K.zeros_like(layer.moving_mean)))
            K.eval(K.update(layer.moving_variance, K.zeros_like(layer.moving_variance)))
        else:
            layer.trainable = False
            
    for layer in model.layers[129:]:
        layer.trainable=True

    decay_rate = 0
    momentum = 0.9

    opt_SGD = SGD(lr=initial_lr, momentum=momentum, decay=decay_rate, nesterov=False)
    opt_Adam = Adam(lr=0.0001) 

    model.compile(loss=loss_function, optimizer=opt_Adam,metrics=['accuracy',f1])
    model.summary()    
    return model


  if args.model=='MobileNetV2':

    pre_trained_model = MobileNetV2(weights='imagenet',include_top=False, input_shape=(args.size, args.size, 3))
    x=pre_trained_model.output
    x=GlobalAveragePooling2D()(x)
    x=Dense(512,activation='relu')(x) 
    preds=Dense(args.classes,activation='softmax')(x) 
    model=Model(inputs=pre_trained_model.input, outputs=preds)
    decay_rate = 0
    momentum = 0.9
    opt_SGD = SGD(lr=initial_lr, momentum=momentum, decay=decay_rate, nesterov=False)
    opt_Adam = Adam(lr=initial_lr) 
    model.compile(loss=loss_function, optimizer=opt_Adam,metrics=['accuracy',f1])
    model.summary()
    return model

  if args.model=='Inception':

    pre_trained_model = InceptionV3(input_shape=(args.size, args.size, 3), include_top=False, weights="imagenet")

    for layer in pre_trained_model.layers:
        if hasattr(layer, 'moving_mean') and hasattr(layer, 'moving_variance'):
            layer.trainable = True
            K.eval(K.update(layer.moving_mean, K.zeros_like(layer.moving_mean)))
            K.eval(K.update(layer.moving_variance, K.zeros_like(layer.moving_variance)))
        else:
            layer.trainable = False
    last_layer = pre_trained_model.get_layer('mixed10')
    last_output = last_layer.output
    x = GlobalMaxPooling2D()(last_output)
    # Add a fully connected layer with 512 hidden units and ReLU activation
    x = Dense(1024, activation='relu')(x)
    x=Dropout(rate=0.5)(x)
    x = Dense(512, activation='relu')(x)
    # Add a dropout rate of 0.7
    x = Dropout(rate=0.5)(x)
    # Add a final sigmoid layer for classification
    x = Dense(args.classes, activation='softmax')(x)
    # Configure and compile the model

    model = Model(pre_trained_model.input, x)
    decay_rate = 0
    momentum = 0.9
    opt_SGD = SGD(lr=initial_lr, momentum=momentum, decay=decay_rate, nesterov=False)
    opt_Adam = Adam(lr=initial_lr) 
    model.compile(loss=loss_function, optimizer=opt_Adam,metrics=['accuracy',f1])
    model.summary()
    return model


  if args.model=='InceptionResNet':

    pre_trained_model = InceptionResNetV2(input_shape=(args.size, args.size, 3), include_top=False, weights="imagenet")

    for layer in pre_trained_model.layers:
        if hasattr(layer, 'moving_mean') and hasattr(layer, 'moving_variance'):
            layer.trainable = True
            K.eval(K.update(layer.moving_mean, K.zeros_like(layer.moving_mean)))
            K.eval(K.update(layer.moving_variance, K.zeros_like(layer.moving_variance)))
        else:
            layer.trainable = False

    last_layer = pre_trained_model.get_layer('conv_7b_ac')
    last_output = last_layer.output
    x = GlobalMaxPooling2D()(last_output)
    # Add a fully connected layer with 512 hidden units and ReLU activation
    x=Dense(1024, activation='relu')(x)
    x=Dropout(rate=0.5)(x)
    x = Dense(1024, activation='relu')(x)
    x=Dropout(rate=0.5)(x)
    x = Dense(512, activation='relu')(x)
    # Add a dropout rate of 0.7
    x = Dropout(0.5)(x)
    # Add a final sigmoid layer for classification
    x = Dense(args.classes, activation='softmax')(x)

    # Configure and compile the model
    model = Model(pre_trained_model.input, x)
    decay_rate = 0
    momentum = 0.9
    opt_SGD = SGD(lr=initial_lr, momentum=momentum, decay=decay_rate, nesterov=False)
    opt_Adam = Adam(lr=initial_lr) 

    model.compile(loss=loss_function, optimizer=opt_Adam,metrics=['accuracy',f1])
    model.summary()
    return model

  if args.model=='Custom':

    inputShape= (args.size, args.size,3)
    model=Sequential()

    model.add(Conv2D(64, (3,3), activation = 'relu', input_shape = inputShape))
    model.add(MaxPooling2D(2,2))
    model.add(BatchNormalization(axis =-1))
    model.add(Dropout(0.5))

    model.add(Conv2D(64, (3,3), activation = 'relu', input_shape = inputShape))
    model.add(MaxPooling2D(2,2))
    model.add(BatchNormalization(axis =-1))
    model.add(Dropout(0.5))

    model.add(Conv2D(32, (3,3), activation = 'relu'))
    model.add(MaxPooling2D(2,2))
    model.add(BatchNormalization(axis = -1))
    model.add(Dropout(0.5))

    model.add(Conv2D(32, (3,3), activation = 'relu'))
    model.add(MaxPooling2D(2,2))
    model.add(BatchNormalization(axis = -1))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(2048, activation = 'relu'))
    model.add(Dropout(0.5))   
    model.add(Dense(1024, activation = 'relu'))
    model.add(Dropout(0.5))     
    model.add(Dense(512, activation = 'relu'))
    model.add(BatchNormalization(axis = -1))
    model.add(Dropout(0.5))
    model.add(Dense(args.classes, activation = 'softmax'))
    decay_rate = 0
    momentum = 0.9
    opt_SGD = SGD(lr=initial_lr, momentum=momentum, decay=decay_rate, nesterov=False)
    opt_Adam = Adam(lr=initial_lr) 

    model.compile(loss=loss_function, optimizer=opt_Adam,metrics=['accuracy',f1])
    model.summary()

    return model
