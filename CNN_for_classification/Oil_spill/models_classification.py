import tensorflow as tf
from tensorflow.keras.applications import vgg16, resnet50, xception , MobileNetV2, InceptionV3 , InceptionResNetV2 , NASNetLarge
#from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Activation ,BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D, GlobalAveragePooling2D,GlobalMaxPooling2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import SGD, Adam, Adadelta
import tensorflow.keras.backend as K
from metrics import *
from keras import initializers

def get_model(args,loss_function='binary_crossentropy',initial_lr=0.0001,weights="imagenet"):
  """
  Select model to classification
  parameters:
    args(argparse) = initial argsparse, contain information of input shape.
    loss_function(str) = define loss function.
    initial_lr(int) = define initial learning rate, using Adam optimizer
  return model compiled 
  """

  if args.model=='Xception':

    if args.optm=='Adam':
        optm = Adam(lr=0.00005,clipnorm=1.) 
    if args.optm=='SGD':
        optm = SGD(lr=0.0001, momentum=0.9, decay=0, nesterov=True)  
            
    pre_trained_model = xception.Xception(weights="/scratch/parceirosbr/bigoilict/share/Polen/clasificacion/weigths/xception_weights_tf_dim_ordering_tf_kernels_notop.h5",include_top=False, input_shape=(args.size, args.size, 3))
    x=pre_trained_model.output
    x=GlobalAveragePooling2D()(x)
    #x=Dropout(rate=0.5)(x)    
    x=Dense(2048, activation='relu')(x)
    #x=Dropout(rate=0.5)(x)
    x=Dense(1024, activation='relu')(x)
    #x=Dropout(rate=0.5)(x)
    x=Dense(512,activation='relu')(x) 
    x=Dropout(rate=0.2)(x)
    preds=Dense(args.classes,activation='softmax')(x) 
    model=Model(inputs=pre_trained_model.input, outputs=preds)

    for layer in pre_trained_model.layers:
        layer.trainable=False


    for layer in model.layers:
        if hasattr(layer, 'moving_mean') and hasattr(layer, 'moving_variance'):
            layer.trainable = True
            K.eval(K.update(layer.moving_mean, K.zeros_like(layer.moving_mean)))
            K.eval(K.update(layer.moving_variance, K.zeros_like(layer.moving_variance)))
        else:
            layer.trainable = False
            
    for layer in model.layers[129:]:
        layer.trainable=True
    #for layer in model.layers[:129]:
    #    layer.trainable=False            
    #for layer in model.layers[129:]:
    #    layer.trainable=True

    #for layer in model.layers:
    #    if hasattr(layer, 'moving_mean') and hasattr(layer, 'moving_variance'):
    #        layer.trainable = True
    #        K.eval(K.update(layer.moving_mean, K.zeros_like(layer.moving_mean)))
    #        K.eval(K.update(layer.moving_variance, K.zeros_like(layer.moving_variance)))
        #else:
        #    layer.trainable = False
            
    #for layer in model.layers[129:]:
    #    layer.trainable = True

    model.compile(loss=loss_function, optimizer=optm,metrics=['accuracy',f1,'AUC','MeanSquaredError'])
    model.summary()    
    return model

  if args.model=='Xception_1':

    if args.optm=='Adam':
        optm = Adam(lr=0.0001,clipnorm=1.) 
    if args.optm=='SGD':
        optm = SGD(lr=0.0001, momentum=0.9, decay=0, nesterov=True)  
            
    pre_trained_model = xception.Xception(weights="/scratch/parceirosbr/bigoilict/share/Polen/clasificacion/weigths/xception_weights_tf_dim_ordering_tf_kernels_notop.h5",include_top=False, input_shape=(args.size, args.size, 3))
    x=pre_trained_model.output
    x=GlobalAveragePooling2D()(x)
    #x=Dropout(rate=0.5)(x)    
    x=Dense(2048, activation='relu')(x)
    #x=Dropout(rate=0.5)(x)
    x=Dense(1024, activation='relu')(x)
    #x=Dropout(rate=0.5)(x)
    x=Dense(512,activation='relu')(x) 
    x=Dropout(rate=0.2)(x)
    preds=Dense(args.classes,activation='softmax')(x) 
    model=Model(inputs=pre_trained_model.input, outputs=preds)

    for layer in pre_trained_model.layers:
        layer.trainable=True


    for layer in model.layers:
       if hasattr(layer, 'moving_mean') and hasattr(layer, 'moving_variance'):
           layer.trainable = True
           K.eval(K.update(layer.moving_mean, K.zeros_like(layer.moving_mean)))
           K.eval(K.update(layer.moving_variance, K.zeros_like(layer.moving_variance)))
    #   else:
    #       layer.trainable = False

    #for layer in model.layers[:129]:
    #    layer.trainable=False            
    #for layer in model.layers[129:]:
    #    layer.trainable=True

    #for layer in model.layers:
    #    if hasattr(layer, 'moving_mean') and hasattr(layer, 'moving_variance'):
    #        layer.trainable = True
    #        K.eval(K.update(layer.moving_mean, K.zeros_like(layer.moving_mean)))
    #        K.eval(K.update(layer.moving_variance, K.zeros_like(layer.moving_variance)))
        #else:
        #    layer.trainable = False
            
    #for layer in model.layers[129:]:
    #    layer.trainable = True

    model.compile(loss=loss_function, optimizer=optm,metrics=['accuracy',f1,'AUC','MeanSquaredError'])
    model.summary()    
    return model


  if args.model=='InceptionV3':

    if args.optm=='Adam':
        optm = Adam(lr=0.0001,clipnorm=1.) 
    if args.optm=='SGD':
        optm = SGD(lr=0.0001, momentum=0.9, decay=0, nesterov=True,clipnorm=1.)  
            
    pre_trained_model = InceptionV3(input_shape=(args.size, args.size, 3), include_top=False, weights="/scratch/parceirosbr/bigoilict/share/Polen/clasificacion/weigths/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5")

    #last_layer = pre_trained_model.get_layer('mixed10')
    last_output = pre_trained_model.output#last_layer.output
    #x = GlobalMaxPooling2D()(last_output)
 
    x = GlobalAveragePooling2D()(last_output)
    # Add a fully connected layer with 512 hidden units and ReLU activation
    #x=Dropout(rate=0.5)(x)   
    x = Dense(2048, activation='relu')(x)
    #x=Dropout(rate=0.2)(x)
    x#=Dropout(rate=0.5)(x)    
    x=Dense(1024, activation='relu')(x)
    #x=Dropout(rate=0.5)(x)
    x=Dense(512,activation='relu')(x) 
    #x=Dropout(rate=0.2)(x)
    #x = Dense(512, activation='relu')(x)    
    x=Dropout(rate=0.2)(x)
    # Add a final sigmoid layer for classification
    x = Dense(args.classes, activation='softmax')(x)
    # Configure and compile the model

    model = Model(pre_trained_model.input, x)

    #for layer in pre_trained_model.layers:
    #    layer.trainable = True
    #for layer in model.layers:
    #    layer.trainable = True


    for layer in pre_trained_model.layers:
        layer.trainable=True

    #for layer in model.layers:
    #   if hasattr(layer, 'mixed10') and hasattr(layer, 'mixed10'):
    #       layer.trainable = True
    #       K.eval(K.update(layer.moving_mean, K.zeros_like(layer.moving_mean)))
    #       K.eval(K.update(layer.moving_variance, K.zeros_like(layer.moving_variance)))
        #else:
        #   layer.trainable = False
    for layer in model.layers[280:]:
        layer.trainable = True
   # for layer in pre_trained_model.layers:
    #    layer.trainable = False
    model.compile(loss=loss_function, optimizer=optm,metrics=['accuracy',f1,'AUC','MeanSquaredError'])
    model.summary()
    return model

  if args.model=='InceptionV3_1':

    if args.optm=='Adam':
        optm = Adam(lr=0.00005,clipnorm=1.) 
    if args.optm=='SGD':
        optm = SGD(lr=0.0001, momentum=0.9, decay=0, nesterov=True,clipnorm=1.)  
            
    pre_trained_model = InceptionV3(input_shape=(args.size, args.size, 3), include_top=False, weights="/scratch/parceirosbr/bigoilict/share/Polen/clasificacion/weigths/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5")

    #last_layer = pre_trained_model.get_layer('mixed10')
    last_output = pre_trained_model.output#last_layer.output
    #x = GlobalMaxPooling2D()(last_output)
 
    x = GlobalAveragePooling2D()(last_output)
    # Add a fully connected layer with 512 hidden units and ReLU activation
    x=Dropout(rate=0.5)(x)   
    #x = Dense(2048, activation='relu')(x)
    x=Dense(1024, activation='relu')(x)
    #x=Dropout(rate=0.2)(x)
    x=Dropout(rate=0.5)(x)    
    x=Dense(1024, activation='relu')(x)
    x=Dropout(rate=0.5)(x)
    x=Dense(512,activation='relu')(x) 
    #x=Dropout(rate=0.2)(x)
    #x = Dense(512, activation='relu')(x)    
    x=Dropout(rate=0.5)(x)
    # Add a final sigmoid layer for classification
    x = Dense(args.classes, activation='softmax')(x)
    # Configure and compile the model

    model = Model(pre_trained_model.input, x)

    #for layer in pre_trained_model.layers:
    #    layer.trainable = True
    #for layer in model.layers:
    #    layer.trainable = True


    for layer in pre_trained_model.layers:
        layer.trainable=False

    #for layer in model.layers:
    #   if hasattr(layer, 'mixed10') and hasattr(layer, 'mixed10'):
    #       layer.trainable = True
    #       K.eval(K.update(layer.moving_mean, K.zeros_like(layer.moving_mean)))
    #       K.eval(K.update(layer.moving_variance, K.zeros_like(layer.moving_variance)))
        #else:
        #   layer.trainable = False
    #for layer in model.layers[280:]:
    #    layer.trainable = True
   # for layer in pre_trained_model.layers:
    #    layer.trainable = False



    model.compile(loss=loss_function, optimizer=optm,metrics=['accuracy',f1,'AUC','MeanSquaredError'])
    model.summary()
    return model

  if args.model=='Resnet50':

    if args.optm=='Adam':
        optm = Adam(lr=0.00005,clipnorm=1.) 
    if args.optm=='SGD':
        optm = SGD(lr=0.0001, momentum=0.9, decay=0, nesterov=True,clipnorm=1.)  
            
    pre_trained_model = resnet50.ResNet50(weights="/scratch/parceirosbr/bigoilict/share/Polen/radar_temp/weigths/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5",include_top=False, input_shape=(args.size, args.size, 3))
    x=pre_trained_model.output
    x=GlobalAveragePooling2D()(x)
    x=Dropout(rate=0.3)(x)
    x=Dense(1024, activation='relu')(x)
    x=Dropout(rate=0.5)(x)
    x=Dense(512,activation='relu')(x) 
    preds=Dense(2,activation='softmax')(x) 

    model=Model(inputs=pre_trained_model.input, outputs=preds)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    #for layer in model.layers:
    #    if hasattr(layer, 'moving_mean') and hasattr(layer, 'moving_variance'):
    #        layer.trainable = True
    #        K.eval(K.update(layer.moving_mean, K.zeros_like(layer.moving_mean)))
    #        K.eval(K.update(layer.moving_variance, K.zeros_like(layer.moving_variance)))
    #    else:
    #        layer.trainable = False
    for layer in pre_trained_model.layers:
        layer.trainable=True      
    for layer in model.layers[165:]:
        layer.trainable=True           
    #for layer in model.layers[165:]:
    #    layer.trainable=True

    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(loss=loss_function, optimizer=optm,metrics=['accuracy',f1,'AUC','MeanSquaredError'])
    model.summary()
    return model

  if args.model=='vgg16':

    if args.optm=='Adam':
        optm = Adam(lr=0.0001,clipnorm=1.) 
    if args.optm=='SGD':
        optm = SGD(lr=0.0001, momentum=0.9, decay=0, nesterov=True,clipnorm=1.)  
            
    pre_trained_model = vgg16.VGG16(weights="/scratch/parceirosbr/bigoilict/share/Polen/radar_temp/weigths/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5",include_top=False, input_shape=(args.size, args.size, 3))
    x=pre_trained_model.output
    x=GlobalAveragePooling2D()(x)
    #x=Dense(1024,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
    x=Dropout(rate=0.5)(x)
    x=Dense(1024,activation='relu')(x) #dense layer 2
    x=Dropout(rate=0.5)(x)
    x=Dense(512,activation='relu')(x) #dense layer 3

    preds=Dense(2,activation='softmax')(x) #final layer with softmax activation

    model=Model(inputs=pre_trained_model.input, outputs=preds)

    for layer in pre_trained_model.layers:
        layer.trainable = True

    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(loss=loss_function, optimizer=optm,metrics=['accuracy',f1,'AUC','MeanSquaredError'])
    model.summary()
    return model

  if args.model=='MobileNetV2':

    if args.optm=='Adam':
        optm = Adam(lr=0.0001,clipnorm=1.) 
    if args.optm=='SGD':
        optm = SGD(lr=0.0001, momentum=0.9, decay=0, nesterov=True,clipnorm=1.)  
            
    pre_trained_model = MobileNetV2(weights='/scratch/parceirosbr/bigoilict/share/Polen/clasificacion/weigths/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5',include_top=False, input_shape=(args.size, args.size, 3))
    x=pre_trained_model.output
    x=GlobalAveragePooling2D()(x)
    x=Dense(2048,activation='relu')(x)
    x=Dense(1024,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
    #preds=Dense(2,activation='softmax')(x) #final layer with softmax activation
    x=Dense(512,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
    preds=Dense(2,activation='softmax')(x) #final layer with softmax activation
    x=Dropout(rate=0.2)(x)
    model=Model(inputs=pre_trained_model.input, outputs=preds)
    for layer in pre_trained_model.layers:
        layer.trainable = True

    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(loss=loss_function, optimizer=optm,metrics=['accuracy',f1,'AUC','MeanSquaredError'])
    model.summary()
    return model

  if args.model=='MobileNetV2_1':

    if args.optm=='Adam':
        optm = Adam(lr=0.0002,clipnorm=1.) 
    if args.optm=='SGD':
        optm = SGD(lr=0.0001, momentum=0.9, decay=0, nesterov=True,clipnorm=1.)  
            
    pre_trained_model = MobileNetV2(weights='/scratch/parceirosbr/bigoilict/share/Polen/clasificacion/weigths/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5',include_top=False, input_shape=(args.size, args.size, 3))
    x=pre_trained_model.output
    x=GlobalAveragePooling2D()(x)
    x=Dense(1024,activation='relu')(x)
    x=Dense(1024,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
    x=Dense(512,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
    preds=Dense(2,activation='softmax')(x) #final layer with softmax activation
    model=Model(inputs=pre_trained_model.input, outputs=preds)

    for layer in model.layers[:20]:
        layer.trainable=True
    for layer in model.layers[20:]:
        layer.trainable=True

    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(loss=loss_function, optimizer=optm,metrics=['accuracy',f1,'AUC','MeanSquaredError'])
    model.summary()
    return model

  if args.model=='inception_resnet_v2':

    if args.optm=='Adam':
        optm = Adam(lr=0.0001,clipnorm=1.) 
    if args.optm=='SGD':
        optm = SGD(lr=0.0001, momentum=0.9, decay=0, nesterov=True,clipnorm=1.)  
            
    pre_trained_model = InceptionResNetV2(weights='/scratch/parceirosbr/bigoilict/share/Polen/radar_temp/weigths/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5',include_top=False, input_shape=(args.size, args.size, 3))

    for layer in pre_trained_model.layers:
        if hasattr(layer, 'moving_mean') and hasattr(layer, 'moving_variance'):
            layer.trainable = True
            K.eval(K.update(layer.moving_mean, K.zeros_like(layer.moving_mean)))
            K.eval(K.update(layer.moving_variance, K.zeros_like(layer.moving_variance)))
        else:
            layer.trainable = False
    for layer in pre_trained_model.layers:
        layer.trainable = True
        
    last_layer = pre_trained_model.get_layer('conv_7b_ac')
    last_output = last_layer.output
    x = GlobalMaxPooling2D()(last_output)
    # Add a fully connected layer with 512 hidden units and ReLU activation
    #x=Dense(1024, activation='relu')(x)
    #x=Dropout(rate=0.5)(x)
    x = Dense(1024, activation='relu')(x)
    #x=Dropout(rate=0.5)(x)
    x = Dense(512, activation='relu')(x)
    # Add a dropout rate of 0.7
    x = Dropout(0.2)(x)
    # Add a final sigmoid layer for classification
    x = Dense(2, activation='softmax')(x)

    # Configure and compile the model
    model = Model(pre_trained_model.input, x)

    model.compile(loss=loss_function, optimizer=optm,metrics=['accuracy',f1,'AUC','MeanSquaredError'])
    model.summary()
    return model

  if args.model=='NASNetLarge':

    if args.optm=='Adam':
        optm = Adam(lr=0.0001,clipnorm=1.) 
    if args.optm=='SGD':
        optm = SGD(lr=0.0001, momentum=0.9, decay=0, nesterov=True,clipnorm=1.)  
            
    pre_trained_model = NASNetLarge(weights='/scratch/parceirosbr/bigoilict/share/Polen/radar_temp/weigths/nasnet_large_no_top.h5',include_top=False, input_shape=(args.size, args.size, 3))


    x=pre_trained_model.output
    x=GlobalAveragePooling2D()(x)
    x=Dropout(rate=0.5)(x)    
    x=Dense(2048, activation='relu')(x)
    x=Dropout(rate=0.5)(x)
    x=Dense(1024, activation='relu')(x)
    x=Dropout(rate=0.5)(x)
    x=Dense(512,activation='relu')(x) 
    #x=Dropout(rate=0.5)(x)
    preds=Dense(2,activation='softmax')(x) 
    model=Model(inputs=pre_trained_model.input, outputs=preds)

    for layer in pre_trained_model.layers:
        layer.trainable=False

    model.compile(loss=loss_function, optimizer=optm,metrics=['accuracy',f1,'AUC','MeanSquaredError'])
    model.summary()
    return model