### models.py

#This is the file which defines Models

import tensorflow as tf
from tensorflow import keras
from keras import layers, Sequential
from keras.layers import Input, Dense, Layer
from keras.models import Model

def Model_Builder(image_param, model_param, augments, seed): #Model_Builder(img_size, color, model_name = 'Model', classes, bnActive = False, doActive = False, dropoutrate = 0.25, agActive = False, Rescaling = False):
    #List of Filters
    FilterList = [32,64]
    
    #Defining Input for Model
    inputs = Input(shape=(image_param['img_size'],image_param['img_size'],1))
    x = inputs

    #Adding Augment Layers if Active in Config
    if augments['active']:
        x = layers.RandomRotation(augments['rotation'], seed = seed)(x)
        x = layers.RandomFlip(seed=seed)(x)
        x = layers.RandomContrast(augments['contrast'], seed= seed)(x)
        x = layers.RandomZoom(augments['zoom'], seed = seed)(x)

    #Normalizing to (0,1) if allowed
    if model_param['rescale']:
        x = layers.Rescaling(1./255)(x)

    #Adding standard CNN layers
    for i in range(len(FilterList)):
        x = layers.Conv2D(FilterList[i], (3,3), padding = 'valid', activation = 'relu')(x)
        if model_param['batchnormactive']:
            x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2,2))(x)
   
   #Flattening
    x = layers.Flatten()(x)

    #Adding Droupout Layers if Active
    if model_param['dropoutactive']:
        x = layers.Dropout(model_param['dropoutrate'])(x)
   
    #Fully connected Layers
    x = layers.Dense(64, activation = 'relu')(x)
    
    #TO ADD: REGRESSION OPTIONS

    #Defining Output for Model
    outputs = layers.Dense(model_param['outputlevels'], activation = model_param['outputactivation'])(x)

    #Defining Model based on created layers
    model = Model(inputs, outputs, name = model_param['name'])

    return model