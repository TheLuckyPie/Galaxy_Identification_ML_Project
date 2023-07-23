# models.py
import os
import tensorflow as tf
from keras import layers
from keras.layers import Input
from keras.models import Model
from datetime import datetime
import time
import numpy as np

tf.get_logger().setLevel('ERROR')

def Model_Builder(image_param, model_param, augments, seed, dataset_param): #Model_Builder(img_size, color, model_name = 'Model', classes, bnActive = False, doActive = False, dropoutrate = 0.25, agActive = False, Rescaling = False):
    """
    Function to Build the Model based on configuration parameters 
    """
    #List of Filters
    FilterList = model_param['filters']
    DenseFilterList = model_param['densefilters']
    
    #Defining Input for Model
    if image_param['color'] == 'grayscale':
        inputs = Input(shape=(image_param['img_size'],image_param['img_size'],1))
    else: 
        inputs = Input(shape=(image_param['img_size'],image_param['img_size'],3))
    x = inputs

    #Adding Augment Layers if Active in Config
    if augments['active']:
        x = layers.RandomRotation(augments['rotation'], seed = seed)(x)
        x = layers.RandomFlip(seed=seed)(x)
        x = layers.RandomContrast(augments['contrast'], seed= seed)(x)
        x = layers.RandomZoom(augments['zoom'], seed = seed)(x)
        x = layers.Rescaling(1./255)(x)

    #Adding standard CNN layers
    for i in range(len(FilterList)):
        x = layers.Conv2D(FilterList[i], (3,3), padding = 'valid', activation = 'relu')(x)
        x = layers.MaxPooling2D((2,2))(x)
   
    #Flattening
    x = layers.Flatten()(x)

    #Fully connected Layers and adding Dropout layers if Active
    for i in range(len(DenseFilterList)):
        x = layers.Dense(DenseFilterList[i], activation ='relu')(x)
        if model_param['dropoutactive']:
            x = layers.Dropout(model_param['dropoutrate'])(x)

    #Defining Output for Model
    outputs = layers.Dense(model_param['outputlevels'], activation = model_param['outputactivation'])(x)

    #Defining Model based on created layers
    model = Model(inputs, outputs, name = model_param['name'])

    return model


def Model_Fitter(model, train_data, valid_data, model_param, training_param, res_dir,dataset_param):
    """
        Function to fit the Model, making sure to save checkpoints and model evaluation parameters 
    """
    #Defining Model Subdirectories for each Model, creating if required and Fitting model while noting training time.
    model_res_filepath = os.path.join(res_dir,model_param['name']+datetime.now().strftime("_%m_%d__%H_%M"))
    checkpoint_filepath = os.path.join(model_res_filepath, 'checkpoints')
    os.makedirs(model_res_filepath, exist_ok=True)
    os.makedirs(checkpoint_filepath, exist_ok=True)
    checkpoint_file = os.path.join(checkpoint_filepath, model_param['name'] + '_best_weights.h5')
    #Checkpoints
    if dataset_param['regressionmodel']: 
        MonitorVal = 'val_loss'
        MonitorMode = 'min'
    else:
        MonitorVal = 'val_auc'
        MonitorMode = 'max'
    
    model_checkpoints = tf.keras.callbacks.ModelCheckpoint(checkpoint_file, save_best_only=True, monitor = MonitorVal, mode = MonitorMode)
    csv_logger = tf.keras.callbacks.CSVLogger(os.path.join(model_res_filepath,model_param['name'] + '_log.csv'))
    
    #Defining callback
    if training_param['earlystopactive']:
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=training_param['earlystoppatience'], mode='min', verbose=1)
        callbacks=[model_checkpoints, csv_logger, early_stopping]
    else:
        callbacks=[model_checkpoints, csv_logger] 
    
    #Fitting Data and recording time elapsed
    start_time = time.time()
    history = model.fit(train_data, epochs=training_param['epochs'], validation_data = valid_data, callbacks = callbacks, shuffle = training_param['shuffle'])
    

    if training_param['earlystopactive']:
        num_epochs_trained = early_stopping.stopped_epoch + 1

    end_time = time.time()
    train_time = end_time - start_time
    training_time = np.array([int(train_time // 60), int(train_time % 60)])

    return history, model_res_filepath, checkpoint_file, training_time