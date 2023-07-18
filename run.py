### run.py

# This is the main driving script for structuring the code of the project.

import os

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import random
import pandas as pd
import glob
from sklearn.metrics import confusion_matrix, classification_report

#get your util functions
#from eval import evaluate_my_model
#from models import MyCustomModel
#from utils import config_load, import_parameters
#from models import Model_Builder
tf.get_logger().setLevel('ERROR')

#from models import myfunction1, myfunction2, ...

def run(): 
    #Prompt to select config:
    #config = config_load(input('Input file name')+".yaml")
    config = config_load("configbase.yaml")

    #Import Config parameters. See README for information about each config option
    seed, dataset_param, image_param, label_param, model_param, training_param = import_parameters(config)

    #Set Directories and make results directory if it doesn't exist
    data_dir, image_dir, label_file, res_dir = set_dirs(config)

    #Seed Configuration to produce reproducible code
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Load the dataset
    dataset = load_data(image_dir, label_file, label_param, image_param) 

    #Split dataset into training, validation and test set and shuffle if required
    train_data, valid_data, test_data = split_data(dataset, dataset_param, seed) #dataset_param['val_split'], dataset_param['batch_size'], dataset_param['shuffle'], seed)


    #Defining Model
    model = Model_Builder(image_param, model_param, model_param['augments'], seed)

    model.compile(optimizer='adam',
        loss=training_param['comploss'], #TYPE?
        metrics=['accuracy', 'AUC'])

    #Fitting Model
    history, model_res_filepath, checkpoint_filepath = Model_Fitter(model, train_data, valid_data, model_param, training_param, res_dir)

    #Evaluating + Plotting to Results Folder
    plot_history(history, model_res_filepath, training_param, model_param)
    #model.load_weights(model_res_filepath)
    plot_roc_auc(model, test_data, model_res_filepath)
    #model.evaluate(test_data)
    
#if __name__ == '__main__':
#config = config_load(input('Input file name')+".yaml")

run()
