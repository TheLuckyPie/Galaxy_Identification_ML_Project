### run.py

# This is the main driving script for structuring the code of the project.

import os

import tensorflow as tf
import numpy as np
import random
import pandas as pd
import glob

#get your util functions
#from eval import evaluate_my_model
#from models import MyCustomModel
from utils import config_load, import_parameters, set_dirs, decode_downsample, get_labels, img_label, trim_file_list, load_data, split_data


#from models import myfunction1, myfunction2, ...

def run(): 
    #Prompt to select config:
    #config = config_load(input('Input file name')+".yaml")
    config = config_load("configbase.yaml")

    #Import Config parameters. See README for information about each config option
    seed, val_split, batch_size, img_size, color, oneHot, confidence, classes = import_parameters(config)

    #Set Directories and make results directory if it doesn't exist
    data_dir, image_dir, label_file, res_dir = set_dirs(config)
    os.makedirs(res_dir, exist_ok=True)

    #Seed Configuration to produce reproducible code
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Load the dataset
    dataset = load_data(image_dir, label_file, classes, confidence, oneHot, img_size)

    #Split dataset into training, validation and test set
    train_data, valid_data, test_data = split_data(dataset, val_split)

    """""
    # Preprocess all of the data based on the train set

    # Define the model
    model = MyCustomModel()

    # Fit the model

    # Evaluate
    evaluate_my_model(model, test_data)
 """
#if __name__ == '__main__':
#config = config_load(input('Input file name')+".yaml")

run()
