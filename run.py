### run.py

# This is the main driving script for structuring the code of the project.

#import required packages
import tensorflow as tf
import random
import os

#Importing all external functions
from utils import *
from models import *
from evalu import * 
from models import *

#Suppress Tensorflow Messages
tf.get_logger().setLevel('ERROR')


def run(): 
    #TOADD: Prompt to select config; config = config_load(input('Input file name')+".yaml")
    config = config_load("configplaygroundbinary.yaml")
    print("Importing Configuration File: " + str(config) + "...")

    #Import Config parameters. See README for information about each config option
    seed, dataset_param, image_param, label_param, model_param, training_param = import_parameters(config)

    #Set Directories and make results directory if it doesn't exist
    data_dir, image_dir, label_file, res_dir = set_dirs(config)

    #Seed Configuration to produce reproducible code
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    print("Imported Configuration File: " + str(config) + "!")
    print("Loading Data and Configuring Dataset...")

    # Load the dataset
    dataset = load_data(image_dir, label_file, label_param, image_param) 

    #Split dataset into training, validation and test set.
    train_data, valid_data, test_data = split_data(dataset, dataset_param, seed)
    
    print("Data Loaded!")    
    print("Building Model...")    
    
    #Defining Model
    model = Model_Builder(image_param, model_param, model_param['augments'], seed)
    model.compile(optimizer='adam', loss=training_param['comploss'], metrics=['accuracy', 'AUC'])

    print("Built Model!")
    print("Fitting Model...") 

    #Fitting Model
    history, model_res_filepath, checkpoint_filepath, training_time = Model_Fitter(model, train_data, valid_data, model_param, training_param, res_dir)
    
    print("Fit Model in " + f"time: {training_time[0]} min {training_time[1]} sec")
    print("Plotting Evaluation Metrics and saving data to results path: ("+ str(model_res_filepath)+ ")\n\n")
    
    #Evaluating + Plotting to Results Folder
    get_eval(history, model_res_filepath, training_param, model_param, model, test_data, dataset_param, training_time)
    
    #SAVEMODEL
    if model_param['savemodel']:
        print("Saving Model...")
        model.save(os.path.join(model_res_filepath,model_param['name']+"_savedModel"))
        print("Saved Model!")

    tf.keras.backend.clear_session()
    print("Done!") 
    
#if __name__ == '__main__':
#config = config_load(input('Input file name')+".yaml")

run()
