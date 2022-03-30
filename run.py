# This is the main driving script for structuring the code of the project.
# The aim of structuring a project in the way we outline in the way we demonstrate here is that it should be easy to
# read along for people who want to know what you have done. Breaking code into chunks is also a good way to make
# code easier to maintian/extend but also easier for you to collaborate on projects with others.
# What we demonstrate here isn't the best possible way to work on a project, but it should be a simple easy way to get
# started.
# The below is a skeleton of code that can be changed however you want. The arguments to classes and functions here are
# what placeholders, you can change this as you see fit.
import os

import tensorflow as tf
import numpy as np
import random

#get your util functions
from eval import evaluate_my_model
from models import MyCustomModel
from utils import load_data, split_data


#from models import myfunction1, myfunction2, ...

def run():
    # Set the seeds so that your code will be reproducible!
    seed = 42
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Set the location where you can load data from, and where you can dump results
    data_directory = 'path/to/data/'
    # For every experiment you run, you will want to change this somehow, such that results are never overwritten
    results_directory = 'path/to/results/'
    os.makedirs(results_directory, exist_ok=True)

    # Load the data into memory
    data = load_data()

    # Define training/validation/test sets
    train_data, valid_data, test_data = split_data(data)

    # Preprocess all of the data based on the train set

    # Define the model
    model = MyCustomModel()

    # Fit the model

    # Evaluate
    evaluate_my_model(model, test_data)

if __name__ == '__main__':
    run()