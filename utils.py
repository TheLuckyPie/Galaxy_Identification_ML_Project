### Utils.py

# This is the script containing utility function to declutter run.py

import os
import yaml
import tensorflow as tf
import glob
import pandas as pd
import matplotlib.pyplot as plt

def config_load(config_name):
        """
        Function to load YAML config file.
        """

        config_path = os.path.join('configs', config_name)
        with open(config_path,"r") as config_file:
                config = yaml.safe_load(config_file)

        return config


def import_parameters(config):
        """
        Function to parameters from specified config file.
        """

        seed = config['seed']
        #Loading different parameter categories
        dataset_param = config['datasetparam']
        image_param = config['imageparam']
        label_param = config['labelparam']
        model_param = config['modelparam']
        training_param = config['trainingparam']

        return seed, dataset_param, image_param, label_param, model_param, training_param

def set_dirs(config):
        """
        Function to set directories.
        """

        data_dir = config['datafolder']
        image_dir = os.path.join(data_dir, config['imgdir'])
        label_file = os.path.join(data_dir, config['labels'])
        res_dir = config['resdir']
        os.makedirs(res_dir, exist_ok=True)

        return data_dir, image_dir, label_file, res_dir

def decode_downsample(data,size=64,color='grayscale'):
        """
        Function to take a file path (tensor) open the image, decode, downsample, and return as greyscale size = size in w and h for downsample
        """

        img = tf.io.read_file(data)
        img = tf.io.decode_jpeg(img,channels=3)
        img = tf.image.resize(img, [size, size])
        if color == 'grayscale':
                img = tf.reduce_mean(img,axis=-1,keepdims=True)

        return img

def get_labels(label_file, classes, confidence, oneHot):
        """
        Function to import labels and if one-hot labels are required (1.0 or 0.0) based on confidence, then set labels accordingly.
        """

        labels = pd.read_csv(label_file).set_index('GalaxyID')[classes]
        labels = labels[(labels[classes] > confidence).any(axis=1)]
        if oneHot == True:
                labels = labels.applymap(lambda x: 1.0 if x > confidence else 0.0)
                labels['Class'] = np.argmax(labels.values, axis=1) #Converts [1,0] or [0,1] rows into 0 or 1 respectively
                labels.drop(columns=labels.columns[:-1], inplace=True)

        return labels

def img_label(img,labels):
        """
         Function to turn image path into ID, and return labels from labels
        """

        id = int(img.split('\\')[-1].split('.')[0])

        return labels.loc[id]

def trim_file_list(files,labels):
        """
        Function to trim a list of files based on whether the file name is in the ID of labels_df
        """

        files = [file for file in files if int(file.split('\\')[-1].split('.')[0]) in labels.index]
        
        return files

def load_data(image_dir, label_file, label_param, image_param): # classes, confidence, oneHot, img_size, color) 
        """
        Function to filter labels, only consider images that are in filtered labels list, creates an image and label dataset and combines them into a single one.
        """

        labels = get_labels(label_file, label_param['classes'], label_param['confidence'], label_param['onehotcoded'])
        
        files = glob.glob(f'{image_dir}/*')
        files = trim_file_list(files, labels)
        print(files[:5])

        
        image_ds = tf.data.experimental.from_list(files).map(lambda x: decode_downsample(x, image_param['img_size'], image_param['color']), num_parallel_calls = tf.data.AUTOTUNE)
        label_ds = tf.data.experimental.from_list([img_label(f,labels) for f in files])
        dataset = tf.data.Dataset.zip((image_ds,label_ds))

        return dataset

def split_data(dataset, dataset_param, seed): #dataset_param['val_split'], dataset_param['batch_size'], dataset_param['shuffle'], seed)
        """
        Function to shuffle, split dataset into train, validation and test dataset based on val_split = test_split percentage
        """
        #if dataset_param['shuffle']:
        #        dataset = tf.random.shuffle(dataset, seed = seed)

        dataset_len = len(dataset)
        val_len = int(dataset_len*dataset_param['val_split'])

        val_ds = dataset.take(val_len).batch(dataset_param['batch_size'])
        test_ds = dataset.skip(val_len).take(val_len).batch(dataset_param['batch_size'])
        train_ds = dataset.skip(2*val_len).batch(dataset_param['batch_size'])

        return train_ds, val_ds, test_ds


def get_preds_and_labels(model, dataset_param, test_data):
        """
        Function to split test_data into test_labels and get prediction labels from model
        """
        test_labels = list(test_data.unbatch().map(lambda im, lab : lab).as_numpy_iterator())
        test_predictions = model.predict(test_data, batch_size = dataset_param['batch_size'])

        return test_labels, test_predictions
