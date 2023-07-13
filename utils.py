### Utils.py

# This is the script containing utility function to declutter run.py

import os
import yaml

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
        
        val_split = config['val_split']
        batch_size = config['batch_size']

        image_param = config['image_prop']
        img_size = image_param['img_size']
        color = image_param['color']

        label_param = config['label_prop']
        oneHot = label_param['onehotcoded']
        confidence = label_param['confidence']
        classes = label_param['classes']
        
        return seed, val_split, batch_size, img_size, color, oneHot, confidence, classes

def set_dirs(config):
        """
        Function to set directories.
        """

        data_dir = config['datafolder']
        image_dir = os.path.join(data_dir, config['imgdir'])
        label_file = os.path.join(data_dir, config['labels'])
        res_dir = config['resdir']

        return data_dir, image_dir, label_file, res_dir

def decode_downsample(data,size=64):
        """
        Function to take a file path (tensor) open the image, decode, downsample, and return as greyscale size = size in w and h for downsample
        """
        img = tf.io.read_file(data)
        img = tf.io.decode_jpeg(img,channels=3)
        img = tf.image.resize(img, [size, size])

        return img

def get_labels(label_file, classes, confidence, oneHot):
        """
        Function to import labels and if one-hot labels are required (1.0 or 0.0) based on confidence, then set labels accordingly.
        """

        labels = pd.read_csv(label_file).set_index('GalaxyID')[classes]
        labels = labels[(labels[classes] > confidence).any(axis=1)]
        if oneHot == True:
               labels = labels.applymap(lambda x: 1.0 if x > confidence else 0.0)
 
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

def load_data(image_dir, label_file, classes, confidence, oneHot, img_size):
        """
        Function to filter labels, only consider images that are in filtered labels list, creates an image and label dataset and combines them into a single one.
        """

        labels = get_labels(label_file, classes, confidence, oneHot)
        
        files = glob.glob(f'{image_dir}/*')
        files = trim_file_list(files, labels)
        print(files[:5])

        image_ds = tf.data.experimental.from_list(files).map(decode_downsample)
        label_ds = tf.data.experimental.from_list([img_label(f,labels) for f in files])
        dataset = tf.data.Dataset.zip((image_ds,label_ds))

        return dataset

def split_data(dataset, val_split):
        """
        Function to split dataset into train, validation and test dataset based on val_split = test_split percentage
        """
        dataset_len = len(dataset)
        val_len = int(dataset_len*val_split)

        val_ds = dataset.take(val_len).batch(64)
        test_ds = dataset.skip(val_len).take(val_len).batch(64)
        train_ds = dataset.skip(2*val_len).batch(64)

        return train_ds, val_ds, train_ds



"""
class MyUsefulClass():
    """Define useful objects."""
    def __init__(self, x):
        self.x = x

    def useful_method(self):
        return self.x * 2
"""