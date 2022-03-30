import tensorflow as tf
from tensorflow import keras


def get_classifier(nlayers, nunits, nclasses):
    """
    A function to define a classifier
    :return: The model object.
    """
    return 0


class MyCustomLayer(keras.layers.Layer):
    """If you need to define custom layers, then do it here."""
    def __init__(self, n_nodes):
        super(MyCustomLayer, self).__init__()


class MyCustomModel(keras.model.Model):
    """Custom models should also be defined here."""
    def __init__(self, n_layers=3, n_nodes=128):
        super(MyCustomModel, self).__init__()
