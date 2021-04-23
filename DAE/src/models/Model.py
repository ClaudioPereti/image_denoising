import sys
sys.path.append('../data/')
sys.path.append('../features/')
import Loader
import Processing
from Loader import load_mnist
from Processing import normalize,add_white_noise
from tensorflow.keras.layers import Dense,Input
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import Conv2D,Conv2DTranspose, MaxPooling2D
from tensorflow.keras import regularizers


def build_encoder(encoding_dim,sparse):
    """build and return the encoder with a specified encoding dimension"""

    input_img = Input(shape=(28*28,))
    encoded = Dense(128, activation='relu')(input_img)
    encoded = Dense(64, activation='relu')(encoded)
    if sparse:
        encoded = Dense(encoding_dim, activation='relu',activity_regularizer=regularizers.l1(10e-5))(encoded)
    else:
        encoded = Dense(encoding_dim, activation='relu',activity_regularizer=regularizers.l1(10e-5))(encoded)

    encoder = Model(input_img,encoded)
    return encoder

def build_decoder(encoding_dim,sparse):
    """"build and return the decoder linked with the encoder"""

    input_img = Input(shape=(28*28,))
    encoder = build_encoder(encoding_dim,sparse)
    input_encoded = encoder(input_img)

    decoded = Dense(64, activation='relu')(input_encoded)
    decoded = Dense(128, activation='relu')(decoded)
    decoded = Dense(28*28,activation='relu')(decoded)

    decoder = Model(input_img,decoded)
    return decoder

def build_autoencoder(encoding_dim=32,sparse = False):
    """build a deep autoencoder

    Args:
        encoding_dim (int): The dimension of the encoder output (default is 32)
        sparse (bool): a flag used to turn the autoencoder in an sparse autoencoder (default is False)

    Returns:
        decoder (keras functional model): the autoencoder built linking decoder and encoder
    """
    decoder = build_decoder(encoding_dim,sparse)
    return decoder


def build_conv_encoder():
    """build and return the convolutional encoder"""
    input_img = Input(shape=(28,28,1))


    encoded = Conv2D(32,(3,3),activation='relu',padding='same')(input_img)
    encoded = MaxPooling2D((2,2),padding = 'same')(encoded)
    encoded = Conv2D(32,(3,3),activation='relu',padding='same')(encoded)
    encoded = MaxPooling2D((2,2),padding = 'same')(encoded)

    encoder = Model(input_img,encoded)
    return encoder

def bottleneck(input_encoded):
    """build and return the bottleneck in a convolutional autoencoder"""

    bottle_neck = Conv2D(34,(3,3),activation='relu',padding='same')(input_encoded)

    return bottle_neck

def build_conv_decoder():
    """build and return the conv decoder linked with the bottleneck and the encoder"""

    input_img = Input(shape=(28,28,1))
    encoder = build_conv_encoder()
    input_encoded = encoder(input_img)
    bottle_neck = bottleneck(input_encoded)

    decoded = Conv2D(32,(3,3),activation='relu',padding='same')(bottle_neck)
    decoded = Conv2DTranspose(32,(3,3),(2,2),padding='same')(decoded)
    decoded = Conv2D(32,(3,3),activation='relu',padding='same')(decoded)
    decoded = Conv2DTranspose(1,(3,3),(2,2),padding='same')(decoded)
    decoded = Conv2D(1,(3,3),activation='sigmoid',padding='same')(decoded)

    decoder = Model(input_img,decoded)
    return decoder

def build_conv_autoencoder():
        """build a deep convolutional autoencoder

    Args:

    Returns:
        conv_autoencoder (keras functional model): the convolutional autoencoder built linking conv. decoder and conv. encoder
    """
    conv_autoencoder = build_conv_decoder()
    return conv_autoencoder
