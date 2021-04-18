
import sys
sys.path.append('../data/')
from Loader import load_mnist
import numpy as np


def normalize(X):
    """ return a numpy array normalized """

    X = X/255.0
    return X

def add_white_noise(X,factor = 1):
    """ return a numpy array with white noise added (normal distribution mean 0 var 1)"""
    #distinguishes dataset of 28*28 image and 754 array

    if len(X.shape) == 3:
        noise = np.random.randn(X.shape[0],X.shape[1],X.shape[2])

    if len(X.shape) == 2:
        noise = np.random.randn(X.shape[0],X.shape[1])

    X_noise = X+factor*noise
    return X_noise
