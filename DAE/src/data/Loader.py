from tensorflow.keras.datasets import mnist
import numpy as np
import pandas as pd

def load_mnist(download = False,as_img = True):
    """
    return a numpy array containing mnist dataset

    Parameters:
        download (bool): True to download mnist dataset, False to load a saved version
        as_img (bool): True to reshape data in array of matrice 28*28, False for 754 dimensional data

   Returns:
       numpy array: mnist dataset

    """
    if download:

        ((X_train,_),(X_test,_))= mnist.load_data()
        X = np.vstack([X_train,X_test])

        return X

    else:

        X = pd.read_csv('../../data/mnist.csv', index_col=0)
        X = np.array(X)

        if as_img:
            #image are 28x28
            X = np.reshape(X,(70000,28,28))

    return X
