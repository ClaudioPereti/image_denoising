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
from Model import build_autoencoder,build_conv_autoencoder

#%%

X = load_mnist(download = True,as_img = False)

X_norm = normalize(X)
X_noise = add_white_noise(X_norm)
X_train, X_val, Y_train, Y_val = train_test_split(X_noise,X_norm,test_size = 0.2)
X_train, X_test, Y_train, Y_test = train_test_split(X_train,Y_train,test_size = 0.2)


autoencoder = build_autoencoder()

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor="val_loss",min_delta=0.0001,patience=5,restore_best_weights=True)

autoencoder.compile(optimizer = 'adam',loss = 'mse',metrics=['mae'])

autoencoder.fit(X_train,Y_train,epochs=30,batch_size=64,shuffle=True,validation_data=(X_val,Y_val),callbacks=[early_stopping])

#%%
def plot_img(n_fig ,autoencoder,X_test,Y_test):
    """
    return 3 series of n_fig image; in order: the ground truth, the corrupted one with the noise and the one reconstructed with the autoencoder

    Parameters:
            n_fig(int): number of figure to plot
            autoencoder(Keras model): autoencoder that denoise the image
            X_test(numpy array): numpy array with the noised image
            y_test(numpy array): numpy array with the image
    Returns:
            3 series of n_fig figure to show the quality of the autoencoder

    """

    plt.figure( figsize=(20,4))
    random_numbers = np.random.randint(0,X_test.shape[0],size=n_fig)
    for i in range(n_fig):
        index = random_numbers[i]

        ax = plt.subplot(3,n_fig,i+1)
        plt.imshow(np.reshape(Y_test[index],(28,28)))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(3,n_fig,i+1+n_fig)
        plt.imshow(np.reshape(X_test[index],(28,28)))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(3,n_fig,i+1+2*n_fig)
        plt.imshow(np.reshape(autoencoder.predict(X_test[index][np.newaxis,:]),(28,28)))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

plot_img(5,autoencoder,X_test,Y_test)
#%%

X = load_mnist(download = True,as_img = True)

X_norm = normalize(X)
X_noise = add_white_noise(X_norm)
X_train, X_val, Y_train, Y_val = train_test_split(X_noise,X_norm,test_size = 0.2)
X_train, X_test, Y_train, Y_test = train_test_split(X_train,Y_train,test_size = 0.2)


autoencoder = build_conv_autoencoder()
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor="val_loss",min_delta=0.0001,patience=5,restore_best_weights=True)

autoencoder.compile(optimizer = 'adam',loss = 'binary_crossentropy',metrics=['mse'])

autoencoder.fit(X_train,Y_train,epochs=30,batch_size=64,shuffle=True,validation_data=(X_val,Y_val),callbacks=[early_stopping])
plot_img(5,autoencoder,X_test,Y_test)
