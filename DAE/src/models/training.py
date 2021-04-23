import sys
sys.path.append('../data/')
sys.path.append('../features/')
sys.path.append('../utils/')
import Loader
import Processing
from Loader import load_mnist
from Processing import normalize,add_white_noise
from tools import peak_signal_noise_rateo,plot_img
from tensorflow.keras.layers import Dense,Input
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import Conv2D,Conv2DTranspose, MaxPooling2D
from Model import build_autoencoder,build_conv_autoencoder

#%%
# train and evaluate a deep autoencoder
# load the data ar array
X = load_mnist(download = True,as_img = False)
# first normalize and after add noise
X_norm = normalize(X)
X_noise = add_white_noise(X_norm)
X_train, X_val, Y_train, Y_val = train_test_split(X_noise,X_norm,test_size = 0.2)
X_train, X_test, Y_train, Y_test = train_test_split(X_train,Y_train,test_size = 0.2)


autoencoder = build_autoencoder()

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor="val_loss",min_delta=0.0001,patience=5,restore_best_weights=True)

autoencoder.compile(optimizer = 'adam',loss = 'mse',metrics=[peak_signal_noise_rateo])

autoencoder.fit(X_train,Y_train,epochs=30,batch_size=64,shuffle=True,validation_data=(X_val,Y_val),callbacks=[early_stopping])

# plot image to have a visual comparison for the autoencoder work
plot_img(5,autoencoder,X_test,Y_test)
# numeric value to test the autoencoder work
peak_signal_noise_rateo(Y_test[:3],autoencoder.predict(X_test[:3][np.newaxis,:]))

#%%
# train and evaluate a deep convolutional autoencoder
# load data as image 28x28
X = load_mnist(download = True,as_img = True)

X_norm = normalize(X)
X_noise = add_white_noise(X_norm)
X_train, X_val, Y_train, Y_val = train_test_split(X_noise,X_norm,test_size = 0.2)
X_train, X_test, Y_train, Y_test = train_test_split(X_train,Y_train,test_size = 0.2)


conv_autoencoder = build_conv_autoencoder()
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor="val_loss",min_delta=0.0001,patience=5,restore_best_weights=True)

conv_autoencoder.compile(optimizer = 'adam',loss = 'binary_crossentropy',metrics=['mse',peak_signal_noise_rateo])

conv_autoencoder.fit(X_train,Y_train,epochs=30,batch_size=64,shuffle=True,validation_data=(X_val,Y_val),callbacks=[early_stopping])

# plot image to have a visual comparison for the autoencoder work
plot_img(5,conv_autoencoder,X_test,Y_test)
# numeric value to test the autoencoder work
peak_signal_noise_rateo(Y_test[:3],conv_autoencoder.predict(X_test[:3][np.newaxis,:]))
