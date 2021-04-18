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

#%%
import importlib
importlib.reload(Loader)

X = load_mnist(download = True,as_img = False)
X.shape
X_norm = normalize(X)
X_noise = add_white_noise(X_norm)
X_train, X_test, Y_train, Y_test = train_test_split(X_noise,X_norm,test_size = 0.2)


def build_encoder(encoding_dim):

    input_img = Input(shape=(28*28,))
    encoded = Dense(128, activation='relu')(input_img)
    encoded = Dense(64, activation='relu')(encoded)
    encoded = Dense(encoding_dim, activation='relu')(encoded)

    encoder = Model(input_img,encoded)
    return encoder

def build_decoder(encoding_dim):

    input_img = Input(shape=(28*28,))
    encoder = build_encoder(encoding_dim)
    input_encoded = encoder(input_img)

    decoded = Dense(64, activation='relu')(input_encoded)
    decoded = Dense(128, activation='relu')(decoded)
    decoded = Dense(28*28,activation='relu')(decoded)

    decoder = Model(input_img,decoded)
    return decoder

def build_autoencoder(encoding_dim=32):
    decoder = build_decoder(encoding_dim)
    return decoder


autoencoder = build_autoencoder()

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor="val_loss",min_delta=0.0001,patience=5,restore_best_weights=True)
# This model maps an input to its reconstruction

autoencoder.compile(optimizer = 'adam',loss = 'mse',metrics=['mae'])


autoencoder.fit(X_train,Y_train,epochs=30,batch_size=64,shuffle=True,validation_data=(X_test,Y_test),callbacks=[early_stopping])

plt.imshow(np.reshape(autoencoder.predict(X_test[0][np.newaxis,:]),(28,28)))


plt.imshow(np.reshape(Y_test[0],(28,28)))

#%%
X = load_mnist(download = True,as_img = True)
X.shape
X_norm = normalize(X)
X_noise = add_white_noise(X_norm)
X_train, X_test, Y_train, Y_test = train_test_split(X_noise,X_norm,test_size = 0.2)

from tensorflow.keras.layers import Conv2D,Conv2DTranspose, MaxPooling2D,UpSampling2D
#%%
input_img = Input(shape=(28,28,1))


encoder = Conv2D(32,(3,3),activation='relu',padding='same')(input_img)
encoder = MaxPooling2D((2,2),padding = 'same')(encoder)
encoder = Conv2D(32,(3,3),activation='relu',padding='same')(encoder)
encoder = MaxPooling2D((2,2),padding = 'same')(encoder)

decoder = Conv2D(32,(3,3),activation='relu',padding='same')(encoder)
decoder = UpSampling2D((2,2))(decoder)
decoder = Conv2D(32,(3,3),activation='relu',padding='same')(decoder)
decoder = UpSampling2D((2,2))(decoder)
decoder = Conv2D(1,(3,3),activation='sigmoid',padding='same')(decoder)

conv_autoencoder = Model(input_img,decoder)

conv_autoencoder.compile(optimizer='adam',loss = 'binary_crossentropy')

conv_autoencoder.fit(X_train,Y_train,batch_size = 256,shuffle=True,epochs=10,validation_data=(X_test,Y_test),callbacks=[early_stopping])

plt.imshow(np.reshape(conv_autoencoder.predict(X_test[3][np.newaxis,:]),(28,28)))


plt.imshow(np.reshape(Y_test[3],(28,28)))
