'''
Copyright

Jelen forráskód a Budapesti Műszaki és Gazdaságtudományi Egyetemen tartott
"Deep Learning a gyakorlatban Python és LUA alapon" tantárgy segédanyagaként készült.

A tantárgy honlapja: http://smartlab.tmit.bme.hu/oktatas-deep-learning
Deep Learning kutatás: http://smartlab.tmit.bme.hu/deep-learning

A forráskódot GPLv3 licensz védi. Újrafelhasználás esetén lehetőség szerint kérjük
az alábbi szerzőt értesíteni.

2018 (c) Csapó Tamás Gábor (csapot kukac tmit pont bme pont hu),
Gyires-Tóth Bálint, Zainkó Csaba


Links:
    [talos] https://github.com/autonomio/talos
'''

import talos

import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping, CSVLogger
import numpy as np

# do not use all GPU memory
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))


(x_train, y_train), (x_test, y_test) = cifar10.load_data()

num_classes = 10

# one-hot enkódolttá alakítás
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# átalakítás FC-DNN-hez
x_train = np.reshape(x_train, (50000, 3072))  # 32x32
x_test = np.reshape(x_test, (10000, 3072))
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# normalizálás, [0-1]
x_train /= 255
x_test /= 255

from keras.activations import relu, elu, softmax

# talos hiperparaméter optimalizáláshoz kell
p = {
    'first_neuron': [128, 256, 512],
    'hidden_layers': [0, 1, 2],
    'dropout': [0, 0.1, 0.2, 0.3, 0.4, 0.5],
    'activation': [relu, elu],
    'last_activation': ['softmax'],
    'optimizer': ['adam', 'sgd', 'rmsprop'],
    'batch_size': [64, 128, 256]
}



# add input parameters to the function
def do_training(x_train, y_train, x_val, y_val, params):
    # replace the hyperparameter inputs with references to params dictionary

    model = Sequential()
    model.add(Dense(params['first_neuron'], activation=params['activation'], input_dim=3072))
    model.add(Dropout(params['dropout']))
    
    model.add(Dense(10, activation=params['last_activation']))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    callbacks = [EarlyStopping(monitor='val_acc', patience=3, verbose=0)]

    # make sure history object is returned by model.fit()
    out = model.fit(x_train, y_train,
                    batch_size=params['batch_size'],
                    epochs=100,
                    validation_data=(x_val, y_val),
                    callbacks=callbacks,
                    verbose=2,
                    shuffle=True)

    return out, model


t = talos.Scan(x_train, y_train,
               params=p,
               model=do_training,
               grid_downsample=.1)
