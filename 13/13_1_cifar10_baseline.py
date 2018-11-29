# !pip3 install hyperas

# based on https://github.com/keras-team/keras/blob/master/examples/cifar10_cnn.py

import keras
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


from keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

batch_size = 32
num_classes = 10
epochs = 100

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# reshape for FC-DNN
x_train = np.reshape(x_train,(50000,3072)) # 32x32
x_test = np.reshape(x_test,(10000,3072))
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Normalization of pixel values (to [0-1] range)

x_train /= 255
x_test /= 255

model = Sequential()
model.add(Dense(256, activation='relu', input_dim=3072))
model.add(Dense(256, activation='relu'))
model.add(Dense(10, activation='softmax'))
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(optimizer=sgd,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

callbacks = [EarlyStopping(monitor='val_acc', patience=3, verbose=0)]

model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              callbacks=callbacks,
              verbose=2,
              shuffle=True)
