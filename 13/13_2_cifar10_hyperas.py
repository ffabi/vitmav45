# !pip3 install hyperas

# based on https://github.com/keras-team/keras/blob/master/examples/cifar10_cnn.py

import hyperas

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


# hiperparaméter optimalizálás hyperas-sal (https://github.com/maxpumperla/hyperas)

# a hyperas-nak kell két függvény:
# -- data() : adatok betöltése
# -- create_model() : hálózat modell

def data():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    num_classes = 10

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

    return x_train, y_train, x_test, y_test


def create_model(x_train, y_train, x_test, y_test):
    
    n_layer1 = {{choice([128, 256, 512])}}
    n_layer2 = {{choice([128, 256, 512])}}
    dropout_1 = {{uniform(0, 1)}}
    dropout_2 = {{uniform(0, 1)}}
    optim = {{choice(['rmsprop', 'adam', 'sgd'])}}
    n_batch = {{choice([64, 128, 256])}}
    
    print('Model hyperparameters: ', n_layer1, n_layer2, dropout_1, dropout_2, optim, n_batch)
    # 3 x 3 x [0-1]x[0-1] x 3 x 3  = kb 8100 kombináció
    
    model = Sequential()
    model.add(Dense(n_layer1, activation='relu', input_dim=3072))
    model.add(Dropout(dropout_1))
    model.add(Dense(n_layer2, activation='relu'))
    model.add(Dropout(dropout_2))
    model.add(Dense(10, activation='softmax'))
    
    model.compile(optimizer=optim,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    import datetime
    current_date = '{date:%Y-%m-%d_%H-%M-%S}'.format(date=datetime.datetime.now())
    print(current_date)
    csv_name = '13_hyperas_cifar10_' + current_date + '_' + \
               str(n_layer1) + '_' + str(n_layer2) + '_' + \
               str(dropout_1) + '_' + str(dropout_2) + '_' + \
               str(optim) + '_' + str(n_batch) + '.csv'
    callbacks = [EarlyStopping(monitor='val_loss', patience=3, verbose=0), \
                 CSVLogger(csv_name, append=True, separator=';')]
    
    result = model.fit(x_train, y_train,
              batch_size=n_batch,
              epochs=100,
              verbose=2,
              validation_data=(x_test, y_test),
              callbacks=callbacks,
              shuffle=True)
    
    validation_acc = np.amax(result.history['val_acc']) 
    print('Best validation acc of epoch:', validation_acc)
    return {'loss': -validation_acc, 'status': STATUS_OK, 'model': model}

from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform

# main hyperopt part
# az algoritmus lehet:
# -- random.suggest -> random search
# -- tpe.suggest -> tree parsen estimator
best_run, best_model = optim.minimize(model=create_model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=5,
                                          trials=Trials())
x_train, y_train, x_test, y_test = data()
print("Evalutation of best performing model:")
print(best_model.evaluate(x_test, y_test))
print("Best performing model chosen hyper-parameters:")
print(best_run)

