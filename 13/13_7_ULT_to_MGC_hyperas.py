# adatok letöltése
# !wget http://smartlab.tmit.bme.hu/csapo/dl/spkr048_ult.tgz
# !tar -xzvf spkr048_ult.tgz

import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.callbacks import EarlyStopping, CSVLogger

# do not use all GPU memory
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import pickle

from subprocess import run





#### adatok betöltése
def data():
    # globális paraméterek
    order = 24
    n_lines = 64
    n_pixels_reduced = 64
    n_mgc = order + 1

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    ult = np.fromfile('spkr048_train_valid_ult.dat', dtype='uint8')
    mgc = np.fromfile('spkr048_train_valid_mgc.dat', dtype='float32')

    # bemenet: ultrahang képek átméretezése
    # 4D, hogy a CNN bemenetének megfelelő legyen
    ult = np.reshape(ult, (-1, n_lines, n_pixels_reduced, 1))
    print('ult shape: ', ult.shape)

    # kimenet: mgc átméretezése
    mgc = np.reshape(mgc, (-1, n_mgc))
    print('mgc shape: ', mgc.shape)

    # train-validation split
    ult_training, ult_validation, mgc_training, mgc_validation = train_test_split(ult, mgc, test_size=0.1, random_state=19)

    # bemenet: [0-1 között]
    ult_training = ult_training / 255.0
    ult_validation = ult_validation / 255.0

    # kimenet: nulla várható érték, egységnyi szórás
    # jellemzőnkénti normalizálás

    mgc_scalers = []
    for i in range(n_mgc):
        mgc_scaler = StandardScaler(with_mean=True, with_std=True)
        mgc_scalers.append(mgc_scaler)
        mgc_training[:, i] = mgc_scalers[i].fit_transform(mgc_training[:, i].reshape(-1, 1)).ravel()
        mgc_validation[:, i] = mgc_scalers[i].transform(mgc_validation[:, i].reshape(-1, 1)).ravel()

    pickle.dump(mgc_scalers, open('mgc_scalers.sav', 'wb'))

    x_train = ult_training
    y_train = mgc_training
    x_test = ult_validation
    y_test = mgc_validation

    return x_train, y_train, x_test, y_test


### hálózat összerakása és tanítás
def create_model(x_train, y_train, x_test, y_test):

    # globális paraméterek
    order = 24
    n_lines = 64
    n_pixels_reduced = 64
    n_mgc = order + 1

    n_conv_layers = {{choice([0, 1, 2])}}
    n_layers = {{choice([1, 2, 3, 4])}}
    n_layer = [{{choice([250, 500, 750, 1000])}}, {{choice([250, 500, 750, 1000])}}, {{choice([250, 500, 750, 1000])}}, {{choice([250, 500, 750, 1000])}}]
    dropout = [{{uniform(0, 0.8)}}, {{uniform(0, 0.8)}}, {{uniform(0, 0.8)}}, {{uniform(0, 0.8)}}]
    act = {{choice(['tanh', 'relu'])}}
    optim = {{choice(['adam', 'rmsprop', 'adamax'])}}
    n_batch = {{choice([16, 32, 64, 128, 256])}}
    
    # hálózat összerakása
    model = Sequential()
    # first layer, if CNN
    if n_conv_layers > 0:
        model.add(Conv2D(8, kernel_size=(3, 3), activation=act, input_shape=(n_lines,n_pixels_reduced, 1), padding='same'))
        model.add(Dropout(0.05))
        model.add(MaxPooling2D((3,3), padding='same'))
        
        for n in range(1, n_conv_layers):
            model.add(Conv2D(32, kernel_size=(3, 3), activation=act, padding='same'))
            model.add(Dropout(0.05))
            model.add(MaxPooling2D((3,3), padding='same'))
        
        model.add(Flatten())

    # first layer, if DNN
    else:
        model.add(Flatten(input_shape=(n_lines, n_pixels_reduced, 1)))
        model.add(Dense(n_layer[0], kernel_initializer='normal', activation=act))
        model.add(Dropout(dropout[0]))
    
    # 2-3-4th layers are optional
    for n in range(1, n_layers):
        model.add(Dense(n_layer[n], kernel_initializer='normal', activation=act))
        model.add(Dropout(dropout[n]))
    for n in range(n_layers, 4):
        n_layer[n] = 0
        dropout[n] = 0
    
    # last layer
    model.add(Dense(n_mgc, kernel_initializer='normal', activation='linear'))

    # compile model
    model.compile(loss='mean_squared_error', optimizer=optim)

    print(model.summary())

    # early stopping to avoid over-training
    # csv logger
    import datetime
    current_date = '{date:%Y-%m-%d_%H-%M-%S}'.format( date=datetime.datetime.now() )
    print(current_date)
    callbacks = [EarlyStopping(monitor='val_loss', patience=3, verbose=0), \
                 CSVLogger('ULT_to_MGC_' + current_date + '.csv', append=True, separator=';')]

    # Run training
    result = model.fit(x_train, y_train,
                            epochs = 100, batch_size = n_batch, shuffle = True, verbose = 2,
                            validation_data=(x_test, y_test),
                            callbacks=callbacks)
    
        # get the highest validation accuracy of the training epochs
    validation_loss = np.amin(result.history['val_loss']) 
    print('Best validation loss of epoch:', validation_loss)
    return {'loss': validation_loss, 'status': STATUS_OK, 'model': model}



# main hyperopt part
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform


best_run, best_model = optim.minimize(model=create_model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=5,
                                          trials=Trials())
print("Best performing model chosen hyper-parameters:")
print(best_run)
print("Evalutation of best performing model:")

### tesztelés
def vocoder_decode(mgc_lsp_coeff, lf0, basefilename_out, Fs = 22050, frlen = 512, frshft = 270, order = 24, alpha = 0.42, stage = 3):
    
    # requirement: SPTK package
    #
    # write files for SPTK
    mgc_lsp_coeff.astype('float32').tofile(basefilename_out + '.mgclsp')
    lf0.astype('float32').tofile(basefilename_out + '.lf0')
    
    # MGC-LSPs -> MGC coefficients
    command = 'sptk lspcheck -m ' + str(order) + ' -s ' + str(Fs / 1000) + ' -c -r 0.1 -g -G 1.0E-10 ' + basefilename_out + '.mgclsp' + ' | ' + \
              'sptk lsp2lpc -m '  + str(order) + ' -s ' + str(Fs / 1000) + ' | ' + \
              'sptk mgc2mgc -m '  + str(order) + ' -a ' + str(alpha) + ' -c ' + str(stage) + ' -n -u ' + \
                      '-M '  + str(order) + ' -A ' + str(alpha) + ' -C ' + str(stage) + ' > ' + basefilename_out + '.mgc'
    #print(command)
    run(command, shell=True)
    
    # MGLSADF synthesis based on pitch and MGC coefficients
    command = 'sptk sopr -magic -1.0E+10 -EXP -INV -m ' + str(Fs) + ' -MAGIC 0.0 ' + basefilename_out + '.lf0' + ' | ' + \
              'sptk excite -n -p ' + str(frshft) + ' | ' + \
              'sptk mglsadf -P 5 -m ' + str(order) + ' -p ' + str(frshft) + ' -a ' + str(alpha) + ' -c ' + str(stage) + ' ' + basefilename_out + '.mgc' + ' | ' + \
              'sptk x2x +fs -o | sox -c 1 -b 16 -e signed-integer -t raw -r ' + str(Fs) + ' - -t wav -r ' + str(Fs) + ' ' + basefilename_out + '.wav'
    #print(command)
    run(command, shell=True)

# globális paraméterek
order = 24
n_lines = 64
n_pixels_reduced = 64
n_mgc = order + 1

ult_test = np.fromfile('spkr048_test1_ult.dat', dtype='uint8')
mgc_test = np.fromfile('spkr048_test1_mgc.dat', dtype='float32')
lf0_test = np.fromfile('spkr048_test1_lf0.dat', dtype='float32')

ult_test = np.reshape(ult_test, (-1, n_lines, n_pixels_reduced, 1))
mgc_test = mgc_test.reshape(-1, order + 1)

# bemenet: [0-1 között]
ult_test = ult_test / 255.0

# betanított hálózat meghívása
mgc_predicted = best_model.predict(ult_test)


# kimenet skálázó visszatöltése
with open('mgc_scalers.sav', 'rb') as scaler_file:
    mgc_scalers = pickle.load(scaler_file)

# kimenet: nulla várható érték, egységnyi szórás
# inverz transzformáció
for i in range(n_mgc):
    mgc_predicted[:, i] = mgc_scalers[i].inverse_transform(mgc_predicted[:, i].reshape(-1, 1)).ravel()

# ULT-to-MGC, 1. jellemző kirajzolása
plt.subplot(211)
plt.plot(mgc_test[:, 0])
plt.title('original MGC_0')
plt.subplot(212)
plt.plot(mgc_predicted[:, 0])
plt.title('predicted MGC_0')

# ULT-to-speech
vocoder_decode(mgc_test, lf0_test, 'spkr048_1_original')
vocoder_decode(mgc_predicted, lf0_test, 'spkr048_1_dnn_hyperas')