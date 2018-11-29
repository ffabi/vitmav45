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
    [hyperas] https://github.com/maxpumperla/hyperas
    [UTI-to-MGC with DNN] Tamás Gábor Csapó, Tamás Grósz, Gábor Gosztolya, László Tóth, Alexandra Markó, ,,DNN-based Ultrasound-to-Speech Conversion for a Silent Speech Interface'', Interspeech 2017, Stockholm, Sweden, pp. 3672-3676, 2017., http://smartlab.tmit.bme.hu/downloads/pdf/csapot/Csapo-et-al-interspeech2017-paper.pdf
'''

# adatok letöltése
# !wget http://smartlab.tmit.bme.hu/csapo/dl/spkr048_ult.tgz
# !tar -xzvf spkr048_ult.tgz
#
# SPTK telepítése (MGC -> beszéd szintézishez)
# !apt install sptk


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

from subprocess import run

# globális paraméterek
order = 24
n_lines = 64
n_pixels_reduced = 64
n_mgc = order + 1


#### adatok betöltése

ult = np.fromfile('spkr048_train_valid_ult.dat', dtype='uint8')
mgc = np.fromfile('spkr048_train_valid_mgc.dat', dtype='float32')

# bemenet: ultrahang képek átméretezése
ult = np.reshape(ult, (-1, n_lines * n_pixels_reduced))
print('ult shape: ', ult.shape)

# egy példa ultrahang kép
plt.imshow(ult[0].reshape(n_lines, n_pixels_reduced))
plt.show()

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

x_train = ult_training
y_train = mgc_training
x_test = ult_validation
y_test = mgc_validation


### hálózat összerakása és tanítás
# hálózat összerakása
model = Sequential()
model.add(Dense(1000, input_dim=n_lines*n_pixels_reduced, activation='relu'))
model.add(Dense(1000, activation='relu'))
model.add(Dense(1000, activation='relu'))
model.add(Dense(1000, activation='relu'))
model.add(Dense(n_mgc, activation='linear'))

model.compile(loss='mean_squared_error', optimizer='adam')

print(model.summary())

# early stopping 
callbacks = [EarlyStopping(monitor='val_loss', patience=3, verbose=0)]

# tanítás
result = model.fit(x_train, y_train,
                        epochs = 100, batch_size = 128, shuffle = True, verbose = 2,
                        validation_data=(x_test, y_test),
                        callbacks=callbacks)

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

ult_test = np.fromfile('spkr048_test1_ult.dat', dtype='uint8')
mgc_test = np.fromfile('spkr048_test1_mgc.dat', dtype='float32')
lf0_test = np.fromfile('spkr048_test1_lf0.dat', dtype='float32')

ult_test = np.reshape(ult_test, (-1, n_lines * n_pixels_reduced))
mgc_test = mgc_test.reshape(-1, order + 1)

# bemenet: [0-1 között]
ult_test = ult_test / 255.0

# betanított hálózat meghívása
mgc_predicted = model.predict(ult_test)

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
vocoder_decode(mgc_predicted, lf0_test, 'spkr048_1_dnn')