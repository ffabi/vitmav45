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
'''

import os
import os.path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D

hyperparams = dict()
max_evals = 1000
hyperparams['n_layer1'] = np.empty(max_evals)
hyperparams['n_layer2'] = np.empty(max_evals)
hyperparams['dropout_1'] = np.empty(max_evals)
hyperparams['dropout_2'] = np.empty(max_evals)
hyperparams['optim'] = list()
hyperparams['n_batch'] = np.empty(max_evals)

val_acc = np.empty(max_evals)
val_loss = np.empty(max_evals)

eval = 0
for file in sorted(os.listdir('.')):
    if "13_hyperas" in file and ".csv" in file: # and eval < 10:
        # csv_name = '13_hyperas_cifar10_' + current_date + '_' + \
        # str(n_layer1) + '_' +  str(n_layer2) + '_' + \
        # str(dropout_1) + '_' + str(dropout_2) + '_' + \
        # str(optim) + '_' + str(n_batch) + '.csv'
        #
        # 13_hyperas_cifar10_2018-11-26_22-26-13_128_256_0.6108763092812357_0.7371698374615214_sgd_256.csv
        hyperparams0 = file.replace('.csv', '').split('_')
        hyperparams['n_layer1'][eval] = int(hyperparams0[5])
        hyperparams['n_layer2'][eval] = int(hyperparams0[6])
        hyperparams['dropout_1'][eval] = np.float32(hyperparams0[7])
        hyperparams['dropout_2'][eval] = np.float32(hyperparams0[8])
        hyperparams['optim'] += hyperparams0[9]
        hyperparams['n_batch'][eval] = int(hyperparams0[10])
        
        # epoch;acc;loss;val_acc;val_loss
        with open(file, 'r') as csv_file:
            lines = csv_file.readlines()
        # last epoch: last line
        results = lines[-1].split(';')
        val_acc[eval] = np.float32(results[3])
        val_loss[eval] = np.float32(results[4])
        
        eval += 1

hyperparams['n_layer1'] = hyperparams['n_layer1'][0:eval]
hyperparams['n_layer2'] = hyperparams['n_layer2'][0:eval]
hyperparams['dropout_1'] = hyperparams['dropout_1'][0:eval]
hyperparams['dropout_2'] = hyperparams['dropout_2'][0:eval]
hyperparams['n_batch'] = hyperparams['n_batch'][0:eval]
val_acc = val_acc[0:eval]
val_loss = val_loss[0:eval]

best_index = np.argmax(val_acc)
print('Best performing model: ', hyperparams['n_layer1'][best_index], hyperparams['n_layer2'][best_index], hyperparams['dropout_1'][best_index], hyperparams['dropout_2'][best_index], hyperparams['n_batch'][best_index])

# vizualizáció:
# hogyan függenek össze az egyes hiperparaméterek a val_acc-cal?

fig = plt.figure(figsize=(10,7))
plt.plot(hyperparams['n_layer1'], val_acc, 'x')
plt.xlabel('n_layer1')
plt.ylabel('val_acc')
plt.show()

fig = plt.figure(figsize=(10,7))
plt.plot(hyperparams['n_layer2'], val_acc, 'x')
plt.xlabel('n_layer2')
plt.ylabel('val_acc')
plt.show()

fig = plt.figure(figsize=(10,7))
plt.plot(hyperparams['dropout_1'], val_acc, 'x')
plt.xlabel('dropout_1')
plt.ylabel('val_acc')
plt.show()

fig = plt.figure(figsize=(10,7))
plt.plot(hyperparams['dropout_2'], val_acc, 'x')
plt.xlabel('dropout_2')
plt.ylabel('val_acc')
plt.show()

fig = plt.figure(figsize=(10,7))
plt.plot(hyperparams['n_batch'], val_acc, 'x')
plt.xlabel('n_batch')
plt.ylabel('val_acc')
plt.show()

# vizualizáció:
# hogyan függ össze hiperparaméterek kombinációja a val_acc-cal?

fig = plt.figure(figsize=(15,10))
ax = fig.add_subplot(111, projection='3d')
p = ax.scatter3D(hyperparams['n_layer1'], hyperparams['n_layer2'], val_acc, c=val_acc, cmap='Greens')
ax.set_xlabel('n_layer1')
ax.set_ylabel('n_layer2')
ax.set_zlabel('val_acc')
fig.colorbar(p)
plt.show()

fig = plt.figure(figsize=(15,10))
ax = fig.add_subplot(111, projection='3d')
p = ax.scatter3D(hyperparams['dropout_1'], hyperparams['dropout_2'], val_acc, c=val_acc, cmap='Greens')
ax.set_xlabel('dropout_1')
ax.set_ylabel('dropout_2')
ax.set_zlabel('val_acc')
fig.colorbar(p)
plt.show()


fig = plt.figure(figsize=(15,10))
ax = fig.add_subplot(111, projection='3d')
p = ax.scatter3D(hyperparams['n_layer1'], hyperparams['dropout_1'], val_acc, c=val_acc, cmap='Greens')
ax.set_xlabel('n_layer1')
ax.set_ylabel('dropout_1')
ax.set_zlabel('val_acc')
fig.colorbar(p)
plt.show()

fig = plt.figure(figsize=(15,10))
ax = fig.add_subplot(111, projection='3d')
p = ax.scatter3D(hyperparams['n_batch'], hyperparams['dropout_1'], val_acc, c=val_acc, cmap='Greens')
ax.set_xlabel('n_batch')
ax.set_ylabel('dropout_1')
ax.set_zlabel('val_acc')
fig.colorbar(p)
plt.show()
