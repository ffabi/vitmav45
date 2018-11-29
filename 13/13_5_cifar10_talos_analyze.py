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
import matplotlib.pyplot as plt

# ez a log kb 600 tanítás eredményét tartalmazza
r = talos.Reporting('112918010504_.csv')

# eredmények megnézése
r.data.head(5)

# hány tanítás volt
r.rounds()

# mi volt a legjobb eredmény  (default: 'val_acc')
r.high()

# más metrikára legjobb eredmény
r.high('acc')

r.low('val_loss')

# hányadik tanításnál volt a legjobb eredmény
r.rounds2high()

# mik voltak a legjobb paraméterek
r.best_params()

# mi a paraméterek korrelációja egy adott metrikával
r.correlate('val_loss')

r.correlate('val_acc')

### ábrák
# 2D regresszió (val_acc vs val_loss)
r.plot_regs()
plt.show()

# acc a tanítás számának függvényében
r.plot_line()
plt.show()

# val_acc eloszlása
r.plot_kde('val_acc')
plt.show()

# val_loss eloszlása
r.plot_kde('val_loss')
plt.show()

# lehet mást is nézni vele
#  pl első rétegben milyen arányban volt 128 / 256 / 512 neuron
# (az összes tanítást összesítve)
r.plot_kde('first_neuron')
plt.show()

# hisztogram - ezt ábrázolja: plot_kde('val_acc')
r.plot_hist(bins=50)
plt.show()

# korreláció heatmap
# hasznos összesítés: látszik, hogy melyik paraméternek
# milyen a korrelációja a val_acc-cal
r.plot_corr()
plt.show()

# oszlopdiagram, 4D
r.plot_bars('batch_size', 'val_acc', 'first_neuron', 'dropout')
plt.show()