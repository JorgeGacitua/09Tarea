from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u



# se importa la velocidad de recesion y la distancia a las variables v_rec y d
v_rec= np.loadtxt('data/hubble_original.dat', usecols = [0])*u.km/u.s
d=np.loadtxt('data/hubble_original.dat', usecols = [1])*u.pc

# se comvierten a unidades mks
v_rec_mks=(v_rec*u.s).to(u.m)/u.s
d_mks=d.to(u.m)*10**6

fig4 = plt.figure(1)
axes = plt.gca()
plt.plot(d.value,v_rec.value,'o')
plt.xlabel('Distancia [%s]' % str(d.unit))
plt.ylabel('Velocidad de recesion [%s]' % str(v_rec.unit))
plt.savefig("hubble_original.eps")
plt.show()
plt.draw()
