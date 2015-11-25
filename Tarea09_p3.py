from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u

np.random.seed(1234)

def Bootstrap(datos ,mean_values,N):
    '''
    Genera una simulacion de bootstrap
    '''
    Nboot = N*(np.log(N)**2)
    for i in range(Nboot):
        a = np.random.randint(low = 0, high = N , size = N)
        mean_values[i] = np.mean(datos[a])



# se importa la velocidad de recesion y la distancia a las variables v_rec y d
banda_i = np.loadtxt('data/DR9Q.dat', usecols = [80]) * 3.631
err_banda_i = np.loadtxt('data/DR9Q.dat', usecols = [81]) * 3.631
banda_z = np.loadtxt('data/DR9Q.dat', usecols = [82]) * 3.631
err_banda_z = np.loadtxt('data/DR9Q.dat', usecols = [83]) * 3.631

p = np.polyfit(banda_i,banda_z,1)
y = np.zeros(len(banda_i))
for i in range(len(banda_i)):
    y[i]= p[0] * banda_i[i] + p[1]


fig1 = plt.figure(1)
#axes = plt.gca()
plt.plot(banda_i, banda_z, 'ro')
plt.plot(banda_i, y, 'b', label = '$H_0$')
#plt.xlabel('Distancia [Mpc]')
#plt.ylabel('Velocidad de recesion [km/s]')
#plt.legend(fontsize=11)
plt.savefig("DR9Q.eps")
plt.show()
plt.draw()
