from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u

np.random.seed(1234)

def biseccion(x, y):
    return (x * y - 1 + np.sqrt((1 + x**2) * (1 + y**2))) / (x + y)

def suma_producto_punto(x,y,a,b):
    '''
    Calcula la suma del producto punto entre dos vectores
    '''
    P=0
    for i in range(a,b):
        P = P + x[i] * y[i]
    return P


def Bootstrap(datos ,mean_values,N):
    '''
    Genera una simulacion de bootstrap
    '''
    Nboot = N**3
    for i in range(Nboot):
        a = np.random.randint(low = 0, high = N , size = N)
        mean_values[i] = np.mean(datos[a])



# se importa la velocidad de recesion y la distancia a las variables v_rec y d
v_rec = np.loadtxt('data/SNIa.dat', usecols = [1]) #* (3.0857 * 10**22)
d = np.loadtxt('data/SNIa.dat', usecols = [2]) #* 10**3


vv=suma_producto_punto(v_rec, v_rec, 0, len(v_rec))
dd=suma_producto_punto(d, d, 0, len(v_rec))
vd=suma_producto_punto(v_rec, d, 0, len(v_rec))

# Para y=b*x => v=h0*d
h0_1 = vd / dd

# Para y=b*x => d=1/h0*v
h0_2 = vv / vd

h0 = biseccion(h0_1, h0_2)

dx = np.linspace(np.min(d), np.max(d), 1000)
y1 = h0_1 * dx
y2 = h0_2 * dx
y0 = h0 * dx

N=len(v_rec)
mean_values = np.zeros(N**3)
Bootstrap(v_rec / d, mean_values, N)
mean_values = np.sort(mean_values)
limite_bajo = mean_values[int(N**3 * 0.025)]
limite_alto = mean_values[int(N**3 * 0.975)]

fig1=plt.figure(1)
plt.hist(mean_values, bins=100)
plt.axvline(h0, color='r', label="h0")
plt.axvline(limite_bajo, color='b')
plt.axvline(limite_alto, color='b')
plt.legend(fontsize=11)
plt.savefig("SNIa_histograma.eps")
plt.show()
plt.draw()
print "El intervalo de confianza al 95% es: [{}:{}]".format(limite_bajo, limite_alto)
print "H0={}".format(h0)

fig2 = plt.figure(2)
axes = plt.gca()
plt.plot(d, v_rec, 'o')
plt.plot(dx,y1,'r', label = '$H_01$')
plt.plot(dx,y2,'g', label = '$H_02$')
plt.plot(dx,y0,'b', label = '$H_0$')
plt.xlabel('Distancia [Mpc]')
plt.ylabel('Velocidad de recesion [km/s]')
plt.legend(fontsize=11)
plt.savefig("SNIa.eps")
plt.show()
plt.draw()
