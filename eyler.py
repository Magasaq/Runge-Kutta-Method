from math import *
import numpy as np
import matplotlib.pyplot as plt

#Initial Values
x = 0.994
y = 0
x_deriv = 0
y_deriv = -2.00158510637908252240537862224
t = 0
tf= 17.0652165601579625588917206249
h= 0.00002
m=(tf-t)/h - (tf-t)%h
Noumber_of_it  = int(m)
a=1.0/6
mu = 0.012277471
xarr = np.array([])
yarr = np.array([])

#Definition of differential equations
def f0(x,y_deriv):
    f0 = x + 2*y_deriv - (1-mu)*(x + mu)/D1 - mu*(x-1+mu)/D2
    return f0
def f1(y,x_deriv):
    f1 = y - 2*x_deriv - (1-mu)*y/D1 - mu*y/D2
    return f1
for i in range(0,Noumber_of_it): 
    D1 = pow(((x + mu)*(x + mu) + y*y),(3/2))
    D2 = pow(((x + mu - 1)*(x + mu - 1) + y*y),(3/2))
    k1_x=f0(x,y_deriv)
    k1_y=f1(y,x_deriv)
    x_next_deriv = x_deriv+h*(k1_x)   
    y_next_deriv = y_deriv+h*(k1_y)  
    x_deriv = x_next_deriv
    y_deriv = y_next_deriv
    x_next = x + x_deriv*h
    y_next = y + y_deriv*h
    x = x_next
    y = y_next
    t=t+h
    xarr = np.append(xarr, [x])
    yarr = np.append(yarr, [y])

plt.plot(xarr,yarr)
plt.show()

print(t, x, y)