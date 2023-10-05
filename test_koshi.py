#solution of equation:
#    d5y/dx5 + 15d4y/dx4 + 90d3y/dx3 + 270d2y/dx2 + 405dy/dx + 243y = 0
#         x in [0,5],     y(0) = 0, dy/dx(0) = 3,  d2y/dx2(0) = -9, dy3/dx3(0) = -8, dy4/dx4(0) = 0
# with runge-kutty method of 4th order

from math import *
import numpy as np
import matplotlib.pyplot as plt


#Initial Values
x = 0
xf = 5     #interval x
h = 0.000005
m = (xf- x)/h
N = int(m)     #number of iterations
y = [0,3,-9,-8,0]
yarr = []
xarr = []
a = 1.0/6

#Defining the differential equations
def Y(y):
    func = -1*(243*y[0] + 405*y[1] + 270*y[2] + 90*y[3] + 15*y[4])
    return func

def update(y,k,h):
    f = np.zeros((5))
    f[0] = y[0]
    for i in range(4):
        f[i+1] = y[i+1] + h*k
    return f

def runge_kutta(y, h):
    k1_y = Y(y)
    y_new = update(y,0.5*k1_y,h)
    k2_y = Y(y_new)
    y_new = update(y_new,0.5*k2_y,h)
    k3_y = Y(y_new)
    y_new = update(y,k3_y,h)
    k4_y = Y(y_new)

    y4_next = y[4] + h*(k1_y+2*(k2_y+k3_y)+k4_y) 
    y3_next = y[3] + h*y4_next
    y2_next = y[2] + h*y3_next
    y1_next = y[1] + h*y2_next
    y0_next = y[0] + h*y1_next
    
    y[4] = y4_next
    y[3] = y3_next
    y[2] = y2_next
    y[1] = y1_next
    y[0] = y0_next

    return y

for i in range(0, N):
    y = runge_kutta(y, h)
    x = x + h
    yarr.append(y[0])
    xarr.append(x)

yarr = np.array(yarr)

# Plot the results
plt.plot(xarr,yarr)
plt.show()

k = Y(y)
print(yarr)
print(k)