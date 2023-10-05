import numpy 
import math
from pylab import *
from sympy import *
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d, Axes3D
from numpy import linalg as LA



def func(a, z_str):
    exec ('z = lambda a: ' + z_str)
    z_str = z_str.replace('a[0]', 'x')
    z_str = z_str.replace('a[1]', 'y')

def repl(a, a_new, z_str):
    exec ('z = lambda a: ' + z_str)
    z_str = z_str.replace('a[0]', 'a_new[0]')
    z_str = z_str.replace('a[1]', 'a_new[1]')
    return z_str


def z_grad(a,z_str):
    func(a,z_str)
    x = Symbol('x')
    y = Symbol('y')

    z_d = eval(z_str) #exec ('z_d =  ' + z_str)

    yprime = z_d.diff(y)
    dif_y = str(yprime).replace('y', str(a[1]))
    dif_y  = dif_y.replace('x', str(a[0]))

    xprime = z_d.diff(x)
    dif_x=str(xprime).replace('x', str(a[0]))
    dif_x=dif_x.replace('y', str(a[1]))

    return numpy.array([eval(dif_y), eval(dif_x)])

def minimize(a,z_str):
    l_min = minimize_scalar(lambda l: z_str(a - l * z_grad(a,z_str))).l
    return a - l_min * z_grad(a,z_str)

def norm(a):
    return math.sqrt(a[0] ** 2 + a[1] ** 2)

def grad_step(dot, z_str):
    return minimize(dot, z_str)

