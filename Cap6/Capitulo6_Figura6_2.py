# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 13:10:01 2020

@author: atapia
"""
import matplotlib.pyplot as plt
import numpy as np

def dibujaRio(datos=np.loadtxt("PuntosRio.csv",
     delimiter = ",")):
    
    plt.style.use('libroGA.mplstyle')

    s = datos[:,0]
    z = datos[:,1]
    
    plt.plot(s,z,'b.')
    plt.grid (True)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('s (m)', fontsize = 15)
    plt.ylabel('z (m)', fontsize = 15)