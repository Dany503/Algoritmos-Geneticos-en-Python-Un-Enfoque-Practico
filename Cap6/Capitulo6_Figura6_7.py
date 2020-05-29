# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 13:25:37 2020

@author: atapia
"""
import matplotlib.pyplot as plt
import numpy as np

def dibujaSolucion(individuo, datos = np.loadtxt("PuntosRio.csv",
     delimiter = ",")):
    s = datos[:,0]
    z = datos[:,1]
        
    plt.style.use('libroGA.mplstyle')
    
    nodos   = np.nonzero(individuo)[0]
    s_nodos = s[nodos]
    z_nodos = z[nodos]
    
    plt.plot(s,z)
    plt.plot(s_nodos,z_nodos,'-ok')
    
    plt.grid (True)

    plt.xlabel('s (m)', fontsize = 15)
    plt.ylabel('z (m)', fontsize = 15)
    plt.legend({'Río','Trazado óptimo'})
    
    # Misma escala en ambos ejes
    plt.gca().set_aspect('equal', adjustable='box')
