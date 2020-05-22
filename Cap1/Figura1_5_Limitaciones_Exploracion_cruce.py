# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 13:05:17 2020

@author: dguti
"""

import matplotlib.pyplot as plt

xmax = 50
xmin = -30
ymax = 50
ymin = -70

plt.scatter([xmin, xmax], [ymin, ymax], marker = "*", color = "b", label ="Valores máximos de las variables")
plt.plot([xmin, xmin, xmax, xmax, xmin], [ymin, ymax, ymax, ymin, ymin], color = "r", linestyle = "--", label = "Límite del área de posibles soluciones")
plt.xlabel("X")
plt.ylabel("Y")
plt.xlim([-100, 100])
plt.ylim([-100, 100])
plt.annotate(s="xmin, ymin", xy =(xmin, ymin-15))
plt.annotate(s="xmax, ymax", xy =(xmax, ymax+5))
plt.grid("True")
plt.legend(loc= "upper left")
plt.savefig("Exploracion_cruce.eps", dpi = 300)