# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 19:38:27 2020

@author: dguti
"""

import pandas as pd
import numpy as np

datos_fitness = pd.read_csv("FitnessTSP.txt", sep = ",", names = ["id", "c", "m", "fitness"])

datos_08 = datos_fitness[datos_fitness["c"] == 0.8]
datos_07 = datos_fitness[datos_fitness["c"] == 0.7]
datos_06 = datos_fitness[datos_fitness["c"] == 0.6]

print("máximo c=0.8, m=0.2:", max(datos_08["fitness"].values))
print("mínimo c=0.8, m=0.2:", min(datos_08["fitness"].values))
print("media c=0.8, m=0.2:", np.mean(datos_08["fitness"].values))
print("desviación c=0.8, m=0.2:", np.std(datos_08["fitness"].values))

print("máximo c=0.7, m=0.3:", max(datos_07["fitness"].values))
print("mínimo c=0.7, m=0.3:", min(datos_07["fitness"].values))
print("media c=0.7, m=0.3:", np.mean(datos_07["fitness"].values))
print("desviación c=0.7, m=0.3:", np.std(datos_07["fitness"].values))

print("máximo c=0.6, m=0.4:", max(datos_06["fitness"].values))
print("mínimo c=0.6, m=0.4:", min(datos_06["fitness"].values))
print("media c=0.6, m=0.4:", np.mean(datos_06["fitness"].values))
print("desviación c=0.6, m=0.4:", np.std(datos_06["fitness"].values))


