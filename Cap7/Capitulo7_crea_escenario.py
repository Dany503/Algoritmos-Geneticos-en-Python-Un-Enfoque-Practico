# Importamos las librerí­as necesarias
import random
import numpy as np
import matplotlib.pyplot as plt

# Con este comando conseguimos que siempre que se ejecute el script 
# se generen los mimos núeros aleatorios
random.seed(1)

# Generamos 75 coordenadas "x" e "y" para cada uno de los puntos
x= [random.uniform(0,2000) for _ in range(75)]
y= [random.uniform(0,2000) for _ in range(75)]

# Representamos los puntos de interés graficamente
cir = plt.Circle((500,1000), 100, color='r', alpha = 0.5)
fig, ax = plt.subplots()
pdi = ax.scatter(x, y)
ax.set_xlabel('Coordenada x')
ax.set_xlim([0,2000])
ax.set_ylabel('Coordenada y')
ax.set_ylim([0,2000])
ax.add_artist(cir)
ax.grid(True)
ax.legend([pdi, cir],['Puntos de interes', 'Punto de conexion'], loc = 2)


