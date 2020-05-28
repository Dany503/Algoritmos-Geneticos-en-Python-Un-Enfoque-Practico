import numpy as np
import matplotlib.pyplot as plt
plt.style.use('libroGA.mplstyle')

f_ind = open('individuos_sensores_multi.txt', 'r')
lines_ind = list(f_ind)
f_ind.close()
f_fitness = np.loadtxt("fitness_sensores_multi.txt", delimiter = ",")
cobertura_1 = f_fitness[:,1]
cobertura_2 = f_fitness[:,2]

plt.scatter(cobertura_1, cobertura_2)
plt.xlim([min(cobertura_1)-3, max(cobertura_1)+3])
plt.xlabel('Cobertura única')
plt.ylim([min(cobertura_2)-10, max(cobertura_2)+10])
plt.ylabel('Redundancia')

fig, ax = plt.subplots(figsize=(8,6))

pdi = ax.scatter(x, y)
ax.set_xlim([0,2000])
ax.set_ylabel('Coordenada y')
ax.set_xlabel('Coordenada x')
ax.set_ylim([0,2000])

aux, ind = eval(lines_ind[0].split('\n')[0])
alcance = 100
x_pdc = ind[0::2]
y_pdc = ind[1::2]
for punto in zip(x_pdc, y_pdc):
    cir = plt.Circle((punto[0],punto[1]), alcance, color = 'r', alpha = 0.5)
    ax.add_artist(cir)
ax.legend([pdi, cir],['Puntos de interes', 'Punto de conexion'], loc = 2)

fig, ax = plt.subplots(figsize=(8,6))

pdi = ax.scatter(x, y)
ax.set_xlim([0,2000])
ax.set_ylabel('Coordenada y')
ax.set_xlabel('Coordenada x')
ax.set_ylim([0,2000])

aux, ind = eval(lines_ind[-1].split('\n')[0])
alcance = 100
x_pdc = ind[0::2]
y_pdc = ind[1::2]
for punto in zip(x_pdc, y_pdc):
    cir = plt.Circle((punto[0],punto[1]), alcance, color = 'r', alpha = 0.5)
    ax.add_artist(cir)
ax.legend([pdi, cir],['Puntos de interes', 'Punto de conexion'], loc = 2)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    