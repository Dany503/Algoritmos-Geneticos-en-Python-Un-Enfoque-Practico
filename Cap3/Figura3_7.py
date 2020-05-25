import array
import random
import numpy as np
import matplotlib.pyplot as plt    
from deap import base
from deap import creator
from deap import tools

# Creamos los objetos para definir el problema y el tipo de individuo
creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0)) 
creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMin)

toolbox = base.Toolbox()

BOUND_LOW, BOUND_UP = 0.0, 1.0 # límites de las variables
NDIM = 2 # número de variables

# Generación de individuos aleatorios
def crea_individuo(low, up, size):
    return [random.uniform(low, up) for i in range(size)]

# Generación de individuos y población inicial    
toolbox.register("attr_float", crea_individuo, BOUND_LOW, BOUND_UP, NDIM)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Registro de operadores genéticos y función objetivo
toolbox.register("mate", tools.cxBlend) # cruce

random.seed(42)

lista_alpha = [0, 0.4, 0.8, 1.2, 1.6]
lista_colores = ["y", "m","b", "g", "c", "k"]
lista_resultados = []

ind1_original = toolbox.individual()
ind2_original = toolbox.individual()
ind1_original[:] = array.array('d', [0.3, 0.3])
ind2_original[:] = array.array('d', [0.7, 0.7])
v1 = np.array(ind1_original)
v2 = np.array(ind2_original)

plt.plot(v1[0], v1[1], color = "r", marker = "*", label = "Ind. Original")
plt.plot(v2[0], v2[1], color = "r", marker = "*")

for alpha, c in zip(lista_alpha, lista_colores):
    lista_distancia = []
    for i in range(10):
        ind1 = toolbox.clone(ind1_original)
        ind2 = toolbox.clone(ind2_original)
        ind3, ind4 = toolbox.mate(ind1 = ind1, ind2= ind2, alpha = alpha)
        v3 = np.array(ind3)
        v4 = np.array(ind4)
        if i== 0: 
            plt.plot(v3[0], v3[1], color = c, marker = "+", label = "$alpha$ = {}".format(alpha))
            plt.plot(v4[0], v4[1], color = c, marker = "+")
        else:
            plt.plot(v3[0], v3[1], color = c, marker = "+")
            plt.plot(v4[0], v4[1], color = c, marker = "+")
plt.legend(fontsize="x-small")
plt.xlim([-1, 2])
plt.ylim([0, 1])
plt.xlabel("$x_{1}$")
plt.ylabel("$x_{2}$")
plt.savefig("Blend.eps", dpi = 300)
