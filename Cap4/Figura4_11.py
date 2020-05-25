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

# Registro de operadores genéticos y función objetivo
toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, indpb=1/NDIM) # mutación

random.seed(42)

lista_eta = [1, 5, 10]
lista_colores = ["m","b", "g"]
lista_resultados = []

ind1_original = toolbox.individual()
ind1_original[:] = array.array('d', [0.5, 0.5])
v1 = np.array(ind1_original)

plt.plot(v1[0], v1[1], color = "r", marker = "*", label = "Ind. Original", markersize=15)

for eta, c in zip(lista_eta, lista_colores):
    for i in range(20): 
        ind1 = toolbox.clone(ind1_original)
        mut = toolbox.mutate(ind1, eta = eta)
        v2 = np.array(mut[0])
        if i== 0: 
            plt.plot(v2[0], v2[1], color=c, marker="+", label="$\eta$ = " +str(eta))
        else:
            plt.plot(v2[0], v2[1], color=c, marker="+")
        
plt.legend(fontsize="x-small")
plt.xlim([-0.5, 1.5])
plt.ylim([-0.5, 1.5])
plt.xlabel("$x_{1}$")
plt.ylabel("$x_{2}$")
plt.savefig("MUTPoly.eps", dpi=300)
