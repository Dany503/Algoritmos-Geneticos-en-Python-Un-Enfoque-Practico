import random
import numpy as np
from deap import base
from deap import creator
from deap import tools
import matplotlib.pyplot as plt
plt.style.use("libroGA.mplstyle")

# Creamos los objetos para definir el problema y el tipo de individuo
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Generación de genes 
toolbox.register("attr_uniform", random.uniform, -100, 100)

# Generación de individuos
toolbox.register("individual", tools.initRepeat, creator.Individual, 
                 toolbox.attr_uniform, 2)

# Registro de operaciones genéticas
toolbox.register("mutate", tools.mutGaussian, mu=0, 
                 sigma= 5, indpb=0.1)

random.seed(42)
ind = toolbox.individual()
ind[0] = 10
ind[1] = 10
v1 = np.array(ind)
plt.plot(v1[0], v1[1], marker="*", color="b", label="Individuo original", markersize=10)
plt.xlim(v1[0]-10, v1[0]+10)
plt.ylim(v1[1]-10, v1[0]+10)

for sigma, color in zip([1, 5, 10], ["r", "g", "y"]):
    for i in range(20):
        mutante, = toolbox.mutate(ind, sigma=sigma)
        v2 = np.array(mutante)
        if i == 0:
            plt.plot(v2[0], v2[1], color=color, marker="*", label="$\sigma$: " + str(sigma))
        else:
            plt.plot(v2[0], v2[1], color=color)
plt.legend(loc="lower left")
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.grid(True)
plt.savefig("Mutacion_Gausiana_cap1.eps", dpi=300)

        