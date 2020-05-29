import array
import random
import numpy as np
import matplotlib.pyplot as plt

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import benchmarks

creator.create("FitnessMin", base.Fitness, weights=(1.0,))
creator.create("Individual", array.array, typecode = 'f', fitness=creator.FitnessMin)

toolbox = base.Toolbox()

minimo = -100
maximo = 100
tamaño = 2
desvTipica = 1

def crea_individuo(a, b, n):
    individuo = [random.uniform(a, b) for _ in range(n)]
    return individuo

def mutacion_gaussiana(individuo, sigma, indpb):
    for i in range(len(individuo)):
        if random.random() <= indpb:
            individuo[i] = random.gauss(individuo[i], sigma)
    return individuo,

toolbox.register("attr", crea_individuo, a=minimo, b=maximo, n=size)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxSimulatedBinaryBounded, eta = 2, 
                 low = minimo, up = maximo) 
toolbox.register("mutate", mutacion_gaussiana, sigma=desvTipica, indpb=0.05)
toolbox.register("select", tools.selRoulette) # seleccion
toolbox.register("evaluate", benchmarks.h1) # evaluación

def main():
    random.seed(1) # modificar el valor de la semilla para probar varias 
    # posibilidades

    CXPB, MUTPB, INDIVIDUOS, NGEN = 0.7, 0.3, 100, 100
    pop = toolbox.population(INDIVIDUOS)

    hof = tools.HallOfFame(1) # guardamos el mejor individuo
    log = tools.Logbook()
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    pop, log = algorithms.eaSimple(pop,toolbox, CXPB, MUTPB, NGEN, halloffame=hof, stats=stats, verbose=True)    
    
    return pop, log, hof
    

if __name__ == "__main__":
    pop, log, hof = main()
    print(hof[0])
    
    print(hof[0].fitness.values) 

    
    
def plot_evolucion(log):
    gen = log.select("gen")
    fit_mins = log.select("min")
    fit_maxs = log.select("max")
    fit_ave = log.select("avg")

    fig, ax1 = plt.subplots()
    ax1.plot(gen, fit_mins, "b")
    ax1.plot(gen, fit_maxs, "r")
    ax1.plot(gen, fit_ave, "--k")
    ax1.fill_between(gen, fit_mins, fit_maxs, where=fit_maxs >= fit_mins, facecolor='g', alpha = 0.2)
    ax1.set_xlabel("Generación")
    ax1.set_ylabel("Fitness")
    ax1.legend(["Min", "Max", "Avg"])
    ax1.set_ylim([0, 2.2])
    plt.grid(True)
    plt.savefig("Evolution_h1.eps", dpi= 300)    