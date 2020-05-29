import array
import random
import numpy as np
import matplotlib.pyplot as plt

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import benchmarks

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", array.array, typecode = 'f', fitness=creator.FitnessMin)

toolbox = base.Toolbox()

minimo = 0
maximo = 500
tamaño = 40

def crea_individuo(a, b, n):
    individuo = [random.uniform(a, b) for _ in range(n)]
    return individuo

def mutTriangular(individuo, m, indpb):
    for i in range(len(individuo)):
        if random.random() <= indpb:
            individuo[i] = random.triangular(individuo[i]-m, individuo[i], individuo[i]+m)
    return individuo,

toolbox.register("attr", crea_individuo, a=minimo, b=maximo, n=tamaño)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", mutTriangular, m=5, indpb=0.08)
toolbox.register("select", tools.selTournament, tournsize = 4)
toolbox.register("evaluate", benchmarks.schwefel)

def main():
    random.seed(1) # modificar el valor de la semilla para probar varias 
    # posibilidades

    CXPB, MUTPB, INDIVIDUOS, NGEN = 0.7, 0.3, 1000, 200
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
    plt.grid(True)
    plt.savefig("Evolution_Schwefel.eps", dpi= 300)
    
if __name__ == "__main__":
    pop, log, hof = main()
    plot_evolucion(log)
    print(hof[0])
    print(hof[0].fitness.values) 
