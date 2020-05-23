import random
import json
import numpy
import matplotlib.pyplot as plt
from deap import algorithms
from deap import base
from deap import creator
from deap import tools

# gr*.json contiene el mapa de distancias entre ciudades en formato JSON 
with open("gr17.json", "r") as tsp_data:
    tsp = json.load(tsp_data)

distance_map = tsp["DistanceMatrix"] 
IND_SIZE = tsp["TourSize"] 

# Creamos los objetos para definir el problema y el tipo de individuo
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
# Generación de un tour aleatorio
toolbox.register("indices", random.sample, range(IND_SIZE), IND_SIZE)

# Generación de inviduos y población
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
toolbox.register("population", tools.initRepeat, list, toolbox.individual, 100)

def evalTSP(individual):
    """ Función objetivo, calcula la distancia que recorre el viajante"""
    # distancia entre el último elemento y el primero
    distance = distance_map[individual[-1]][individual[0]]
    # distancia entre el resto de ciudades
    for gene1, gene2 in zip(individual[0:-1], individual[1:]):
        distance += distance_map[gene1][gene2]
    return distance,

# registro de operaciones genéticas
toolbox.register("mate", tools.cxOrdered) 
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05) 
toolbox.register("select", tools.selTournament, tournsize=3) 
toolbox.register("evaluate", evalTSP) 

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
    ax1.set_ylim([2000, 6000])
    plt.grid(True)
    plt.savefig("Evolution.eps", dpi= 300)

def main(c, m):
    CXPB, MUTPB, NGEN = c, m, 120 
    pop = toolbox.population()
    MU, LAMBDA = len(pop), len(pop)
    hof = tools.HallOfFame(1) 
    stats = tools.Statistics(lambda ind: ind.fitness.values) 
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)
    logbook = tools.Logbook() 
    
    pop, logbook = algorithms.eaMuPlusLambda(pop, toolbox, MU, 
                                             LAMBDA, CXPB, MUTPB, 
                                             NGEN, stats=stats, 
                                             halloffame=hof, verbose= False)
    
    return hof, logbook
    
if __name__ == "__main__":
    random.seed(100) 
    prob_cruce = [0.8, 0.7, 0.6]
    prob_mutacion = [0.2, 0.3, 0.4]
    fichero_fitness = open("FitnessTSP.txt", "w")
    fichero_individuos = open("IndividuosTSP.txt", "w")
    for c, m in zip(prob_cruce, prob_mutacion):
        for i in range(10):
            best, log = main(c, m)
            fichero_fitness.write(str(i)+","+str(c)+","+str(m)+",")
            fichero_fitness.write(str(best[0].fitness.values[0]))
            fichero_fitness.write("\n")
            fichero_individuos.write(str(i)+","+str(c)+","+str(m)+",")
            fichero_individuos.write(str(best[0]))
            fichero_individuos.write("\n")
    fichero_fitness.close()
    fichero_individuos.close()