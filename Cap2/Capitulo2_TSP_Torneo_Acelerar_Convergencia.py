import random
import json
import numpy
import matplotlib.pyplot as plt
from deap import algorithms
from deap import base
from deap import creator
from deap import tools

plt.style.use("libroGA.mplstyle")

# gr*.json contiene el mapa de distancias entre ciudades en formato JSON 
with open("gr17.json", "r") as tsp_data:
    tsp = json.load(tsp_data)

# matriz de distancia
distance_map = tsp["DistanceMatrix"]
# número de ciudades que visitar 
IND_SIZE = tsp["TourSize"] 

# Creamos los objetos para definir el problema y el tipo de individuo
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
# Generación de un tour aleatorio
toolbox.register("indices", random.sample, range(IND_SIZE), IND_SIZE)

# Generación de inviduos y población
toolbox.register("individual", tools.initIterate, creator.Individual, 
                 toolbox.indices)
toolbox.register("population", tools.initRepeat, list, 
                 toolbox.individual, 100)

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
toolbox.register("evaluate", evalTSP)                           

def configura_torneo(k):
    toolbox.register("select", tools.selTournament, tournsize=k)    

fig, ax1 = plt.subplots()

def plot_evolucion(log, k):
    gen = log.select("gen")
    fit_mins = log.select("min")
    ax1.plot(gen, fit_mins, label="k="+str(k))
    ax1.set_xlabel("Generación")
    ax1.set_ylabel("Fitness mejor individuo")
    ax1.legend()
    ax1.set_ylim([2000, 4500])
    plt.grid(True)
    plt.savefig("EvolucionTSP_selection.eps", dpi=300)

def main():
    random.seed(100) 
    CXPB, MUTPB, NGEN = 0.7, 0.3, 120
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
                                             halloffame=hof)
    return hof, logbook
    
if __name__ == "__main__":
    for k in [3, 5, 10]:
        configura_torneo(k)
        best, log = main()
        plot_evolucion(log, k) 
    
    