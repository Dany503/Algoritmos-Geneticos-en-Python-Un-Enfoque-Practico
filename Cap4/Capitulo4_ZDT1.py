import array
import random
import json
import numpy as np
import matplotlib.pyplot as plt    
from deap import base
from deap import benchmarks
from deap import creator
from deap import tools
from deap import algorithms

BOUND_LOW, BOUND_UP = 0.0, 1.0 
NDIM = 10 

# Creamos los objetos para definir el problema y el tipo de individuo
creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0)) 
creator.create("Individual", array.array, typecode='d', 
               fitness=creator.FitnessMin)

# Generación de individuos aleatorios
def crea_individuo(low, up, size):
    return [random.uniform(low, up) for i in range(size)]

toolbox = base.Toolbox()

# Generación de individuos y población inicial    
toolbox.register("attr_float", crea_individuo, BOUND_LOW, 
                 BOUND_UP, NDIM)
toolbox.register("individual", tools.initIterate, creator.Individual, 
                 toolbox.attr_float)
toolbox.register("population", tools.initRepeat, list, 
                 toolbox.individual)

# Registro de operadores genéticos y función objetivo
toolbox.register("mate", tools.cxSimulatedBinaryBounded, 
                 low=BOUND_LOW, up=BOUND_UP, eta=20.0) 
toolbox.register("mutate", tools.mutPolynomialBounded, 
                 low=BOUND_LOW, up=BOUND_UP, eta=20.0, 
                 indpb=1.0/NDIM) 
toolbox.register("select", tools.selNSGA2) 
toolbox.register("evaluate", benchmarks.zdt1) 

def plot_frente():
    """
    Representación del frente de Pareto que hemos obtenido
    """
    datos_pareto = np.loadtxt("fitnessmulti.txt", delimiter=",")    
    plt.scatter(datos_pareto[:, 0], datos_pareto[:, 1], s=30)    
    
    # obtenermos el Pareto óptimo
    with open("zdt1_front.json") as optimal_front_data:
        pareto_optimo = np.array(json.load(optimal_front_data))
    plt.scatter(pareto_optimo[:, 0], pareto_optimo[:, 1], 
                s=10, alpha=0.4)
    plt.xlabel("FZDT11")
    plt.ylabel("FZDT12")
    plt.grid(True)
    plt.legend(["Pareto obtenido","Pareto óptimo"], loc="upper right")
    plt.savefig("ParetoBenchmark.eps", dpi=300, bbox_inches="tight")    
    
def main():
    CXPB , MUTPB , NGEN = 0.7, 0.3, 200
    MU, LAMBDA = 100, 100
    pop = toolbox.population(MU)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    logbook = tools.Logbook()  
    pareto = tools.ParetoFront() 
    pop, logbook = algorithms.eaMuPlusLambda(pop , toolbox , mu=MU,
                                             lambda_=LAMBDA, cxpb=CXPB,
                                             mutpb=MUTPB, ngen=NGEN, 
                                             stats=stats, halloffame=pareto,
                                             verbose=False)

    return pop, logbook, pareto
        
if __name__ == "__main__":
    random.seed(42)    
    pop, log, pareto = main()
    res_individuos = open("individuosmulti.txt", "w")
    res_fitness = open("fitnessmulti.txt", "w")
    for ind in pareto:
        res_individuos.write(str(ind))
        res_individuos.write("\n")
        res_fitness.write(str(ind.fitness.values[0]))
        res_fitness.write(",")
        res_fitness.write(str(ind.fitness.values[1]))
        res_fitness.write("\n")
    res_fitness.close()
    res_individuos.close()
plot_frente()
