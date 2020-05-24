import random
import numpy as np
import matplotlib.pyplot as plt    
from deap import base
from deap import creator
from deap import tools
from deap import algorithms

random.seed(42) 

# Límites de los valores del conjunto
LIMITE_INF, LIMITE_SUP = 0, 30
# Número de elementos del conjunto 
TAM_CONJUNTO = 30 
SUMA_OBJETIVO = 333
CONJUNTO = np.array(random.sample(range(LIMITE_INF, LIMITE_SUP), 
                                  TAM_CONJUNTO))

# Creamos los objetos para definir el problema y el tipo de individuo
creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0)) 
creator.create("Individual", list, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()

# Generación de individuos aleatorios
def crea_individuo(size):
    return [random.randint(0, 1) for i in range(size)]

# Generación de individuos y población inicial    
toolbox.register("attr", crea_individuo, TAM_CONJUNTO)
toolbox.register("individual", tools.initIterate, creator.Individual, 
                 toolbox.attr)
toolbox.register("population", tools.initRepeat, list, 
                 toolbox.individual)

# Función objetivo
def funcion_objetivo(individuo, suma_objetivo):
    """
    Función objetivo para el problema de la suma de subconjutos
    Entrada: Recibe como entrada el individuo y la suma objetivo 
    que se tiene que satisfacer con la suma de los elementos del subconjuto.
    Salida: Como objetivo 1 se devuelve el número de elementos del
    subconjunto. Como objetivo 2 se devuelve la diferencia con respecto
    a la suma objetivo.
    """
    subconjunto = CONJUNTO[np.array(individuo) == 1]
    suma_subconjunto = np.sum(subconjunto)
    diferencia = suma_objetivo - suma_subconjunto
    n_elementos = sum(individuo)
    if diferencia < 0: # nos pasamos
        return 10000, 10000 # pena de muerte    
    if n_elementos == 0: # no se selecciona ninguna elemento
        return 10000, 10000
    return n_elementos, diferencia

# Registro de operadores genéticos y función objetivo
toolbox.register("mate", tools.cxTwoPoint) 
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05) 
toolbox.register("select", tools.selNSGA2) 
toolbox.register("evaluate", funcion_objetivo, suma_objetivo=SUMA_OBJETIVO) 

def plot_frente():
    """
    Representación del frente de Pareto que hemos obtenido
    """
    datos_pareto = np.loadtxt("fitnessconjuntos.txt", delimiter=",")    
    plt.scatter(datos_pareto[:, 0], datos_pareto[:, 1], marker="+", 
                color="b", s=50)    
    plt.xlabel("Elementos")
    plt.ylabel("Diferencias")
    plt.grid(True)
    plt.xlim([0, 17])
    plt.ylim([-20, 300])
    plt.legend(["Frente de Pareto"], loc="upper right")
    plt.savefig("Pareto_conjunto.eps", dpi = 300)
    
def main():
    CXPB, MUTPB, NGEN = 0.7, 0.3, 200
    MU, LAMBDA = 300, 300
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
    pop, log, pareto = main()
    res_individuos = open("individuosconjuntos.txt", "w")
    res_fitness = open("fitnessconjuntos.txt", "w")
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