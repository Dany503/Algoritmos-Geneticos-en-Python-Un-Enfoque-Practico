# Importamos las librerías necesarias
import random

from deap import base
from deap import creator
from deap import tools
from deap import algorithms
import numpy as np




def area(punto):
    """
    Función que recibe un punto de conexión y evalúa si se encuentra 
    dentro del área de estudio
    """
    if punto[0] > 2000:
        return False
    if punto[1] > 2000:
        return False
    return True

def cobertura(interes, conexion):
    """
    Función que recibe un punto de interés y uno de conexión y evalúa la 
    distancia entre ambos. Si la distancia es menor a 250 metros devuelve True, 
    en caso contrario False.
    """
    distancia = np.sqrt((interes[0]-conexion[0])**2 
                      + (interes[1]-conexion[1])**2)
    if distancia <= alcance:
        return True
    else:
        return False

#def crea_individuo():
#    individuo = [0]*numero*2
#    for i in range(len(individuo)):
#        individuo[i] = np.random.uniform(0, 2000)
#    return individuo

def crea_individuo():
    individuo = [0]*numero*2
    for i in range(numero):
        p_pdi = random.randint(0, len(x) - 1)
        individuo[i] = x[p_pdi]
        individuo[i+numero] = y[p_pdi]
    return individuo


def mutacion(individuo, indpb):
    """
    Función que a partir de un individuo y una probabilidad de mutación 
    modifica los genes del mismo
    """
    #Para cada gen del individuo
    for index, gen in enumerate(individuo):
        if random.random() < indpb: 
            individuo[index] = random.gauss(gen, 50)
            if individuo[index] < 0:
                individuo[index] = 0
            if individuo[index] > 2000:
                individuo[index] = 2000
    return individuo,

def fitness(individuo):
    # Separamos los valores de x e y de los puntos de conexión
    x_pdc = individuo[0::2]
    y_pdc = individuo[1::2]
    # Vector para indicar qué puntos de interés están cubiertos
    pdi_vector = [0]*75
    pdi_vector_2 = [0]*75
    # Para cada punto de conexión (x_pdc,y_pdc)
    for pdc in zip(x_pdc, y_pdc):
        if area(pdc) == False: # Si están fuera del área, se descarta
            return penaliza,
        # Para cada punto de interés (x,y)
        for index, pdi in enumerate(zip(x, y)):
            if cobertura(pdi, pdc):
                pdi_vector[index] = 1
                pdi_vector_2[index] += 1
    return sum(pdi_vector), sum(pdi_vector_2) # Devolvemos el número de puntos cubiertos


def multiple_objetivo_ga(c, m, toolbox):
    NGEN = 2000 
    MU = 1000
    LAMBDA = MU 
    CXPB = c
    MUTPB = m
   
    pop = toolbox.ini_poblacion(n = MU)
    hof = tools.ParetoFront()
 
    stats = tools.Statistics(key = lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
   
    logbook = tools.Logbook()
   
    pop, logbook = algorithms.eaMuPlusLambda(pop, toolbox, 
                    MU, LAMBDA, CXPB, MUTPB, NGEN,
                    stats= stats, halloffame=hof, verbose = False)
   
    return pop, hof, logbook


# Con este comando conseguimos que siempre que se ejecute el script 
# se generen los mimos números aleatorios
random.seed(1)

# Generamos 75 coordenadas "x" e "y" para cada uno de los puntos
x = [random.uniform(0,2000) for _ in range(75)]
y = [random.uniform(0,2000) for _ in range(75)]

numero = 50
penaliza = -9e10
alcance = 100

creator.create("FitnessMin", base.Fitness, weights = (+1,+1))
creator.create("Individual", list, fitness = creator.FitnessMin)
 
toolbox = base.Toolbox() 
toolbox.register("individual", tools.initIterate, 
                    creator.Individual, crea_individuo)
toolbox.register("ini_poblacion", tools.initRepeat, 
                list, toolbox.individual)
toolbox.register("evaluate", fitness)
toolbox.register("mate", tools.cxBlend, alpha = 0.5)
toolbox.register("mutate", mutacion, indpb = 0.05)
toolbox.register("select", tools.selNSGA2)

pop, pareto , log = multiple_objetivo_ga(0.7, 0.3, toolbox)
res_individuos = open("individuos_sensores_multi.txt", "a")
res_fitness = open("fitness_sensores_multi.txt", "a")
for ide, ind in enumerate(pareto):
    res_individuos.write(str(ide))
    res_individuos.write(",")
    res_individuos.write(str(list(ind)))
    res_individuos.write("\n")
    res_fitness.write(str(ide))
    res_fitness.write(",")
    res_fitness.write(str(ind.fitness.values[0]))
    res_fitness.write(",")
    res_fitness.write(str(ind.fitness.values[1]))
    res_fitness.write("\n")
res_fitness.close()
res_individuos.close()