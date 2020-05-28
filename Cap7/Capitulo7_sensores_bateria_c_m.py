import random
import time

from deap import base
from deap import creator
from deap import tools
from deap import algorithms
import numpy as np

def area(punto):
    if punto[0] > 2000:
        return False
    if punto[1] > 2000:
        return False
    return True

def cobertura(interes, conexion):
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
    for index, gen in enumerate(individuo):
        if random.random() < indpb: 
            individuo[index] = random.gauss(gen, 50)
            if individuo[index] < 0:
                individuo[index] = 0
            if individuo[index] > 2000:
                individuo[index] = 2000
    return individuo,

def fitness(individuo):
    x_pdc = individuo[0::2]
    y_pdc = individuo[1::2]
    pdi_vector = [0]*75
    for pdc in zip(x_pdc, y_pdc):
        if area(pdc) == False: 
            return penaliza,
        for index, pdi in enumerate(zip(x, y)):
            if pdi_vector[index] == 0: 
                if cobertura(pdi, pdc):
                    pdi_vector[index] = 1
    return sum(pdi_vector), 

def unico_objetivo_ga(c, m, toolbox):
    NGEN = 700
    MU = 300 
    LAMBDA = MU 
    CXPB = c
    MUTPB = m
   
    pop = toolbox.ini_poblacion(n = MU)
    hof = tools.HallOfFame(1, similar = np.array_equal)
 
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

random.seed(1)

x = [random.uniform(0,2000) for _ in range(75)]
y = [random.uniform(0,2000) for _ in range(75)]

numero = 50
penaliza = -9e10
alcance = 100

creator.create("FitnessMin", base.Fitness, weights = (1.0,))
creator.create("Individual", list, fitness = creator.FitnessMin)
 
toolbox = base.Toolbox() 
toolbox.register("individual", tools.initIterate, 
                    creator.Individual, crea_individuo)
toolbox.register("ini_poblacion", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", fitness)
toolbox.register("mate", tools.cxBlend, alpha = 0.5)
toolbox.register("mutate", mutacion, indpb = 0.05)
toolbox.register("select", tools.selTournament, tournsize = 3)


# Probabilidades que queremos probar
parameters = [(0.6, 0.4), (0.7, 0.3), (0.8, 0.2)] 
# Para cada combinación de c y m
for c, m in parameters:

    # Lanzamos 10 veces el algoritmo
    for i in range(0, 10): 
        print(str(c) + ', ' + str(m) + ', ' +str(i))
        # Abrimos dos archivos de texto para almacenar los resultados
        res_individuos = open("individuos_sensores.txt", "a")
        res_fitness = open("individuos_sensores.txt", "a")
        
        # Hacemos la llamada al algoritmo
        t0 = time.time()
        pop_new, pareto_new, log = unico_objetivo_ga(c, m, toolbox)
        t1 = time.time()
        print(t1-t0)
        
        # Almacenamos la solución en los ficheros de texto
        for ide, ind in enumerate(pareto_new):
            res_individuos.write(str(i))
            res_individuos.write(",")
            res_individuos.write(str(c))
            res_individuos.write(",")
            res_individuos.write(str(m))
            res_individuos.write(",")
            res_individuos.write(str(ind))
            res_individuos.write("\n")
            res_fitness.write(str(i))
            res_fitness.write(",")
            res_fitness.write(str(c))
            res_fitness.write(",")
            res_fitness.write(str(m))
            res_fitness.write(",")
            res_fitness.write(str(ind.fitness.values[0]))
            res_fitness.write("\n")
            
        # Borramos la solución y cerramos los archivos
        del(pop_new)
        del(pareto_new)
        res_fitness.close()
        res_individuos.close()
