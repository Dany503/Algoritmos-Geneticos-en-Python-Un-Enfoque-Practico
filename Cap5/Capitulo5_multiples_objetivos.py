import numpy as np
from deap import base
from deap import creator
from deap import tools
from deap import algorithms
import matplotlib.pyplot as plt
import random

t = np.arange(0, 24)
P_PV = np.array([0, 0, 0, 0, 0, 0, 0, 0, 6, 10, 15, 20, 
                 30, 40, 40, 20, 15, 10, 8, 2, 0, 0, 0, 0])
P_WT = np.array([51, 51, 58, 51, 64, 51, 44, 51, 44, 51, 51, 46, 
                 81, 74, 65, 65, 65, 51, 39, 63, 38, 66, 74, 74])

P_DM = np.array([67, 67, 90, 114, 120, 130, 150, 190, 200, 206, 227, 227, 250, 250, 200, 180, 160, 160, 190, 150, 100, 50, 20, 20])
P_dem = P_DM - P_PV - P_WT
# Generador diesel
P_DE_min = 5
P_DE_max = 80
# Microturbina
P_MT_min = 10
P_MT_max = 140

P_ESS_min = -120
P_ESS_max = 120
SOC_ESS_max = 280
SOC_ESS_min = 70
SOC_ini = 140

penaliza = 99999999999999

def evalua_despachable(P_DE, P_MT):
    for p in P_DE:
        if p < P_DE_min and p != 0:
            return penaliza
    if any(P_DE) > P_DE_max:
        return penaliza
    for p in P_MT:
        if p < P_MT_min and p != 0:
            return penaliza
    if any(P_MT) > P_MT_max:
        return penaliza
    return 0

# Generador diesel
def coste_DE(P_DE):
    f_DE = 0.0012
    e_DE = 0.2455
    d_DE = 1.925
    if (P_DE == 0):
        return 0
    else:
        return d_DE + e_DE*P_DE + f_DE*(P_DE**2) 

# Microturbina
def coste_MT(P_MT):
    f_MT = 0.0002
    e_MT = 0.2015
    d_MT = 7.4344
    if (P_MT == 0):
        return 0
    else:
        return d_MT + e_MT*P_MT + f_MT*(P_MT**2) 

def evolucion_SOC(P_ESS, SOC_ini):
    Delta_t = 1
    SOC = np.zeros(24)
    SOC[0] = SOC_ini
    for i in range(1, SOC.size):
        SOC[i] = SOC[i-1] - P_ESS[i]*Delta_t
    return SOC

def evalua_ESS(P_ESS, SOC):
    if any(P_ESS < P_ESS_min):
        return penaliza
    if any(P_ESS > P_ESS_max):
        return penaliza
    if any(SOC < SOC_ESS_min):
        return penaliza
    if any(SOC > SOC_ESS_max):
        return penaliza
    return 0

def crea_individuo():
    """Función que crea individuos"""
    individuo = np.zeros(48)
    for i in range(0,24):
        individuo[i] = random.uniform(P_DE_min,min(P_dem[i],P_DE_max))
        if P_dem[i] < 0:
            individuo[i] = 0
        individuo[24+i] = P_dem[i] - individuo[i]
        if individuo[24+i] < P_MT_min:
            individuo[24+i] = 0
        if individuo[24+i] > P_MT_max:
            individuo[24+i] = P_MT_max
    return individuo

def mutacion(individuo, indpb):
    """Mutación Gaussiana"""
    for j, i in enumerate(individuo):
        if random.random() < indpb[0]:
            individuo[j] = random.gauss(individuo[j],30)
        if random.random() < indpb[1]:
            individuo[j] = 0
    return individuo,

def fitness(individuo):
    # Calculamos los valores de potencia
    P_DE = individuo[:24]
    P_MT = individuo[24:]
    P_ESS = P_DM - P_PV - P_WT - P_DE - P_MT
    # Evaluamos generadores despachables
    if evalua_despachable(P_DE, P_MT) == penaliza:
        return penaliza, penaliza
    # Calculamos la evolucion del estado de carga y la evaluamos
    SOC = evolucion_SOC(P_ESS, SOC_ini)
    if evalua_ESS(P_ESS, SOC) == penaliza:
        return penaliza, penaliza
    # Finalmente, si todas las restricciones se cumplen se calcula el coste
    coste_combustible = 0
    for i in range(0, 24):
        coste_combustible += coste_DE(P_DE[i])
        coste_combustible += coste_MT(P_MT[i])
    coste_ESS = np.sum(np.abs(SOC-SOC_ini))
    return coste_combustible, coste_ESS

def multiple_objetivo_ga(c, m, toolbox):
    NGEN = 1000
    MU = 3000
    LAMBDA = MU 
    CXPB = c
    MUTPB = m
   
    pop = toolbox.ini_poblacion(n = MU)
    pareto = tools.ParetoFront(similar = np.array_equal)
 
    stats = tools.Statistics(key = lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
   
    logbook = tools.Logbook()
   
    pop, logbook = algorithms.eaMuPlusLambda(pop, toolbox, MU, LAMBDA, CXPB, MUTPB, NGEN,
                              stats= stats, halloffame=pareto, verbose = False)
   
    return pop, pareto, logbook


creator.create("FitnessMin", base.Fitness, weights = (-1.0, -1.0))
creator.create("Individual", np.ndarray, fitness = creator.FitnessMin) 

toolbox = base.Toolbox() 
toolbox.register("individual", tools.initIterate, creator.Individual, crea_individuo)
toolbox.register("ini_poblacion", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", fitness)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", mutacion, indpb=(0.05, 0.05))
toolbox.register("select", tools.selNSGA2)

pop, pareto, log = multiple_objetivo_ga(0.7, 0.3, toolbox)
res_individuos = open("individuos_microrred_multi.txt", "a")
res_fitness = open("fitness_microrred_multi.txt", "a")
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











