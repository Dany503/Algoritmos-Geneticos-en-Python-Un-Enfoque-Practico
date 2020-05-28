import random
import matplotlib.pyplot as plt

from deap import base
from deap import creator
from deap import tools
from deap import algorithms
import numpy as np
import time

t0 = time.time()

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

def crea_individuo():
    individuo = [0]*numero*2
    for i in range(len(individuo)):
        individuo[i] = np.random.uniform(0, 2000)
    return individuo

#def crea_individuo():
#    individuo = [0]*numero*2
#    for i in range(numero):
#        p_pdi = random.randint(0, len(x) - 1)
#        individuo[i] = x[p_pdi]
#        individuo[i+numero] = y[p_pdi]
#    return individuo

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
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", mutacion, indpb = 0.05)
toolbox.register("select", tools.selTournament, tournsize = 3)

pop_new , pareto_new , log = unico_objetivo_ga(0.7, 0.3, toolbox)

gen = log.select("gen")
fit_mins = log.select("min")
fit_maxs = log.select("max")
fit_ave = log.select("avg")

fig, ax1 = plt.subplots(figsize =(8,5))
ax1.plot(gen, fit_mins , "b")
ax1.plot(gen, fit_maxs , "r")
ax1.plot(gen, fit_ave , "--k")
ax1.fill_between(gen, fit_mins , fit_maxs , where=fit_maxs >= fit_mins, facecolor= "g", alpha = 0.2)
ax1.set_xlabel("Generacion", fontsize = 15)
ax1.set_ylabel("Fitness", fontsize = 15)
ax1.set_ylim([0, 80])
ax1.legend(["Minimo", "Maximo", "Media"])
plt.savefig("Figuras/fitness_single.pdf", bbox_inches = 'tight')

t1 = time.time()
print(t1-t0)