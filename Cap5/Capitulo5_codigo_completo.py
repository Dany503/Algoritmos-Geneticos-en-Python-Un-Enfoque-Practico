# Incluimos las librerías que utilizaremos
import numpy as np
from deap import base
from deap import creator
from deap import tools
from deap import algorithms
import matplotlib.pyplot as plt
import random

# Definimos el número de objetivos y las probabilidades
multi = True
c = 0.7
m = 0.3

# Definimos los perfiles de potencia datos de nuestro problema
t = np.arange(0, 24)
P_PV = np.array([0, 0, 0, 0, 0, 0, 0, 0, 6, 10, 15, 20, 
                 30, 40, 40, 20, 15, 10, 8, 2, 0, 0, 0, 0])
P_WT = np.array([51, 51, 58, 51, 64, 51, 44, 51, 44, 51, 51, 46, 
                 81, 74, 65, 65, 65, 51, 39, 63, 38, 66, 74, 74])

P_DM = np.array([67, 67, 90, 114, 120, 130, 150, 190, 200, 206, 
                 227, 227, 250, 250, 200, 180, 160, 160, 190, 150, 
                 100, 50, 20, 20])
P_dem = P_DM - P_PV - P_WT

# Parámetros del generador diesel
P_DE_min = 5
P_DE_max = 80

# Parámetros de la microturbina
P_MT_min = 10
P_MT_max = 140

# Parámetros del sistema de almacenamiento
P_ESS_min = -120
P_ESS_max = 120
SOC_ESS_max = 280
SOC_ESS_min = 70
SOC_ini = 140

# Penalización por pena de muerte
penaliza = 99999999999999

def evalua_despachable(P_DE, P_MT):
    """ 
    Función que evalúa si las unidades despachables cumplen
    las restricciones establecidas
    """
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

def coste_DE(P_DE):
    """
    Función que devuelve el coste de combustible del 
    generador diesel
    """
    f_DE = 0.0012
    e_DE = 0.2455
    d_DE = 1.925
    if (P_DE == 0):
        return 0
    else:
        return d_DE + e_DE*P_DE + f_DE*(P_DE**2) 

def coste_MT(P_MT):
    """
    Función que devuelve el coste de combustible de la
    microturbina
    """
    f_MT = 0.0002
    e_MT = 0.2015
    d_MT = 7.4344
    if (P_MT == 0):
        return 0
    else:
        return d_MT + e_MT*P_MT + f_MT*(P_MT**2) 

def evolucion_SOC(P_ESS, SOC_ini):
    """
    Función que calcula la evolución del estado de carga a partir de
    los flujos de potencia 
    """
    Delta_t = 1
    SOC = np.zeros(24)
    SOC[0] = SOC_ini
    for i in range(1, SOC.size):
        SOC[i] = SOC[i-1] - P_ESS[i]*Delta_t
    return SOC

def evalua_ESS(P_ESS, SOC):
    """ 
    Función que evalúa si se cumplen las restricciones del 
    sistema de almacenamiento
    """
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
    """
    Función que genera de forma aleatoria individuos
    para la población inicial
    """
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
    """ 
    Función que genera una mutación en individuos padres
    """
    for j, i in enumerate(individuo):
        if random.random() < indpb[0]:
            individuo[j] = random.gauss(individuo[j],30)
        if random.random() < indpb[1]:
            individuo[j] = 0
    return individuo,
    
def plot_evolucion(log):
    gen = log.select("gen")
    fit_mins = log.select("min")
    fit_maxs = log.select("max")
    fit_ave = log.select("avg")

    fig, ax1 = plt.subplots()
    ax1.plot(gen, fit_mins, "b")
    ax1.plot(gen, fit_maxs, "r")
    ax1.plot(gen, fit_ave, "--k")
    ax1.fill_between(gen, fit_mins, fit_maxs, 
                     where=fit_maxs >= fit_mins, 
                     facecolor="g", alpha= 0.2)
    ax1.set_xlabel("Generación")
    ax1.set_ylabel("Fitness")
    ax1.legend(["Min", "Max", "Avg"])
    plt.grid(True)

def fitness(individuo):
    """
    Función de fitness
    """
    # Calculamos los valores de potencia
    P_DE = individuo[:24]
    P_MT = individuo[24:]
    P_ESS = P_DM - P_PV - P_WT - P_DE - P_MT
    # Evaluamos generadores despachables
    if evalua_despachable(P_DE, P_MT) == penaliza:
        return penaliza+1,
    # Calculamos la evolucion del estado de carga y la evaluamos
    SOC = evolucion_SOC(P_ESS, SOC_ini)
    if evalua_ESS(P_ESS, SOC) == penaliza:
        return penaliza+2,
    # Finalmente, si todas las restricciones se cumplen se calcula el coste
    coste = 0
    for i in range(0, 24):
        coste += coste_DE(P_DE[i])
        coste += coste_MT(P_MT[i])
    return coste,

def unico_objetivo_ga(c, m):
    """
    Ejecución del algoritmo genético para el caso en el
    que existe un único objetivo
    """
    MU = 3000
    LAMBDA = MU 
    CXPB, MUTB, NGEN = c, m, 1000
   
    pop = toolbox.ini_poblacion(n=MU)
    hof = tools.HallOfFame(1, similar=np.array_equal)
 
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
   
    logbook = tools.Logbook()
   
    pop, logbook = algorithms.eaMuPlusLambda(pop, toolbox, MU, LAMBDA, 
        CXPB, MUTB, NGEN, stats=stats, halloffame=hof, verbose=False)
   
    return pop, hof, logbook

def fitness_multi(individuo):
    # Calculamos los valores de potencia
    P_DE = individuo[:24]
    P_MT = individuo[24:]
    P_ESS = P_DM - P_PV - P_WT - P_DE - P_MT
   
    # Evaluamos generadores despachables
    if evalua_despachable(P_DE, P_MT) == penaliza:
        return penaliza, penaliza
  
    # Calculamos la evolución del estado de carga y la evaluamos
    SOC = evolucion_SOC(P_ESS, SOC_ini)
    if evalua_ESS(P_ESS, SOC) == penaliza:
        return penaliza, penaliza
 
    # Si todas las restricciones se cumplen se calculan los costes
    coste_combustible = 0
    for i in range(0, 24):
        coste_combustible += coste_DE(P_DE[i])
        coste_combustible += coste_MT(P_MT[i])
    coste_ESS = np.sum(np.abs(SOC-SOC_ini))
    return coste_combustible, coste_ESS



def multiple_objetivo_ga(c, m, toolbox):
    """
    Función que  realiza  la  llamada  al  algoritmo  genético
    """
    # Definimos  los parámetros  genéticos
    MU = 1500
    LAMBDA = MU 
    CXPB, MUTB, NGEN = c, m, 500
   
    # Inicializamos  la  población y el hall of fame
    pop = toolbox.ini_poblacion(n=MU)
    hof = tools.HallOfFame(1, similar=np.array_equal)
 
    # Indicamos  las  estadísticas a registrar
    stats = tools.Statistics(key = lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
   
    # Creamos  un ahoja de  registros
    logbook = tools.Logbook()
   
    # Se llama al algoritmo
    pop, logbook = algorithms.eaMuPlusLambda(pop, toolbox, 
                        MU, LAMBDA, CXPB, MUTB, NGEN,
                        stats=stats, halloffame=hof, verbose=False)
   
    # Se devuelve el resultado obtenido
    return pop, hof, logbook

if multi == False:
    creator.create("FitnessMin", base.Fitness, weights=(-1,))
    creator.create("Individual", np.ndarray, fitness=creator.FitnessMin)
     
    toolbox = base.Toolbox() 
    toolbox.register("individual", tools.initIterate, 
                     creator.Individual, crea_individuo)
    toolbox.register("ini_poblacion", tools.initRepeat, 
                     list, toolbox.individual)
    toolbox.register("evaluate", fitness)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", mutacion, indpb=(0.05, 0.05))
    toolbox.register("select", tools.selTournament, tournsize=3)
else:
    creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
    creator.create("Individual", np.ndarray, fitness=creator.FitnessMin) 
    
    toolbox = base.Toolbox() 
    toolbox.register("individual", tools.initIterate, 
                     creator.Individual, crea_individuo)
    toolbox.register("ini_poblacion", tools.initRepeat, 
                     list, toolbox.individual)
    toolbox.register("evaluate", fitness_multi)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", mutacion, indpb=(0.05, 0.05))
    toolbox.register("select", tools.selNSGA2)

# Se realiza la llamada a la resolución del problema
if multi == False:
    pop_new, hof, log = unico_objetivo_ga(c, m)
    plot_evolucion(log)
else:
    pop_new, pareto_new, log = multiple_objetivo_ga(c, m)
    
    # Se abren dos ficheros de texto para almacenar los resultados
    res_individuos = open("individuos_microrred_multi.txt", "a")
    res_fitness = open("fitness_microrred_multi.txt", "a")
    
    # Para cada punto del frente Pareto se almacenan los resultados
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