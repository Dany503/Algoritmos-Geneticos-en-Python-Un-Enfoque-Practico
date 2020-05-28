# Importamos lasl ibrerías necesarias
import random
import numpy as np
from deap import base
from deap import creator
from deap import tools
from deap import algorithms

# Definimos el tipo de objetivo
multi = False

# Definimos las funciones utilizadas
def cobertura(interes, conexion):
    """
    Función que recibe un punto de interés y uno de conexión y evalúa la 
    distancia entre ambos. Si la distancia es menor a 100 metros (alcance) 
    devuelve True, en caso contrario False.
    """
    # Calculamos distancia entre ambos puntos
    distancia = np.sqrt((interes[0]-conexion[0])**2 
                      + (interes[1]-conexion[1])**2)
    # Evaluamos si esta dentro del alcance
    if distancia <= alcance:
        return True
    else:
        return False

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

def crea_individuo():
    """
    Función que genera individuos de forma aleatoria
    """
    # inicializamos el individuo como un vector de 2*(numero de puntos)
    individuo = [0]*numero*2
    
    # Generamos un punto de conexión en la posición de 
    # un punto de interés aleatorio
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
    for i in range(len(individuo)):
        # Se genera un número aleatorio para decidir si el gen 
        # correspondiente debe mutar o no
        if random.random() < indpb:  
            # En caso de que el gen mute, se varía su valor de acuerdo a 
            # una distribución normal centrada en el valor previo y con
            # desviación típica 50
            individuo[i] = random.gauss(individuo[i], 50)
            # Se comprueba si las coordenadas siguen dentro del área. 
            # En caso contrario se fuerza.
            if individuo[i] < 0:
                individuo[i] = 0
            if individuo[i] > 2000:
                individuo[i] = 2000
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
    ax1.set_xlabel("Generaci�n")
    ax1.set_ylabel("Fitness")
    ax1.legend(["Min", "Max", "Avg"])
    plt.grid(True)

def fitness(individuo):
    # Separamos los valores de x e y de los puntos de conexión
    x_pdc = individuo[0::2]
    y_pdc = individuo[1::2]
    
    # Vector para indicar qué puntos de interés están cubiertos.
    # Inicialmente indicamos con un 0 que ninguno se cubre
    pdi_vector = [0]*75
    
    # Para cada punto de conexión (x_pdc,y_pdc)
    for pdc in zip(x_pdc,y_pdc):
        # Si están fuera del área, se descarta
        if area(pdc) == False: 
            return penaliza,
        # Para cada punto de interés (x,y)
        for index, pdi in enumerate(zip(x,y)):
            # Si el punto no esta cubierto por otro punto de conexión
            if pdi_vector[index] == 0: 
                # Analizamos la cobertura
                if cobertura(pdi, pdc):
                    pdi_vector[index] = 1
    
    # Devolvemos el número de puntos cubiertos
    return sum(pdi_vector), 

def fitness_multi(individuo):
    # Separamos los valores de x e y de los puntos de conexión
    x_pdc = individuo[0::2]
    y_pdc = individuo[1::2]
    
    # Vector para indicar qué puntos de interés están cubiertos
    pdi_vector = [0]*75
    pdi_vector_2 = [0]*75
    
    # Para cada punto de conexión (x_pdc,y_pdc)
    for pdc in zip(x_pdc, y_pdc):
        # Si están fuera del área, se descarta
        if area(pdc) == False: 
            return penaliza,
        # Para cada punto de interés (x,y)
        for index, pdi in enumerate(zip(x, y)):
            if cobertura(pdi, pdc):
                pdi_vector[index] = 1
                pdi_vector_2[index] += 1
    # Devolvemos el número de puntos cubiertos
    return np.sum(pdi_vector), np.sum(pdi_vector_2) 

def unico_objetivo_ga(c, m):
    """
    Función que realiza la llamada al algoritmo genético
    """
    # Definimos los parámetros genéticos
    MU = 300
    LAMBDA = MU 
    CXPB, MUTPB, NGEN = c, m, 700
   
    # Inicializamos la población y el hall of fa
    pop = toolbox.ini_poblacion(n = MU)
    hof = tools.HallOfFame(1, similar = np.array_equal)
 
    # Indicamos las estadísticas a registrar
    stats = tools.Statistics(key = lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
   
    # Creamos una hojas de regitros
    logbook = tools.Logbook()
   
    # Se llama al algoritmo 
    pop, logbook = algorithms.eaMuPlusLambda(pop, toolbox, 
                    MU, LAMBDA, CXPB, MUTPB, NGEN,
                    stats= stats, halloffame=hof, verbose = False)
   
    # Se devuelve el resultado obtenido
    return pop, hof, logbook

def multiple_objetivo_ga(c, m):
    """ 
    Función que realiza la llamada al algoritmo genético
    """
    # Definimos los parámetros genéticos
    MU = 300 
    LAMBDA = MU 
    CXPB, MUTPB, NGEN = c, m, 800
   
    # Inicializamos la población y el frente pareto
    pop = toolbox.ini_poblacion(n=MU)
    hof = tools.ParetoFront()
 
    # Indicamos las estadísticas a registrar
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
   
    # Creamos una hoja de registros
    logbook = tools.Logbook()
   
    # Se llama al algoritmo
    pop, logbook = algorithms.eaMuPlusLambda(pop, toolbox, 
                    MU, LAMBDA, CXPB, MUTPB, NGEN,
                    stats= stats, halloffame=hof, verbose = False)
   
    # Se devuelve el resultado obtenido
    return pop, hof, logbook

# Con este comando conseguimos que siempre que se ejecute el script 
# se generen los mimos números aleatorios
random.seed(1)

# Generamos 75 coordenadas "x" e "y" para cada uno de los puntos
x= [random.uniform(0,2000) for _ in range(75)]
y= [random.uniform(0,2000) for _ in range(75)]

# Definimos el alcance de los puntos de conexión
alcance = 100

# Definimos el número de putnos de conexión
numero = 50

# Pena de muerte
penaliza = -9999

if multi == False:
    # Se define el problema de maximización y la estructura de los indviduos
    creator.create("FitnessMax", base.Fitness, weights = (+1,))
    creator.create("Individual", np.ndarray, fitness = creator.FitnessMax)
     
    # Creamos el toolbox
    toolbox = base.Toolbox()
    
    # Indicamos qué función proveerá al algoritmo de la población inicial
    toolbox.register("individual", tools.initIterate, 
                        creator.Individual, crea_individuo)
    toolbox.register("ini_poblacion", tools.initRepeat, 
                    list, toolbox.individual)
    
    # Se define la estrategia de crossover
    toolbox.register("mate", tools.cxBlend, alpha = 0.5)
    
    # Se indica la función de mutación
    toolbox.register("mutate", mutacion, indpb = 0.05)
    
    # Se define el algoritmo de selección
    toolbox.register("select", tools.selTournament, tournsize = 3)
                    
    # Definimos la función de fitness como evaluadora de nuestro problema
    toolbox.register("evaluate", fitness)
else:
    # Creación del problema de maximización mutliobjetivo
    creator.create("FitnessMax", base.Fitness, weights=(+1,+1))
    # Se define la estructura del individuo considerado
    creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)
     
    # Creamos la caja de herramientas
    toolbox = base.Toolbox() 
    # Registramos el individuo y la creación de la población inicial
    toolbox.register("individual", tools.initIterate, 
                        creator.Individual, crea_individuo)
    toolbox.register("ini_poblacion", tools.initRepeat, 
                    list, toolbox.individual)
    # Registramos las funciones de cruce, mutación, selección y evaluación
    toolbox.register("evaluate", fitness)
    toolbox.register("mate", tools.cxBlend, alpha = 0.5)
    toolbox.register("mutate", mutacion, indpb=0.05)
    toolbox.register("select", tools.selNSGA2)

# Llamada al algoritmo
if multi == False:
    pop_new , hof , log = unico_objetivo_ga(0.7, 0.3)
    plot_evolucion(log)
else:
    # Llamada a la función que lanza el algoritmo
    pop, pareto , log = multiple_objetivo_ga(0.7, 0.3)
    
    # Se abren dos ficheros de texto para almacenar los resultados
    res_individuos = open("individuos_sensores_multi.txt", "a")
    res_fitness = open("fitness_sensores_multi.txt", "a")
    
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










