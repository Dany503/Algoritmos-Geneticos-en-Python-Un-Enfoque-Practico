import json
import numpy as np
import matplotlib.pyplot as plt

plt.figure()
with open("zdt1_front.json") as optimal_front_data:
        optimal_front = json.load(optimal_front_data)
        
voptimalfront = np.array(optimal_front)

plt.scatter(voptimalfront[:,0], voptimalfront[:,1], color ="b")
plt.xlabel("$fZDT11$")
plt.ylabel("$fZDT12$")
plt.annotate("Soluciones dominadas", (0.5,0.6))
plt.annotate("Soluciones no dominadas", (0.1,0.2))
plt.legend(["Frente de Pareto"])
plt.grid(True)
plt.savefig("Paretozdt1.eps", dpi = 300)
