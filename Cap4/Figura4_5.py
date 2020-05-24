import json
import numpy as np
import matplotlib.pyplot as plt

plt.figure()
with open("zdt1_front.json") as optimal_front_data:
        optimal_front_zdt1 = json.load(optimal_front_data)

with open("zdt2_front.json") as optimal_front_data_zdt2:
        optimal_front_zdt2 = json.load(optimal_front_data_zdt2)

with open("zdt3_front.json") as optimal_front_data_zdt3:
        optimal_front_zdt3 = json.load(optimal_front_data_zdt3)
        
zdt1 = np.array(optimal_front_zdt1)
zdt2 = np.array(optimal_front_zdt2)
zdt3 = np.array(optimal_front_zdt3)

plt.scatter(zdt1[:,0], zdt1[:,1], color ="b", label = "ZDT1")
plt.scatter(zdt2[:,0], zdt2[:,1], color ="r", label = "ZDT2")
plt.scatter(zdt3[:,0], zdt3[:,1], color ="g", label = "ZDT3")
plt.xlabel("$F1$")
plt.ylabel("$F2$")
plt.legend()
plt.grid(True)
plt.savefig("tipos_frente.eps", dpi = 300)
