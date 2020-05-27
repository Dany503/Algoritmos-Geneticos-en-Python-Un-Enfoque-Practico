import numpy as np
import array

# comparación de listas

l1 = [10, 20, 30]
l2 = [10, 30, 40]
print(l2>l1)

#%% comparación de arrays

v1 = array.array("i", [10, 20, 30])
v2 = array.array("i", [10, 30, 40])
print(v2>v1)

#%% comparación arrays de numpy

v1 = np.array([10, 20, 30])
v2 = np.array([10, 30, 40])
print(v2>v1)
