import numpy as np

# arrays de numpy

v1 = np.array([1, 2, 3, 4, 5, 6])
print(v1)
vista1 = v1[0:3]
print(vista1)
vista1[:] = 0
print(vista1)
print(v1)

#%% listas

l1 = [1, 2, 3, 4, 5, 6]
print(l1)
l1_trozo = l1[0:3]
print(l1_trozo)
l1_trozo[:] = [0, 0, 0]
print(l1_trozo)
print(l1)

#%% arrays
import array
a1 = array.array("i", [1, 2, 3, 4, 5, 6])
print(a1)
a1_trozo = a1[0:3]
print(a1_trozo)
a1_trozo[:] = array.array("i", [0, 0, 0])
print(a1_trozo)
print(a1)
