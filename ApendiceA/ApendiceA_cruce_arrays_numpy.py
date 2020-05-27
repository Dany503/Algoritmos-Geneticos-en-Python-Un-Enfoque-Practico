import numpy as np
import random

v1 = np.array([1, 2, 3, 4, 5, 6])
v2 = np.array([10, 20, 30, 40, 50, 60])
def cxOnePoint(ind1, ind2):
    size = min(len(ind1), len(ind2))
    cxpoint = random.randint(1, size - 1)
    ind1[cxpoint:], ind2[cxpoint:] = ind2[cxpoint:].copy(), ind1[cxpoint:].copy()
    return ind1, ind2

v3, v4 = cxOnePoint(v1, v2)
 
print(v3)
print(v4)
