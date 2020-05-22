import numpy as np
import matplotlib.pyplot as plt
plt.style.use("libroGA.mplstyle")

def f(x):
    return np.sin(5 * x) * (1 - np.tanh(x ** 2)) 

x = np.linspace(-2, 2, 400)

y = [f(i) for i in x]

plt.plot(x, y)
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.ylim([-1, 1])
plt.savefig("Grafica.png", dpi = 300)


