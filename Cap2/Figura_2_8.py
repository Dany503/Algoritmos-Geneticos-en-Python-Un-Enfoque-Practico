# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 20:16:59 2020

@author: dguti
"""

import math
import matplotlib.pyplot as plt
plt.style.use("libroGA.mplstyle")

x_values = range(1, 20)

x_cuadrado = [x**2 for x in x_values]
x_log = [math.log(x) for x in x_values]
x_lineal = [2*x for x in x_values]
x_xlog = [x*math.log(x) for x in x_values]
x_cubico = [x**3 for x in x_values]
x_exp = [2**x for x in x_values]

plt.plot(x_values, x_cuadrado, marker="*", color="b", label="$n^2$")
plt.plot(x_values, x_lineal, marker="*", color="g", label="$2*n$")
plt.plot(x_values, x_xlog, marker="*", color="y", label="$n*\log(n)$")
plt.plot(x_values, x_cubico, marker="*", color="m", label="$n^3$")
plt.plot(x_values, x_exp, marker="*", color="r", label="$2^n$")
plt.yscale("log")
plt.xlabel("$n$")
plt.ylabel("Operaciones")
plt.grid(True)
plt.legend()
plt.savefig("complejidad.eps", dpi=300)