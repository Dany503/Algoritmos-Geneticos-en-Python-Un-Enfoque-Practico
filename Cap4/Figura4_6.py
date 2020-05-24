import matplotlib.pyplot as plt
x = [10, 10, 20]
y = [10, 20, 10]

plt.plot(x[:2], y[:2], marker = "*", color = "r", label = "Frente Pareto")
plt.plot([x[0], x[-1]], [y[0], y[-1]], marker = "*", color = "r")
plt.plot([x[0], x[-1]], [y[1], y[-1]], color = "g", ls = "--", label = "Distancia de S1")
plt.plot([10, 10], [20, 40], color ="r")
plt.plot([20, 40], [10, 10], color ="r")
plt.axis([0, 25, 0, 30])
plt.xlabel("F1")
plt.ylabel("F2")
plt.annotate("S1", (x[0]+0.5, y[0]+0.5))
plt.annotate("S2", (x[1]+0.5, y[1]+0.5))
plt.annotate("S3", (x[2]+0.5, y[2]+0.5))
plt.annotate("$dS1$", (15, 15))
plt.legend(loc="upper right")
plt.grid(True)
plt.savefig("DistanciaS1.eps", dpi = 300)
