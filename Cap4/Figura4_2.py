import matplotlib.pyplot as plt
x = [10, 10, 20, 20]
y = [10, 20, 10, 20]

plt.scatter(x, y, marker = "*", color = "r", s = 60)
plt.axis([0, 30, 0, 30])
plt.xlabel("F1")
plt.ylabel("F2")
plt.annotate("S1", (x[0]+0.5, y[0]+0.5))
plt.annotate("S2", (x[1]+0.5, y[1]+0.5))
plt.annotate("S3", (x[2]+0.5, y[2]+0.5))
plt.annotate("S4", (x[3]+0.5, y[3]+0.5))
plt.grid(True)
plt.savefig("SolucionesPareto.eps", dpi = 300)