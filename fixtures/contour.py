import numpy as np
import matplotlib.pyplot as plt

t = 1.0
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
Z = np.sin(t * X) * np.cos(t * Y)

fig, ax = plt.subplots()
ax.contourf(X, Y, Z, levels=20, cmap='RdBu')
ax.set_title(f"Contour t={t:.1f}")

plt.show()
