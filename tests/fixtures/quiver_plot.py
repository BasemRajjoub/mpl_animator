import numpy as np
import matplotlib.pyplot as plt

t = 0.5
x = np.linspace(-2, 2, 10)
y = np.linspace(-2, 2, 10)
X, Y = np.meshgrid(x, y)
U = -np.sin(t * Y)
V = np.cos(t * X)

fig, ax = plt.subplots()
ax.quiver(X, Y, U, V)
ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-2.5, 2.5)
ax.set_aspect("equal")
ax.set_title(f"Quiver  t = {t:.2f}")

plt.show()
