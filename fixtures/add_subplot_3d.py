import numpy as np
import matplotlib.pyplot as plt

t = 1.0
u = np.linspace(0, 2 * np.pi, 50)
v = np.linspace(0, np.pi, 50)
x = t * np.outer(np.cos(u), np.sin(v))
y = t * np.outer(np.sin(u), np.sin(v))
z = t * np.outer(np.ones(np.size(u)), np.cos(v))

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.plot_surface(x, y, z, cmap='coolwarm')
ax.set_title(f"Sphere r={t:.2f}")

plt.show()
