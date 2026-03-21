import numpy as np
import matplotlib.pyplot as plt

t = 0.5
u = np.linspace(0, 2 * np.pi, 30)
v = np.linspace(0, np.pi, 20)
X = np.outer(np.cos(u), np.sin(v)) * t
Y = np.outer(np.sin(u), np.sin(v)) * t
Z = np.outer(np.ones(np.size(u)), np.cos(v)) * t

fig = plt.figure(figsize=(12, 5))
ax1 = fig.add_subplot(121, projection="3d")
ax2 = fig.add_subplot(122, projection="3d")

ax1.plot_surface(X, Y, Z, alpha=0.5, cmap="viridis")
ax1.set_title("Surface")

ax2.plot_wireframe(X, Y, Z, alpha=0.5, color="steelblue")
ax2.set_title("Wireframe")

plt.show()
