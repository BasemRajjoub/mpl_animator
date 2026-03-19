import numpy as np
import matplotlib.pyplot as plt

t = 0.0
np.random.seed(42)
n = 100
xs = np.cos(np.linspace(0, 2*np.pi, n) + t)
ys = np.sin(np.linspace(0, 2*np.pi, n) + t)
zs = np.linspace(0, 1, n)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xs, ys, zs, c=zs, cmap='plasma')
ax.set_title(f"Helix t={t:.2f}")

plt.show()
