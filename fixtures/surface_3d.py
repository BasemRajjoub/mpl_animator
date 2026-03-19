import numpy as np
import matplotlib.pyplot as plt

t = 1.0
x = np.linspace(-3, 3, 50)
y = np.linspace(-3, 3, 50)
X, Y = np.meshgrid(x, y)
Z = np.sin(t * np.sqrt(X**2 + Y**2))

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis')
ax.set_zlim(-1.5, 1.5)
ax.set_title(f"t = {t:.2f}")

plt.show()
