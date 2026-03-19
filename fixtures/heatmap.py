import numpy as np
import matplotlib.pyplot as plt

t = 1.0
x = np.linspace(-2, 2, 100)
y = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x, y)
Z = np.exp(-t * (X**2 + Y**2))

fig, ax = plt.subplots()
ax.imshow(Z, extent=[-2, 2, -2, 2], cmap='hot', origin='lower')
ax.set_title(f"Gaussian, sigma={t:.2f}")

plt.show()
