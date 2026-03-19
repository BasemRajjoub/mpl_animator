import numpy as np
import matplotlib.pyplot as plt

azim = 0.0      # <- animate this: camera azimuth rotates 0 -> 360 degrees

np.random.seed(42)
n = 200
x = np.random.randn(n)
y = np.random.randn(n)
z = np.random.randn(n)
colors = np.sqrt(x**2 + y**2 + z**2)

fig = plt.figure(figsize=(7, 6))
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.scatter(x, y, z, c=colors, cmap='viridis', s=20, alpha=0.7)
ax.set_title(f"3D scatter  azim={azim:.0f} deg")
ax.view_init(elev=20, azim=azim)

plt.show()
