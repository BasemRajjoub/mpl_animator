import numpy as np
import matplotlib.pyplot as plt

azim = 0.0      # <- animate this: camera azimuth rotates 0 -> 360 degrees

u = np.linspace(0, 2 * np.pi, 60)
v = np.linspace(0, np.pi, 60)
x = np.outer(np.cos(u), np.sin(v))
y = np.outer(np.sin(u), np.sin(v))
z = np.outer(np.ones(np.size(u)), np.cos(v))

fig = plt.figure(figsize=(7, 6))
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.plot_surface(x, y, z, cmap='plasma', alpha=0.9)
ax.set_title(f"Rotating sphere  azim={azim:.0f} deg")
ax.view_init(elev=25, azim=azim)

plt.show()
