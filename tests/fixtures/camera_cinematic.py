"""Cinematic 3D camera: azimuth + elevation both animated simultaneously.

Use: --var azim elev --range "0,360" "20,60" --frames 60 --fps 25
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

azim = 0.0
elev = 20.0

u = np.linspace(0, 2 * np.pi, 60)
v = np.linspace(0, np.pi, 30)
xs = np.outer(np.cos(u), np.sin(v))
ys = np.outer(np.sin(u), np.sin(v))
zs = np.outer(np.ones(len(u)), np.cos(v))

fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(xs, ys, zs, cmap='plasma', alpha=0.85)
ax.set_title(f'Camera  azim={azim:.0f}°  elev={elev:.0f}°')
ax.view_init(elev=elev, azim=azim)
ax.set_box_aspect([1, 1, 1])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.tight_layout()
plt.show()
