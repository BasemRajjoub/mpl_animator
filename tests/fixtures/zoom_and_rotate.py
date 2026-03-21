"""3D surface that simultaneously rotates (azim) and zooms (xlim/ylim/zlim scale).

Use: --var azim zoom --range "0,360" "0.5,1.5" --frames 60 --fps 25
"""
import numpy as np
import matplotlib.pyplot as plt

azim = 45.0
zoom = 1.0

u = np.linspace(0, 2 * np.pi, 50)
v = np.linspace(0, np.pi, 25)
xs = zoom * np.outer(np.cos(u), np.sin(v))
ys = zoom * np.outer(np.sin(u), np.sin(v))
zs = zoom * np.outer(np.ones(len(u)), np.cos(v))

fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(xs, ys, zs, cmap='viridis', alpha=0.9)
ax.set_title(f'azim={azim:.0f}°  zoom={zoom:.2f}')
ax.view_init(elev=25, azim=azim)
ax.set_xlim(-zoom * 1.1, zoom * 1.1)
ax.set_ylim(-zoom * 1.1, zoom * 1.1)
ax.set_zlim(-zoom * 1.1, zoom * 1.1)

plt.tight_layout()
plt.show()
