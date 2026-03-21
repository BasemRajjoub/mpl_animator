import numpy as np
import matplotlib.pyplot as plt

t = 0.0
theta = np.linspace(0, 2 * np.pi, 300)
r = 1.0 + 0.5 * np.sin(3 * theta + t)
x_polar = r * np.cos(theta)
y_polar = r * np.sin(theta)

u = np.linspace(0, 2 * np.pi, 40)
v = np.linspace(0, np.pi, 40)
xs = np.outer(np.cos(u + t), np.sin(v))
ys = np.outer(np.sin(u + t), np.sin(v))
zs = np.outer(np.ones(len(u)), np.cos(v))

fig = plt.figure(figsize=(13, 5))

ax2d = fig.add_subplot(1, 2, 1)
ax3d = fig.add_subplot(1, 2, 2, projection='3d')

ax2d.plot(x_polar, y_polar, color='royalblue', lw=2)
ax2d.fill(x_polar, y_polar, alpha=0.15, color='royalblue')
ax2d.set_aspect('equal')
ax2d.set_title(f'Polar rose  t={t:.2f}')
ax2d.set_xlim(-2, 2)
ax2d.set_ylim(-2, 2)
ax2d.text(0.02, 0.98, r'$r = 1 + 0.5\sin(3\theta + t)$',
          transform=ax2d.transAxes, va='top', fontsize=9,
          bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

ax3d.plot_surface(xs, ys, zs, cmap='coolwarm', alpha=0.85)
ax3d.set_title(f'Rotating sphere  t={t:.2f}')
ax3d.set_box_aspect([1, 1, 1])

fig.suptitle(f'Mixed 2D/3D   t = {t:.2f}', fontsize=13)
plt.tight_layout()
plt.show()
