# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Lissajous 3D curve parameters
a  = 3.0          # <- variable to animate (frequency ratio X)
b  = 2.0
c  = 1.0
delta = np.pi / 4

t = np.linspace(0, 2 * np.pi, 1000)
x = np.sin(a * t + delta)
y = np.sin(b * t)
z = np.sin(c * t)

# colour by arc-length parameter
colors = plt.cm.plasma(np.linspace(0, 1, len(t)))

fig = plt.figure(figsize=(8, 6))
ax  = fig.add_subplot(111, projection='3d')

ax.scatter(x, y, z, c=colors, s=2, alpha=0.8)
ax.set_title(f"3D Lissajous  a={a:.1f}, b={b:.1f}, c={c:.1f}")
ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)

plt.tight_layout()
plt.show()
