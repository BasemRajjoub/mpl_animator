"""Cinematic 3D demo: camera orbits while a torus knot spins on its own axis.

Three variables sweep simultaneously:
  azim  -- camera azimuth  (0 -> 360 deg)   : full orbit around the object
  spin  -- object self-rotation (0 -> 2*pi) : the knot spins on its vertical axis
  elev  -- camera elevation (20 -> 40 deg)  : slow crane-up for cinema feel

Animate with:
    python mpl_animator.py examples/orbit_static.py \
        --var azim spin elev \
        --range "0,360" "0,6.28318" "20,40" \
        --frames 90 --fps 25 --dpi 120 --ping-pong
    python orbit_static_animated.py --sequential
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ── Animated variables (swept each frame) ────────────────────────────────────
azim = 45.0          # camera azimuth in degrees
spin = 0.0           # object self-rotation angle in radians
elev = 20.0          # camera elevation in degrees

# ── Torus-knot geometry ───────────────────────────────────────────────────────
# p=2, q=3 torus knot gives a compact, visually rich shape
p, q = 2, 3
N = 300
phi = np.linspace(0, 2 * np.pi, N)

R, r_tube = 1.0, 0.25          # major radius, tube radius
tube_u = np.linspace(0, 2 * np.pi, 20)

# Spine of the knot (parametric curve on a torus)
cx = (R + r_tube * np.cos(q * phi)) * np.cos(p * phi)
cy = (R + r_tube * np.cos(q * phi)) * np.sin(p * phi)
cz = r_tube * np.sin(q * phi)

# Apply object self-rotation around Z axis
cx_rot = cx * np.cos(spin) - cy * np.sin(spin)
cy_rot = cx * np.sin(spin) + cy * np.cos(spin)
cz_rot = cz

# Build tube surface around the spine
# Frenet-Serret tangent
dx = np.gradient(cx_rot)
dy = np.gradient(cy_rot)
dz = np.gradient(cz_rot)
tang_len = np.sqrt(dx**2 + dy**2 + dz**2) + 1e-12
tx, ty, tz = dx / tang_len, dy / tang_len, dz / tang_len

# Normal (cross tangent with up, fallback to Z)
ux = ty * 0 - tz * 1
uy = tz * 0 - tx * 0
uz = tx * 1 - ty * 0
n_len = np.sqrt(ux**2 + uy**2 + uz**2) + 1e-12
nx, ny, nz = ux / n_len, uy / n_len, uz / n_len

# Binormal
bx = ty * nz - tz * ny
by = tz * nx - tx * nz
bz = tx * ny - ty * nx

tube_r = 0.12
xs = cx_rot[:, None] + tube_r * (nx[:, None] * np.cos(tube_u) + bx[:, None] * np.sin(tube_u))
ys = cy_rot[:, None] + tube_r * (ny[:, None] * np.cos(tube_u) + by[:, None] * np.sin(tube_u))
zs = cz_rot[:, None] + tube_r * (nz[:, None] * np.cos(tube_u) + bz[:, None] * np.sin(tube_u))

# ── Ground ring (static reference) ───────────────────────────────────────────
ring_t = np.linspace(0, 2 * np.pi, 120)
ring_x = 1.4 * np.cos(ring_t)
ring_y = 1.4 * np.sin(ring_t)
ring_z = np.full_like(ring_t, -0.45)

# ── Figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(7, 7), facecolor='#0a0a12')
ax = fig.add_subplot(111, projection='3d')
ax.set_facecolor('#0a0a12')

# Object surface — colour mapped by height
ax.plot_surface(xs, ys, zs, cmap='plasma', alpha=0.92, linewidth=0, antialiased=True)

# Ground ring
ax.plot(ring_x, ring_y, ring_z, color='#304060', lw=1.2, alpha=0.6)

# Camera position
ax.view_init(elev=elev, azim=azim)

# Axes limits and style
lim = 1.6
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_zlim(-lim, lim)
ax.set_box_aspect([1, 1, 1])

ax.set_title(f'azim={azim:.0f}°  elev={elev:.0f}°  spin={np.degrees(spin):.0f}°',
             color='#aabbcc', pad=12, fontsize=10)

ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor('#1a1a2e')
ax.yaxis.pane.set_edgecolor('#1a1a2e')
ax.zaxis.pane.set_edgecolor('#1a1a2e')
ax.tick_params(colors='#445566', labelsize=7)
ax.xaxis.label.set_color('#445566')
ax.yaxis.label.set_color('#445566')
ax.zaxis.label.set_color('#445566')

plt.tight_layout()
plt.show()
