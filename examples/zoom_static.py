import numpy as np
import matplotlib.pyplot as plt

# t drives the zoom: 0 = fully zoomed out, 1 = fully zoomed in
# Smooth ease-in-out via a cubic (smoothstep): s = 3t^2 - 2t^3
t = 0.0
s = 3 * t**2 - 2 * t**3          # smoothstep easing in [0, 1]

# Target region to zoom into (centre of the Mandelbrot-like flower pattern)
x_full = (-3.0, 3.0)
y_full = (-2.5, 2.5)
x_zoom = (-0.35, 0.35)
y_zoom = (-0.35, 0.35)

# Interpolate limits using the eased value
xlim = (x_full[0] + s * (x_zoom[0] - x_full[0]),
        x_full[1] + s * (x_zoom[1] - x_full[1]))
ylim = (y_full[0] + s * (y_zoom[0] - y_full[0]),
        y_full[1] + s * (y_zoom[1] - y_full[1]))

# Build the flower / rose curve at full resolution so it stays sharp when zoomed
theta = np.linspace(0, 2 * np.pi, 2000)
k = 5
r = np.cos(k * theta)
x = r * np.cos(theta)
y = r * np.sin(theta)

# Second layer: Lissajous figure underneath
lx = np.sin(3 * theta)
ly = np.sin(4 * theta + np.pi / 4)

fig, ax = plt.subplots(figsize=(7, 7))
ax.set_facecolor('#0d1117')
fig.patch.set_facecolor('#0d1117')

ax.plot(lx, ly, color='#1e4d8c', lw=0.8, alpha=0.5)
ax.plot(x, y, color='#5ba4f5', lw=1.5, alpha=0.9)
ax.fill(x, y, color='#1a3a6b', alpha=0.25)

ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax.set_aspect('equal')
ax.set_title(f'Zoom  t={t:.2f}  s={s:.3f}', color='white', pad=10)
ax.tick_params(colors='#555')
ax.spines['top'].set_edgecolor('#333')
ax.spines['bottom'].set_edgecolor('#333')
ax.spines['left'].set_edgecolor('#333')
ax.spines['right'].set_edgecolor('#333')

plt.tight_layout()
plt.show()
