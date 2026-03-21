import numpy as np
import matplotlib.pyplot as plt

t = 0.0
x = np.linspace(0, 2 * np.pi, 400)
y = np.sin(x)

peak_x = np.pi / 2 + t
peak_y = np.sin(peak_x)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x, y, 'royalblue', lw=2)
ax.scatter([peak_x], [peak_y], color='crimson', s=80, zorder=5)
ax.annotate(
    f'peak\n({peak_x:.2f}, {peak_y:.2f})',
    xy=(peak_x, peak_y),
    xytext=(peak_x + 0.6, peak_y - 0.4),
    fontsize=10,
    arrowprops=dict(arrowstyle='->', color='crimson'),
    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor='crimson'),
)
ax.set_xlim(0, 2 * np.pi)
ax.set_ylim(-1.5, 1.5)
ax.set_title(f'Traveling annotation  t={t:.2f}')
ax.set_xlabel('x')
ax.set_ylabel('sin(x)')
plt.tight_layout()
plt.show()
