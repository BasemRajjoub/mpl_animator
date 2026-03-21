import numpy as np
import matplotlib.pyplot as plt

t = 0.0
x = np.linspace(0, 2 * np.pi, 300)
y = np.sin(x + t)
energy = 0.5 * np.trapz(y ** 2, x)
phase_deg = np.degrees(t) % 360

labels = ['low', 'rising', 'high', 'falling']
label = labels[int((t / (2 * np.pi)) * len(labels)) % len(labels)]

fig, (ax_plot, ax_text) = plt.subplots(1, 2, figsize=(11, 5),
                                        gridspec_kw={'width_ratios': [3, 1]})

ax_plot.plot(x, y, lw=2, color='steelblue')
ax_plot.fill_between(x, y, alpha=0.1, color='steelblue')
ax_plot.set_ylim(-1.5, 1.5)
ax_plot.set_title(f'Phase shift = {t:.3f} rad')
ax_plot.set_xlabel('x')
ax_plot.set_ylabel('y')

ax_text.axis('off')
ax_text.text(0.5, 0.80, 'Stats', ha='center', fontsize=14, fontweight='bold',
             transform=ax_text.transAxes)
ax_text.text(0.5, 0.60, f'Phase: {phase_deg:.1f} deg', ha='center', fontsize=11,
             transform=ax_text.transAxes)
ax_text.text(0.5, 0.42, f'Energy: {energy:.3f}', ha='center', fontsize=11,
             transform=ax_text.transAxes)
ax_text.text(0.5, 0.24, f'State: {label}', ha='center', fontsize=12,
             color='darkgreen', fontweight='bold', transform=ax_text.transAxes,
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

plt.tight_layout()
plt.show()
