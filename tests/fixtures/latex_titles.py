import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['mathtext.fontset'] = 'stix'

t = 1.0
x = np.linspace(0, 2 * np.pi, 400)
y = np.sin(t * x) * np.exp(-0.1 * x)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x, y, color='steelblue', lw=2)
ax.fill_between(x, y, alpha=0.15, color='steelblue')
ax.set_title(r'$f(x) = \sin(' + f'{t:.1f}' + r'x)\,e^{-0.1x}$', fontsize=16)
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$f(x)$')
ax.axhline(0, color='k', lw=0.8, ls='--')
ax.set_ylim(-1.2, 1.2)
ax.text(0.98, 0.95, r'$\omega = ' + f'{t:.2f}' + r'$',
        transform=ax.transAxes, ha='right', va='top', fontsize=13,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))
plt.tight_layout()
plt.show()
