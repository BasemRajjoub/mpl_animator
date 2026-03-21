"""2D sine wave with independent frequency and amplitude variables.

Use: --var freq amp --range "1,5" "0.3,1.5" --frames 60 --fps 25
"""
import numpy as np
import matplotlib.pyplot as plt

freq = 1.0
amp  = 1.0

x = np.linspace(0, 2 * np.pi, 400)
y = amp * np.sin(freq * x)
energy = 0.5 * np.trapz(y ** 2, x)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(x, y, lw=2, color='steelblue')
ax.fill_between(x, y, alpha=0.15, color='steelblue')
ax.set_xlim(0, 2 * np.pi)
ax.set_ylim(-2, 2)
ax.set_title(f'freq={freq:.2f} Hz   amp={amp:.2f}   energy={energy:.3f}')
ax.set_xlabel('x')
ax.set_ylabel('y')

plt.tight_layout()
plt.show()
