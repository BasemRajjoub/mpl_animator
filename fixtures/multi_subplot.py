import numpy as np
import matplotlib.pyplot as plt

t = 1.0
x = np.linspace(0, 4 * np.pi, 300)
y1 = np.sin(t * x)
y2 = np.cos(t * x)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

ax1.plot(x, y1, 'r-')
ax1.set_title(f"sin({t:.1f}x)")
ax1.set_ylim(-1.5, 1.5)

ax2.plot(x, y2, 'g-')
ax2.set_title(f"cos({t:.1f}x)")
ax2.set_ylim(-1.5, 1.5)

plt.show()
