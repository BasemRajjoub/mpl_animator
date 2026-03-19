import numpy as np
import matplotlib.pyplot as plt

t = 1.0
x = np.linspace(0, 10, 200)
y = np.sin(x)
y += t * np.cos(2 * x)

fig, ax = plt.subplots()
ax.plot(x, y, 'teal')
ax.set_ylim(-3, 3)
ax.set_title(f"Combined waves, t={t:.1f}")

plt.show()
