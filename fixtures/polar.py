import numpy as np
import matplotlib.pyplot as plt

t = 3.0
theta = np.linspace(0, 2 * np.pi, 300)
r = 1 + t * np.cos(5 * theta)

fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.plot(theta, r, 'purple')
ax.set_title(f"Rose curve, a={t:.1f}")

plt.show()
