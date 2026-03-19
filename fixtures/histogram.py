import numpy as np
import matplotlib.pyplot as plt

t = 1.0
np.random.seed(0)
data = np.random.normal(0, t, 1000)

fig, ax = plt.subplots()
ax.hist(data, bins=30, color='coral', edgecolor='black')
ax.set_xlim(-5, 5)
ax.set_ylim(0, 150)
ax.set_title(f"Normal dist, std={t:.2f}")

plt.show()
