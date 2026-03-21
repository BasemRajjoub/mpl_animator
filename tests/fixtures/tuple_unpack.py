import numpy as np
import matplotlib.pyplot as plt

t = 0.5
x_val, y_val = np.sin(t), np.cos(t)

fig, ax = plt.subplots()
ax.plot([0, x_val], [0, y_val], 'ro-', markersize=10)
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_aspect("equal")
ax.set_title(f"t = {t:.2f}")

plt.show()
