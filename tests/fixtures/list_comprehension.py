import numpy as np
import matplotlib.pyplot as plt

t = 0.5
x = np.linspace(0, 2 * np.pi, 100)
curves = [np.sin((t + k) * x) for k in range(4)]

fig, ax = plt.subplots()
for i, c in enumerate(curves):
    ax.plot(x, c, label=f"k={i}")
ax.set_ylim(-1.5, 1.5)
ax.legend()
ax.set_title(f"t = {t:.2f}")

plt.show()
