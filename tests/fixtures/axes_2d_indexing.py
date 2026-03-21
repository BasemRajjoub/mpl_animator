import numpy as np
import matplotlib.pyplot as plt

t = 0.5
x = np.linspace(0, 2 * np.pi, 100)

fig, axs = plt.subplots(2, 2, figsize=(8, 8))

axs[0, 0].plot(x, np.sin(t * x))
axs[0, 0].set_title("sin(tx)")
axs[0, 0].set_ylim(-1.5, 1.5)

axs[0, 1].plot(x, np.cos(t * x))
axs[0, 1].set_title("cos(tx)")
axs[0, 1].set_ylim(-1.5, 1.5)

axs[1, 0].plot(x, np.sin(2 * t * x))
axs[1, 0].set_title("sin(2tx)")
axs[1, 0].set_ylim(-1.5, 1.5)

axs[1, 1].plot(x, np.cos(2 * t * x))
axs[1, 1].set_title("cos(2tx)")
axs[1, 1].set_ylim(-1.5, 1.5)

plt.tight_layout()
plt.show()
