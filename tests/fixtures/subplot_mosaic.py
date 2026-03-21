import numpy as np
import matplotlib.pyplot as plt

t = 0.5
x = np.linspace(0, 2 * np.pi, 100)

fig, axs = plt.subplot_mosaic([["left", "right"], ["bottom", "bottom"]],
                               figsize=(10, 6))

axs["left"].plot(x, np.sin(t * x), "b-")
axs["left"].set_title("sin")
axs["left"].set_ylim(-1.5, 1.5)

axs["right"].plot(x, np.cos(t * x), "r-")
axs["right"].set_title("cos")
axs["right"].set_ylim(-1.5, 1.5)

axs["bottom"].plot(x, np.sin(t * x) * np.cos(t * x), "g-")
axs["bottom"].set_title("sin * cos")
axs["bottom"].set_ylim(-1.0, 1.0)

plt.tight_layout()
plt.show()
