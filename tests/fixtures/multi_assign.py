import numpy as np
import matplotlib.pyplot as plt

t = 0.5
a = b = np.sin(t)

fig, ax = plt.subplots()
ax.bar([0, 1], [a, b], color=["steelblue", "tomato"])
ax.set_ylim(-1.5, 1.5)
ax.set_title(f"a = b = sin({t:.2f})")

plt.show()
