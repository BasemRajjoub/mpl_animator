import numpy as np
from matplotlib import pyplot as plt

t = 0.5
x = np.linspace(0, 2 * np.pi, 200)
y = np.sin(t * x)

fig, ax = plt.subplots()
ax.plot(x, y, 'b-')
ax.set_ylim(-1.5, 1.5)
ax.set_title(f"from matplotlib import pyplot  t = {t:.2f}")

plt.show()
