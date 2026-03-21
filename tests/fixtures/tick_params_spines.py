import numpy as np
import matplotlib.pyplot as plt

t = 0.5
x = np.linspace(0, 10, 100)
y = np.sin(t * x)

fig, ax = plt.subplots()
ax.plot(x, y, 'b-')
ax.set_ylim(-1.5, 1.5)
ax.tick_params(axis="both", which="major", labelsize=10)
ax.set_title(f"t = {t:.2f}")

plt.show()
