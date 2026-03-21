import numpy as np
import matplotlib.pyplot as plt

t = 0.5
x = np.linspace(0, 10, 100)

if t > 0.5:
    color = "red"
else:
    color = "blue"

y = np.sin(t * x)

fig, ax = plt.subplots()
ax.plot(x, y, color=color)
ax.set_ylim(-1.5, 1.5)
ax.set_title(f"t = {t:.2f}")

plt.show()
