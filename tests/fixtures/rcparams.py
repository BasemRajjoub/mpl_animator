import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 14

t = 0.5
x = np.linspace(0, 2 * np.pi, 200)
y = np.sin(t * x)

fig, ax = plt.subplots()
ax.plot(x, y, lw=2)
ax.set_ylim(-1.5, 1.5)
ax.set_title(f"rcParams  t = {t:.2f}")

plt.show()
