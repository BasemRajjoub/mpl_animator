import numpy as np
import matplotlib.pyplot as plt

t = 0.5
x = np.linspace(0, 4 * np.pi, 300)
y = np.sin(t * x)

fig, ax = plt.subplots()
ax.plot(x, y, 'b-')
ax.fill_between(x, y, where=(y > 0), alpha=0.3, color="green")
ax.fill_between(x, y, where=(y < 0), alpha=0.3, color="red")
ax.set_ylim(-1.5, 1.5)
ax.set_title(f"fill_between  t = {t:.2f}")

plt.show()
