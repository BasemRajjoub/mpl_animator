import numpy as np
import matplotlib.pyplot as plt


def wave(x, phase):
    return np.sin(x + phase)


t = 0.5
x = np.linspace(0, 2 * np.pi, 200)
y = wave(x, t)

fig, ax = plt.subplots()
ax.plot(x, y, 'b-')
ax.set_ylim(-1.5, 1.5)
ax.set_title(f"wave(x, {t:.2f})")

plt.show()
