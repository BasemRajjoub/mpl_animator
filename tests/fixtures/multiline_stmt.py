import numpy as np
import matplotlib.pyplot as plt

t = 2.0
x = np.linspace(0, 10, 200)
y = np.sin(
    t * x
) * np.exp(-0.1 * x)

fig, ax = plt.subplots(
    figsize=(8, 4)
)
ax.plot(
    x, y,
    'blue',
    linewidth=2
)
ax.set_ylim(-1.5, 1.5)
ax.set_title(f"Damped wave, freq={t:.1f}")

plt.show()
