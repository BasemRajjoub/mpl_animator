"""Fixture: augmented assignment inside a for-loop that depends on animated var."""
import matplotlib.pyplot as plt
import numpy as np

n = 5
x = np.linspace(0, 10, 100)
y = np.zeros(100)
for k in range(n):
    y += np.sin(k * x)

fig, ax = plt.subplots()
ax.plot(x, y)
ax.set_title(f"Fourier sum n={n}")
plt.show()
