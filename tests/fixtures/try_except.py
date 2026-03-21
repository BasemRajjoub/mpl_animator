import numpy as np
import matplotlib.pyplot as plt

t = 0.5
x = np.linspace(0, 10, 100)

try:
    y = np.sin(t * x)
except Exception:
    y = np.zeros(100)

fig, ax = plt.subplots()
ax.plot(x, y, 'b-')
ax.set_ylim(-1.5, 1.5)
ax.set_title(f"try/except  t = {t:.2f}")

plt.show()
