import numpy as np
import matplotlib.pyplot as plt

t = 0.5
x = np.linspace(0, 10, 100)
y1 = np.sin(t * x)
y2 = np.exp(-0.1 * t * x)

fig, ax = plt.subplots()
ax.plot(x, y1, 'b-', label='sin')
ax.set_ylabel('sin', color='b')

ax2 = ax.twinx()
ax2.plot(x, y2, 'r-', label='exp')
ax2.set_ylabel('exp', color='r')

ax.set_title(f"Twin axes  t={t:.2f}")
plt.show()
