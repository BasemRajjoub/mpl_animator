# Source: https://matplotlib.org/stable/gallery/lines_bars_and_markers/fill_between_demo.html
# Fill between plot from matplotlib gallery with animatable frequency
import matplotlib.pyplot as plt
import numpy as np

freq = 2.0

x = np.arange(0.0, 2, 0.01)
y1 = np.sin(freq * np.pi * x)
y2 = 0.8 * np.sin(2 * freq * np.pi * x)

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(6, 6))

ax1.fill_between(x, y1)
ax1.set_title('fill between y1 and 0')

ax2.fill_between(x, y1, 1)
ax2.set_title('fill between y1 and 1')

ax3.fill_between(x, y1, y2)
ax3.set_title('fill between y1 and y2')
ax3.set_xlabel('x')

fig.tight_layout()

plt.show()
