# Source: https://matplotlib.org/stable/gallery/lines_bars_and_markers/stem_plot.html
# Stem plot from matplotlib gallery with animatable scale factor
import matplotlib.pyplot as plt
import numpy as np

scale = 1.0

x = np.linspace(0.1, 2 * np.pi, 41)
y = np.exp(np.sin(x)) * scale

fig, ax = plt.subplots()
ax.stem(x, y)
ax.set_title('Stem plot')

plt.show()
