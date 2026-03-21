# Source: https://matplotlib.org/stable/gallery/lines_bars_and_markers/scatter_with_legend.html
# Scatter plot with legend from matplotlib gallery with animatable alpha
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(19680801)

alpha = 0.3

fig, ax = plt.subplots()
for color in ['tab:blue', 'tab:orange', 'tab:green']:
    n = 750
    x, y = np.random.rand(2, n)
    scale = 200.0 * np.random.rand(n)
    ax.scatter(x, y, c=color, s=scale, label=color,
               alpha=alpha, edgecolors='none')

ax.legend()
ax.grid(True)

plt.show()
