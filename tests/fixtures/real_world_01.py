# Source: https://matplotlib.org/stable/gallery/lines_bars_and_markers/simple_plot.html
# Simple line plot from matplotlib gallery with animatable frequency parameter
import matplotlib.pyplot as plt
import numpy as np

freq = 2
t = np.arange(0.0, 2.0, 0.01)
s = 1 + np.sin(freq * np.pi * t)

fig, ax = plt.subplots()
ax.plot(t, s)

ax.set(xlabel='time (s)', ylabel='voltage (mV)',
       title='Simple line plot')
ax.grid()

plt.show()
