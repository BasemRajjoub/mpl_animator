# Source: https://matplotlib.org/stable/gallery/lines_bars_and_markers/bar_stacked.html
# Stacked bar chart from matplotlib gallery with animatable bar width
import matplotlib.pyplot as plt
import numpy as np

width = 0.5

species = ("Adelie", "Chinstrap", "Gentoo")
below = np.array([70, 31, 58])
above = np.array([82, 37, 66])

fig, ax = plt.subplots()

ax.bar(species, below, width, label="Below")
ax.bar(species, above, width, label="Above", bottom=below)

ax.set_title("Number of penguins with above average body mass")
ax.legend(loc="upper right")

plt.show()
