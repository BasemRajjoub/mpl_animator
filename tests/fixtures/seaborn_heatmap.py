"""Fixture: seaborn heatmap with animated parameter."""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

n = 10
data = np.random.default_rng(42).random((n, n))

fig, ax = plt.subplots()
sns.heatmap(data, ax=ax, vmin=0, vmax=1)
ax.set_title(f"Heatmap {n}x{n}")
plt.show()
