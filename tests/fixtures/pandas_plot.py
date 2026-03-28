"""Fixture: pandas DataFrame.plot() with animated parameter."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

n = 50
x = np.linspace(0, 10, n)
df = pd.DataFrame({"x": x, "y": np.sin(x)})

fig, ax = plt.subplots()
df.plot(x="x", y="y", ax=ax)
ax.set_title(f"Pandas plot n={n}")
plt.show()
