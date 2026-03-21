import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

t = 0.5

fig, ax = plt.subplots()
c = Circle((t, 0.5), 0.2, color="steelblue", alpha=0.7)
ax.add_patch(c)
ax.set_xlim(-0.5, 2)
ax.set_ylim(-0.5, 1.5)
ax.set_aspect("equal")
ax.set_title(f"x = {t:.2f}")

plt.show()
