import numpy as np
import matplotlib.pyplot as plt

t = 0.5
first, *rest = [np.sin(t * k) for k in range(1, 5)]

fig, ax = plt.subplots()
ax.bar(range(len(rest)), rest, color="steelblue")
ax.axhline(y=first, color="red", linestyle="--", label=f"first={first:.2f}")
ax.set_ylim(-1.5, 1.5)
ax.legend()
ax.set_title(f"Starred assignment  t = {t:.2f}")

plt.show()
