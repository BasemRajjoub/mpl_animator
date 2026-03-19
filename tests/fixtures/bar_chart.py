import numpy as np
import matplotlib.pyplot as plt

t = 1.0
categories = ['A', 'B', 'C', 'D', 'E']
values = np.array([3, 7, 2, 5, 8]) * t

fig, ax = plt.subplots()
ax.bar(categories, values, color='steelblue')
ax.set_ylim(0, 40)
ax.set_title(f"Sales x{t:.1f}")

plt.show()
