import numpy as np
import matplotlib.pyplot as plt

t = 0.5
x = np.linspace(0, 10, 100)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for i, ax in enumerate(axes):
    ax.plot(x, np.sin((t + i) * x))
    ax.set_title(f"freq={t + i:.1f}")
    ax.set_ylim(-1.5, 1.5)

plt.tight_layout()
plt.show()
