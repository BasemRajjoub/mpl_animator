import numpy as np
import matplotlib.pyplot as plt

t = 0.5
x = np.linspace(0, 2 * np.pi, 100)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 8))

ax1.plot(x, np.sin(t * x))
ax1.set_title("sin")
ax1.set_ylim(-1.5, 1.5)

ax2.plot(x, np.cos(t * x))
ax2.set_title("cos")
ax2.set_ylim(-1.5, 1.5)

ax3.plot(x, np.sin(2 * t * x))
ax3.set_title("sin(2t)")
ax3.set_ylim(-1.5, 1.5)

ax4.plot(x, np.cos(2 * t * x))
ax4.set_title("cos(2t)")
ax4.set_ylim(-1.5, 1.5)

plt.tight_layout()
plt.show()
