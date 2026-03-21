import numpy as np
import matplotlib.pyplot as plt

t = 0.5
x = np.linspace(1, 100, 200)
y = x ** t

fig, ax = plt.subplots()
ax.plot(x, y, 'b-')
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_title(f"y = x^{t:.2f}")

plt.show()
