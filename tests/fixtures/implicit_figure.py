import numpy as np
import matplotlib.pyplot as plt

t = 1.0
x = np.linspace(0, 5, 100)
y = t * x ** 2

plt.plot(x, y, 'r-')
plt.title(f"y = {t:.1f} * x^2")
plt.ylim(0, 50)

plt.show()
