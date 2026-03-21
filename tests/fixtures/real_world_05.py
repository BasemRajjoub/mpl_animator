# Source: https://matplotlib.org/stable/gallery/images_contours_and_fields/contourf_demo.html
# Contour filled plot from matplotlib gallery with animatable delta (grid spacing)
import matplotlib.pyplot as plt
import numpy as np

delta = 0.025

x = y = np.arange(-3.0, 3.01, delta)
X, Y = np.meshgrid(x, y)
Z1 = np.exp(-X**2 - Y**2)
Z2 = np.exp(-(X - 1)**2 - (Y - 1)**2)
Z = (Z1 - Z2) * 2

fig, ax = plt.subplots()
CS = ax.contourf(X, Y, Z, 10, cmap=plt.cm.bone)
ax.set_title('Contourf plot')
ax.set_xlabel('x')
ax.set_ylabel('y')

plt.show()
