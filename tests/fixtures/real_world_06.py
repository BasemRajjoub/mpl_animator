# Source: https://matplotlib.org/stable/gallery/mplot3d/wire3d.html
# 3D wireframe plot from matplotlib gallery with animatable rstride
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import axes3d

elev = 30

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

X, Y, Z = axes3d.get_test_data(0.05)

ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10)
ax.view_init(elev=elev, azim=30)

plt.show()
