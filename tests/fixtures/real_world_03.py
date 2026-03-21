# Source: https://matplotlib.org/stable/gallery/statistics/errorbar_features.html
# Errorbar plot from matplotlib gallery with animatable error scale
import matplotlib.pyplot as plt
import numpy as np

error_scale = 1.0

x = np.arange(0.1, 4, 0.5)
y = np.exp(-x)

error = error_scale * (0.1 + 0.2 * x)

fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True)
ax0.errorbar(x, y, yerr=error, fmt='-o')
ax0.set_title('variable, symmetric error')

lower_error = 0.4 * error
upper_error = error
asymmetric_error = [lower_error, upper_error]

ax1.errorbar(x, y, xerr=asymmetric_error, fmt='o')
ax1.set_title('variable, asymmetric error')
ax1.set_yscale('log')

plt.show()
