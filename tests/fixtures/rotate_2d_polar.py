import numpy as np
import matplotlib.pyplot as plt

angle = 0.0     # <- animate this: rotates the polar axis offset 0 -> 2*pi

theta = np.linspace(0, 2 * np.pi, 500)
r = 1 + 0.5 * np.sin(5 * theta)

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(1, 1, 1, projection='polar')
ax.plot(theta, r, color='royalblue', lw=2)
ax.fill(theta, r, alpha=0.2, color='royalblue')
ax.set_theta_offset(angle)
ax.set_title(f"Rotating polar  angle={np.degrees(angle):.0f} deg", pad=15)

plt.show()
