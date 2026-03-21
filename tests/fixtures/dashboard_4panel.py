import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

t = 0.5
x = np.linspace(0, 4 * np.pi, 500)
y_sin = np.sin(t * x)
y_cos = np.cos(t * x)
y_combo = y_sin * y_cos
freqs = np.fft.rfftfreq(len(x), d=x[1] - x[0])
spectrum = np.abs(np.fft.rfft(y_sin))

fig = plt.figure(figsize=(12, 8))
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])

ax1.plot(x, y_sin, color='steelblue', lw=1.5)
ax1.set_title(r'$\sin(' + f'{t:.1f}' + r'x)$')
ax1.set_ylim(-1.5, 1.5)

ax2.plot(x, y_cos, color='tomato', lw=1.5)
ax2.set_title(r'$\cos(' + f'{t:.1f}' + r'x)$')
ax2.set_ylim(-1.5, 1.5)

ax3.plot(x, y_combo, color='purple', lw=1.5)
ax3.fill_between(x, y_combo, alpha=0.2, color='purple')
ax3.set_title(r'$\sin \cdot \cos$  product')
ax3.set_ylim(-1.0, 1.0)

ax4.plot(freqs[:80], spectrum[:80], color='darkorange', lw=1.5)
ax4.set_title(f'FFT spectrum  (f={t:.1f})')
ax4.set_xlabel('frequency')
ax4.set_xlim(0, None)

fig.suptitle(f'Signal dashboard   t = {t:.2f}', fontsize=14, fontweight='bold')
plt.show()
