import numpy as np
import matplotlib.pyplot as plt

# signal parameters
f   = 10.0                          # ← this is the variable to animate
t   = np.linspace(0, 1, 1000)
y   = np.sin(2*np.pi*f*t) + 0.4*np.sin(2*np.pi*2*f*t)

# spectrum
freqs    = np.fft.rfftfreq(len(t), d=t[1]-t[0])
spectrum = np.abs(np.fft.rfft(y))

# plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6))
fig.tight_layout(pad=3)

ax1.plot(t, y, 'royalblue', lw=1.5)
ax1.set_xlim(0, 1); ax1.set_ylim(-2.2, 2.2)
ax1.set_xlabel("Time (s)"); ax1.set_ylabel("Amplitude")
ax1.set_title(f"Signal  f = {f:.1f} Hz")
ax1.grid(True, alpha=0.3)

ax2.plot(freqs, spectrum, 'tomato', lw=1.5)
ax2.set_xlim(0, 150); ax2.set_ylim(0, 600)
ax2.set_xlabel("Frequency (Hz)"); ax2.set_ylabel("|FFT|")
ax2.set_title("Frequency Spectrum")
ax2.grid(True, alpha=0.3)

plt.show()
