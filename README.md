# mpl-animator

Turn any static matplotlib script into an animated GIF or MP4 by sweeping a variable — no rewriting required.

## Install

```bash
pip install mpl-animator
```

Or just drop `mpl_animator.py` into your project and use it directly.

## Usage

```bash
# Sweeps f from 3→60, writes wave_static_animated.gif
mpl-animator wave_static.py --var f --range "3,60"

# MP4 instead (requires ffmpeg)
mpl-animator plot.py --var t --range "0,2*pi" --format mp4

# Multiple variables at once
mpl-animator orbit.py --var azim spin elev --range "0,360" "0,6.28" "20,40" --ping-pong

# Discrete values instead of a range
mpl-animator plot.py --var n --values "5,10,50,200,1000"

# Save the generated .py script too (useful for --sequential re-runs)
mpl-animator plot.py --var t --range "0,1" --save-script
```

The GIF lands in your current directory as `<script>_animated.gif`. Pass `--out` for a custom path.

## What it does

1. Parses your script's AST to find everything that depends on the animated variable
2. Splits code into static (runs once) and dynamic (recalculated per frame)
3. Renders frames in parallel across all CPU cores, then stitches to GIF or MP4

No annotations. No rewriting. Point and shoot.

## Examples

### Wave & spectrum

```python
# wave_static.py
import numpy as np
import matplotlib.pyplot as plt

f   = 10.0                          # <- animate this
t   = np.linspace(0, 1, 1000)
y   = np.sin(2*np.pi*f*t) + 0.4*np.sin(2*np.pi*2*f*t)
freqs    = np.fft.rfftfreq(len(t), d=t[1]-t[0])
spectrum = np.abs(np.fft.rfft(y))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6))
ax1.plot(t, y, 'royalblue', lw=1.5);  ax1.set_title(f"Signal  f = {f:.1f} Hz")
ax2.plot(freqs, spectrum, 'tomato', lw=1.5);  ax2.set_title("Frequency Spectrum")
plt.tight_layout(); plt.show()
```

```bash
mpl-animator examples/wave_static.py --var f --range "3,60" --frames 60 --fps 20
```

![wave animation](https://raw.githubusercontent.com/BasemRajjoub/mpl_animator/main/examples/wave_static_animated.gif)

---

### 3D Lissajous

[examples/lissajous_3d_static.py](examples/lissajous_3d_static.py)

```bash
mpl-animator examples/lissajous_3d_static.py --var a --range "1,6" --frames 80 --fps 20
```

![lissajous animation](https://raw.githubusercontent.com/BasemRajjoub/mpl_animator/main/examples/lissajous_3d_static_animated.gif)

---

### Cinematic orbit (3 variables)

[examples/orbit_static.py](examples/orbit_static.py) — camera orbits 360°, object spins a full turn, camera cranes up:

```bash
mpl-animator examples/orbit_static.py \
    --var azim spin elev \
    --range "0,360" "0,6.28318" "20,40" \
    --frames 90 --fps 25 --dpi 120 --ping-pong
```

![orbit animation](https://raw.githubusercontent.com/BasemRajjoub/mpl_animator/main/examples/orbit_static_animated.gif)

---

## Library API

```python
from mpl_animator import animate

src = open("my_plot.py").read()

# Returns the generated animation script as a string
code = animate(src, var="t", range_str="0,6.28", frames=60, fps=25)

# Multi-variable
code = animate(src, var=["azim", "spin"], range_str=["0,360", "0,6.28"], frames=90, ping_pong=True)

# Explicit values
code = animate(src, var="n", values="5,10,50,200,1000")
```

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--var` | `t` | Variable(s) to animate |
| `--range` | `0,1` | `start,end` per variable (math expressions ok) |
| `--values` | — | Explicit values instead of a range |
| `--frames` | 120 | Frame count |
| `--fps` | 25 | Frames per second |
| `--format` | `gif` | `gif` or `mp4` |
| `--out` | auto | Output filename |
| `--dpi` | 100 | Render DPI |
| `--workers` | auto | Parallel workers (0 = cpu_count) |
| `--loop` | 0 | GIF loops (0 = forever) |
| `--ping-pong` | off | Play forward then reversed |
| `--reverse` | off | Sweep end → start |
| `--save-script` | off | Save the generated `.py` alongside the GIF |

## Tests

```bash
pytest tests/test_animator.py   # 208 unit + integration tests
pytest tests/test_gallery.py    # 276 tests across all matplotlib gallery examples
pytest tests/ -m slow           # tests that generate actual GIF/MP4 files
```

Validated against all 510 official matplotlib gallery examples.

## Known limitations

- **Accumulating scripts** — re-runs from scratch each frame; not suitable for scripts that build state iteratively
- **`__main__` guards** — variables assigned inside `if __name__ == "__main__":` may not partition correctly
- **Variable domain safety** — no awareness of valid domains; bad ranges cause per-frame errors (skipped gracefully)

---

Author: [Basem Rajjoub](https://basemrajjoub.com) · Built with [Claude Code](https://claude.ai/claude-code)
