# mpl-animator

Turn any static matplotlib script into an animated GIF or MP4 by sweeping a variable - no rewriting required.

## Features

- **Zero boilerplate** - point it at an existing script, it figures out the rest
- **Multi-variable animation** - sweep multiple variables simultaneously with `--var azim spin elev --range "0,360" "0,6.28" "20,40"`; all variables share the same frame clock, each advances independently
- **AST-based dependency tracking** - automatically identifies which variables and calculations need to update each frame, across all animated variables
- **Parallel rendering** - renders frames across all CPU cores via `multiprocessing`, falls back to sequential automatically
- **GIF and MP4 output** - export as animated GIF (Pillow) or MP4 (ffmpeg); just add `--format mp4`
- **Loop control** - `--loop 0` (forever, default), `--loop 1` (play once), `--loop N`
- **Ping-pong** - `--ping-pong` plays forward then reversed for seamless looping GIFs
- **Reverse** - `--reverse` sweeps each range end → start
- **Axis rotation** - animate `azim` with `ax.view_init` to rotate 3D plots; animate `angle` with `ax.set_theta_offset` to spin polar plots
- **Math expressions in ranges** - `--range "0,2*pi"` just works
- **2D, 3D, and polar plots** - handles `plot_surface`, `scatter3D`, subplots, polar axes, and more
- **Single-file, standalone** - `mpl_animator.py` can be dropped into any project with no install required; only depends on `matplotlib`, `numpy`, and `Pillow` (no exotic dependencies)
- **Library API** - importable as a Python module for use in notebooks or pipelines

## Install

```bash
pip install mpl-animator
```

Or just copy the file - no install needed:

```bash
# copy mpl_animator.py into your project, then use it directly
python mpl_animator.py my_plot.py --var t --range "0,1"
```

## Usage

If installed via pip, use the `mpl-animator` command. If using the file directly, replace `mpl-animator` with `python mpl_animator.py` - everything else is identical.

```bash
# Basic: animate variable `f` from 3 to 60 (outputs GIF by default)
mpl-animator wave_static.py --var f --range "3,60"

# Export as MP4 instead of GIF (requires ffmpeg)
mpl-animator plot.py --var t --range "0,2*pi" --format mp4

# Math expressions in range, custom frame count and FPS
mpl-animator plot.py --var t --range "0,2*pi" --frames 60 --fps 30

# Rotate a 3D plot by animating the camera azimuth
mpl-animator my_3d_plot.py --var azim --range "0,360" --frames 72 --fps 20

# Multi-variable: camera orbits (azim) while the object spins (spin) and camera rises (elev)
mpl-animator orbit_static.py --var azim spin elev --range "0,360" "0,6.28" "20,40" --frames 90 --fps 25 --ping-pong

# Ping-pong loop (plays forward then reversed — great for seamless GIFs)
mpl-animator plot.py --var t --range "0,1" --frames 60 --ping-pong

# Control output quality and parallelism
mpl-animator plot.py --var alpha --range "0,1" --dpi 150 --workers 8

# Custom output filename
mpl-animator plot.py --var t --range "0,1" --out my_animation.gif
```

This generates a `<script>_animated.py` file. Run it to produce the output:

```bash
python wave_static_animated.py              # parallel (default)
python wave_static_animated.py --sequential # single-threaded fallback
```

## Examples

### Example 1 - Wave & spectrum (2D)

`wave_static.py` plots a signal and its frequency spectrum for a fixed frequency `f`:

```python
# wave_static.py  (key lines)
f   = 10.0                          # <- variable to animate
t   = np.linspace(0, 1, 1000)
y   = np.sin(2*np.pi*f*t) + 0.4*np.sin(2*np.pi*2*f*t)

freqs    = np.fft.rfftfreq(len(t), d=t[1]-t[0])
spectrum = np.abs(np.fft.rfft(y))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6))
ax1.plot(t, y, 'royalblue', lw=1.5)
ax2.plot(freqs, spectrum, 'tomato', lw=1.5)
plt.show()
```

Animate `f` from 3 Hz to 60 Hz:

```bash
python mpl_animator.py examples/wave_static.py --var f --range "3,60" --frames 60 --fps 20
python wave_static_animated.py
```

The animator detects that `y`, `spectrum` depend on `f`, moves them into the per-frame `update()`, and keeps the figure/axes creation static - so only the data redraws each frame.

![wave animation](https://raw.githubusercontent.com/BasemRajjoub/mpl_animator/main/examples/wave_static_animated.gif)

---

### Example 2 - 3D Lissajous curve

`lissajous_3d_static.py` draws a 3D Lissajous figure for fixed frequency ratio `a`:

```python
# lissajous_3d_static.py  (key lines)
a  = 3.0          # <- variable to animate
b  = 2.0
c  = 1.0

t = np.linspace(0, 2 * np.pi, 1000)
x = np.sin(a * t + delta)
y = np.sin(b * t)
z = np.sin(c * t)

fig = plt.figure(figsize=(8, 6))
ax  = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, c=colors, s=2, alpha=0.8)
ax.set_title(f"3D Lissajous  a={a:.1f}, b={b:.1f}, c={c:.1f}")
plt.show()
```

Animate `a` from 1 to 6, sweeping through different curve topologies:

```bash
python mpl_animator.py examples/lissajous_3d_static.py --var a --range "1,6" --frames 80 --fps 20
python lissajous_3d_static_animated.py
```

For 3D plots the animator calls `fig.clear()` and recreates the axes each frame (required to preserve the `projection='3d'` state), then re-runs all drawing commands with the new value of `a`.

![lissajous animation](https://raw.githubusercontent.com/BasemRajjoub/mpl_animator/main/examples/lissajous_3d_static_animated.gif)

---

---

### Example 3 - Cinematic 3D orbit (multi-variable)

`orbit_static.py` draws a torus knot for fixed camera position and object rotation:

```python
# orbit_static.py  (key lines)
azim = 45.0          # camera azimuth in degrees      <- animated
spin = 0.0           # object self-rotation (radians) <- animated
elev = 20.0          # camera elevation in degrees    <- animated

# Object geometry: rotate the knot around its own Z axis
cx_rot = cx * np.cos(spin) - cy * np.sin(spin)
cy_rot = cx * np.sin(spin) + cy * np.cos(spin)

fig = plt.figure(figsize=(7, 7))
ax  = fig.add_subplot(111, projection='3d')
ax.plot_surface(xs, ys, zs, cmap='plasma', alpha=0.92)
ax.view_init(elev=elev, azim=azim)   # <- camera controlled by both azim and elev
plt.show()
```

All three variables sweep simultaneously from the same frame clock — the camera orbits a full 360°, the object spins one full turn, and the camera gently cranes upward. `--ping-pong` plays the sequence forward then reversed for a seamless loop:

```bash
python mpl_animator.py examples/orbit_static.py \
    --var azim spin elev \
    --range "0,360" "0,6.28318" "20,40" \
    --frames 90 --fps 25 --dpi 120 --ping-pong
python orbit_static_animated.py --sequential
```

The animator identifies that `cx_rot`, `cy_rot`, and the tube surface `xs`/`ys`/`zs` all depend on `spin`, and that `ax.view_init` depends on `azim` and `elev` — all three dependency chains are tracked and placed in `update()` automatically.

![orbit animation](https://raw.githubusercontent.com/BasemRajjoub/mpl_animator/main/examples/orbit_static_animated.gif)

---

## How it works

1. Parses your script's AST to find which variables depend on the animated one
2. Splits code into **static** (run once) and **dynamic** (recalculated per frame)
3. Generates a new script with `FuncAnimation` (sequential 2D GIF), PNG+stitch (sequential MP4 or 3D), and `multiprocessing` (parallel) renderers

## Library usage

```python
from mpl_animator import animate

src = open("my_plot.py").read()

# Single variable
animated_code = animate(src, var="t", range_str="0,6.28", frames=60, fps=25)
open("my_plot_animated.py", "w").write(animated_code)

# Multiple variables — pass lists of equal length
animated_code = animate(
    src,
    var=["azim", "spin", "elev"],
    range_str=["0,360", "0,6.28", "20,40"],
    frames=90, fps=25, ping_pong=True,
)

# Export as MP4
animated_code = animate(src, var="t", range_str="0,6.28", fmt="mp4")

# Ping-pong loop (seamless forward+reverse)
animated_code = animate(src, var="t", range_str="0,1", ping_pong=True, loop=0)
```

## Supported plot types

2D (plot, scatter, bar, hist, contour, imshow, ...), 3D (plot_surface, scatter3D, ...), polar, and anything else matplotlib draws.

## Tests

```bash
pytest tests/ -v              # fast tests
pytest tests/ -v -m slow      # slow tests that generate actual GIFs/MP4s
```

---

Author: [Basem Rajjoub](https://basemrajjoub.com)

Built with the assistance of [Claude Code](https://claude.ai/claude-code)
