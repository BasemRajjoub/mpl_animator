# mpl_animator.py

Turn any static matplotlib script into an animated GIF by sweeping a variable.

## Install

```bash
pip install matplotlib numpy Pillow
```

## Usage

```bash
# Basic: animate variable `f` from 3 to 60
python mpl_animator.py wave_static.py --var f --range "3,60"

# Math expressions in range, custom frame count and FPS
python mpl_animator.py plot.py --var t --range "0,2*pi" --frames 60 --fps 30

# Control output quality and parallelism
python mpl_animator.py plot.py --var alpha --range "0,1" --dpi 150 --workers 8

# Custom output filename
python mpl_animator.py plot.py --var t --range "0,1" --out my_animation.gif
```

This generates a `<script>_animated.py` file. Run it to produce the GIF:

```bash
python wave_static_animated.py              # parallel (default)
python wave_static_animated.py --sequential # single-threaded fallback
```

## How it works

1. Parses your script's AST to find which variables depend on the animated one
2. Splits code into **static** (run once) and **dynamic** (recalculated per frame)
3. Generates a new script with `FuncAnimation` (sequential) and `multiprocessing` (parallel) renderers

## Library usage

```python
from mpl_animator import animate

src = open("my_plot.py").read()
animated_code = animate(src, var="t", range_str="0,6.28", frames=60, fps=25)
open("my_plot_animated.py", "w").write(animated_code)
```

## Supported plot types

2D (plot, scatter, bar, hist, contour, imshow, ...), 3D (plot_surface, scatter3D, ...), polar, and anything else matplotlib draws.

## Tests

```bash
pytest test_animator.py -v              # fast tests (111)
pytest test_animator.py -v -m slow      # slow tests that generate actual GIFs
```

---

Built with the assistance of [Claude Code](https://claude.ai/claude-code)
