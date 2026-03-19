"""
animator.py — convert ANY static matplotlib script to an animation.
Uses only matplotlib + multiprocessing + Pillow. No celluloid needed.

Usage:
    python animator.py plot.py --var f --range "3,60"
    python animator.py plot.py --var t --range "0,2*pi" --frames 120 --fps 30
    python animator.py plot.py --var alpha --range "0,1" --workers 8
    (run generated script with --sequential to skip parallel)
"""

import ast, argparse, math
from pathlib import Path

# ── CLI ──────────────────────────────────────────────────────────
p = argparse.ArgumentParser()
p.add_argument("script")
p.add_argument("--var",     default="t",   help="Variable to animate")
p.add_argument("--range",   default="0,1", help="start,end (math ok: 2*pi)")
p.add_argument("--frames",  default=120,   type=int)
p.add_argument("--fps",     default=25,    type=int)
p.add_argument("--workers", default=0,     type=int, help="0=auto=cpu_count")
p.add_argument("--dpi",     default=100,   type=int)
p.add_argument("--out",     default=None)
args = p.parse_args()

src   = Path(args.script).read_text()
lines = src.splitlines()

def parse_val(s):
    return eval(s.strip(), {"pi": math.pi, "e": math.e,
                             "sqrt": math.sqrt, "tau": 2*math.pi})

start_val, end_val = [parse_val(x) for x in args.range.split(",")]
VAR      = args.var
FRAMES   = args.frames
FPS      = args.fps
WORKERS  = args.workers
DPI      = args.dpi
STEP     = (end_val - start_val) / FRAMES
OUT_GIF  = args.out or (Path(args.script).stem + "_animated.gif")
INTERVAL = 1000 // FPS

# ════════════════════════════════════════════════════════════════
# 1. AST SCAN
# ════════════════════════════════════════════════════════════════
tree = ast.parse(src)

fig_line   = None
show_line  = None
first_plot = None
last_plot  = None

PLOT_METHODS = {
    "plot","scatter","bar","barh","fill","fill_between","fill_betweenx",
    "imshow","pcolormesh","pcolor","contour","contourf","tricontour","tricontourf",
    "step","stairs","stem","eventplot","hexbin","hist","hist2d",
    "pie","boxplot","violinplot","errorbar","streamplot",
    "axhline","axvline","axhspan","axvspan","axline",
    "set_title","set_xlabel","set_ylabel","set_zlabel",
    "set_xlim","set_ylim","set_zlim","set_xticks","set_yticks",
    "set_xticklabels","set_yticklabels",
    "grid","legend","colorbar","text","annotate",
    "semilogx","semilogy","loglog","tight_layout","suptitle",
    "plot_surface","plot_wireframe","plot_trisurf","scatter3D",
    "bar3d","contour3D","contourf3D","view_init",
}

line_assigns: dict[int, set[str]] = {}

for node in ast.walk(tree):
    ln = getattr(node, "lineno", None)
    if ln is None:
        continue
    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
        call = node.value
        if isinstance(call.func, ast.Attribute):
            attr = call.func.attr
            if attr == "show":
                show_line = ln
            if attr in PLOT_METHODS:
                first_plot = first_plot or ln
                last_plot  = ln
    if isinstance(node, ast.Assign):
        assigned = set()
        for target in node.targets:
            for n in ast.walk(target):
                if isinstance(n, ast.Name):
                    assigned.add(n.id)
        if assigned:
            line_assigns[ln] = assigned
        for child in ast.walk(node.value):
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute):
                if child.func.attr in ("subplots","figure","subplot","subplot_mosaic"):
                    if fig_line is None or ln < fig_line:
                        fig_line = ln

# ════════════════════════════════════════════════════════════════
# 2. TRANSITIVE DEPENDENCY TRACKING
# ════════════════════════════════════════════════════════════════
rhs_uses: dict[str, set[str]] = {}
for node in ast.walk(tree):
    if isinstance(node, ast.Assign):
        used = {n.id for n in ast.walk(node.value) if isinstance(n, ast.Name)}
        for target in node.targets:
            for n in ast.walk(target):
                if isinstance(n, ast.Name):
                    rhs_uses.setdefault(n.id, set()).update(used)

def all_dependents(root):
    found, queue = set(), {root}
    while queue:
        cur = queue.pop()
        for k, v in rhs_uses.items():
            if cur in v and k not in found:
                found.add(k); queue.add(k)
    return found

dep_vars = all_dependents(VAR)

# ════════════════════════════════════════════════════════════════
# 3. PARTITION LINES into static / dynamic / plot
# ════════════════════════════════════════════════════════════════
setup_end  = fig_line or first_plot or show_line or len(lines)
plot_end   = (show_line - 1) if show_line else (last_plot or len(lines))

setup_lines = lines[:setup_end]
plot_lines  = lines[setup_end:plot_end]

static_lines:  list[str] = []   # outside update()
dynamic_lines: list[str] = []   # inside update(), before drawing

for i, line in enumerate(setup_lines):
    assigned = line_assigns.get(i + 1, set())
    if assigned & (dep_vars | {VAR}):
        if assigned != {VAR}:   # skip original `var = value`
            dynamic_lines.append(line)
    else:
        static_lines.append(line)

# ════════════════════════════════════════════════════════════════
# 4. AXES VARS → .clear() snippet (UN-indented, applied at gen time)
# ════════════════════════════════════════════════════════════════
ax_vars: list[str] = []
for node in ast.walk(tree):
    if isinstance(node, ast.Assign):
        for child in ast.walk(node.value):
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute):
                if child.func.attr in ("subplots","subplot","subplot_mosaic"):
                    for target in node.targets:
                        for elt in ast.walk(target):
                            if isinstance(elt, ast.Name) and elt.id != "fig":
                                if elt.id not in ax_vars:
                                    ax_vars.append(elt.id)

# Returns UNINDENTED clear lines
def clear_lines(ax_vars):
    if not ax_vars:
        return ["ax.clear()"]
    out = []
    for v in ax_vars:
        out += [
            f"_axs = list({v}) if hasattr({v},'__iter__') else [{v}]",
            f"[_a.clear() for _a in _axs]",
        ]
    return out

CLEAR = clear_lines(ax_vars)

# ════════════════════════════════════════════════════════════════
# 5. INJECT matplotlib.use("Agg") before first mpl import
# ════════════════════════════════════════════════════════════════
final_static: list[str] = []
agg_done = False
for line in static_lines:
    if "matplotlib.use" in line:
        continue
    if not agg_done and ("import matplotlib" in line or "pyplot" in line):
        final_static.append("import matplotlib; matplotlib.use('Agg')")
        agg_done = True
    final_static.append(line)
if not agg_done:
    final_static.insert(0, "import matplotlib; matplotlib.use('Agg')")

# ════════════════════════════════════════════════════════════════
# 6. HELPER: indent a list of strings by N spaces
# ════════════════════════════════════════════════════════════════
def ind(lst, n=4):
    pad = " " * n
    if isinstance(lst, str):
        lst = lst.splitlines()
    return "\n".join(pad + l for l in lst)

# ════════════════════════════════════════════════════════════════
# 7. BUILD SECTIONS
# ════════════════════════════════════════════════════════════════
S = "\n"   # newline shortcut

static_block = S.join(final_static)

# update() body
update_body = (
    f"    {VAR} = {start_val!r} + _frame * {STEP!r}\n"
    + (ind(dynamic_lines) + "\n" if dynamic_lines else "")
    + ind(CLEAR) + "\n"
    + ind(plot_lines)
)

# worker body: re-create fig inside worker process, no ax.clear() needed
worker_body = (
    "    import matplotlib; matplotlib.use('Agg')\n"
    "    import matplotlib.pyplot as plt\n"
    "    import numpy as np\n"
    + ind(final_static, 4) + "\n"      # re-runs setup: creates fig & axes
    + f"    {VAR} = {start_val!r} + _frame * {STEP!r}\n"
    + (ind(dynamic_lines, 4) + "\n" if dynamic_lines else "")
    + ind(plot_lines, 4)
)

# ════════════════════════════════════════════════════════════════
# 8. ASSEMBLE using <<<PLACEHOLDER>>> (avoids f-string brace clashes)
# ════════════════════════════════════════════════════════════════
TEMPLATE = '''\
"""Auto-generated by animator.py from <<<SOURCE>>>
   <<<VAR>>> sweeps <<<START>>> -> <<<END>>> over <<<FRAMES>>> frames @ <<<FPS>>>fps
"""
<<<STATIC>>>
import matplotlib.animation as animation
import multiprocessing, os, tempfile, shutil, time
from PIL import Image

# ── frame update ─────────────────────────────────────────────────
def update(_frame):
<<<UPDATE_BODY>>>

# ── sequential (FuncAnimation) ───────────────────────────────────
def render_sequential():
    ani = animation.FuncAnimation(
        fig, update, frames=<<<FRAMES>>>, interval=<<<INTERVAL>>>, blit=False)
    ani.save("<<<OUTGIF>>>", writer="pillow", fps=<<<FPS>>>)
    print("  Saved ->", "<<<OUTGIF>>>")

# ── parallel worker (one PNG per frame) ──────────────────────────
def _render_one(job):
    _frame, _tmpdir = job
<<<WORKER_BODY>>>
    _path = os.path.join(_tmpdir, f"frame_{_frame:05d}.png")
    plt.savefig(_path, dpi=<<<DPI>>>, bbox_inches="tight")
    plt.close("all")
    return _path

def render_parallel(n_workers):
    tmpdir = tempfile.mkdtemp(prefix="anim_")
    chunk  = max(1, <<<FRAMES>>> // (n_workers * 4))
    print(f"  Rendering <<<FRAMES>>> frames on {n_workers} workers...")
    try:
        jobs  = [(_i, tmpdir) for _i in range(<<<FRAMES>>>)]
        with multiprocessing.Pool(n_workers) as pool:
            paths = list(pool.imap(_render_one, jobs, chunksize=chunk))
        paths.sort()
        print(f"  Stitching {len(paths)} frames...")
        imgs = [Image.open(q).convert("RGBA") for q in paths]
        imgs[0].save("<<<OUTGIF>>>", save_all=True, append_images=imgs[1:],
                     loop=0, duration=<<<INTERVAL>>>, optimize=False)
        print("  Saved ->", "<<<OUTGIF>>>")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

# ── main ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    n_workers = <<<WORKERS>>> or multiprocessing.cpu_count()
    t0 = time.perf_counter()
    if "--sequential" in sys.argv:
        print("Sequential mode...")
        render_sequential()
    else:
        print(f"Parallel mode ({n_workers} workers)...")
        try:
            render_parallel(n_workers)
        except Exception as err:
            print(f"  Parallel failed ({err}), falling back to sequential...")
            render_sequential()
    print(f"  Total: {time.perf_counter()-t0:.1f}s")
'''

result = (TEMPLATE
    .replace("<<<SOURCE>>>",      args.script)
    .replace("<<<STATIC>>>",      static_block)
    .replace("<<<VAR>>>",         VAR)
    .replace("<<<START>>>",       repr(start_val))
    .replace("<<<END>>>",         repr(end_val))
    .replace("<<<STEP>>>",        repr(STEP))
    .replace("<<<UPDATE_BODY>>>", update_body)
    .replace("<<<WORKER_BODY>>>", worker_body)
    .replace("<<<FRAMES>>>",      str(FRAMES))
    .replace("<<<INTERVAL>>>",    str(INTERVAL))
    .replace("<<<FPS>>>",         str(FPS))
    .replace("<<<OUTGIF>>>",      OUT_GIF)
    .replace("<<<DPI>>>",         str(DPI))
    .replace("<<<WORKERS>>>",     str(WORKERS))
)

out_script = Path(args.script).stem + "_animated.py"
Path(out_script).write_text(result)
print(f"✓  Written  -> {out_script}")
print(f"   Variable : {VAR}  {start_val} -> {end_val}  | {FRAMES} frames @ {FPS}fps")
print(f"   Workers  : {WORKERS or 'auto (cpu_count)'}")
print(f"   Run      : python {out_script}")
print(f"   Seq only : python {out_script} --sequential")
