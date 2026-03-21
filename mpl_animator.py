"""
mpl_animator.py -- convert ANY static matplotlib script to an animation.
Author: Basem Rajjoub (https://basemrajjoub.com)
Version: 0.1.5
Uses only matplotlib + multiprocessing + Pillow (GIF) / ffmpeg (MP4).

Usage (CLI):
    python mpl_animator.py plot.py --var f --range "3,60"
    python mpl_animator.py plot.py --var t --range "0,2*pi" --frames 120 --fps 30
    python mpl_animator.py plot.py --var alpha --range "0,1" --format mp4
    python mpl_animator.py plot.py --var t --range "0,1" --ping-pong
    python mpl_animator.py plot.py --var t --range "0,1" --reverse
    python mpl_animator.py plot.py --var t --range "0,1" --loop 3
    python mpl_animator.py plot.py --var t alpha --range "0,6.28" "0,1"
    (run generated script with --sequential to skip parallel)

Usage (library):
    from mpl_animator import animate
    code = animate(open("plot.py", encoding="utf-8").read(), var="f", range_str="3,60", fmt="mp4")
"""

import ast
import math
import operator
import shutil
import warnings
from dataclasses import dataclass, field
from pathlib import Path

# -- Method categories --
DRAW_METHODS = {
    "plot", "scatter", "bar", "barh", "fill", "fill_between", "fill_betweenx",
    "imshow", "pcolormesh", "pcolor", "contour", "contourf",
    "tricontour", "tricontourf",
    "step", "stairs", "stem", "eventplot", "hexbin", "hist", "hist2d",
    "pie", "boxplot", "violinplot", "errorbar", "streamplot",
    "axhline", "axvline", "axhspan", "axvspan", "axline",
    "semilogx", "semilogy", "loglog",
    "plot_surface", "plot_wireframe", "plot_trisurf", "scatter3D",
    "bar3d", "contour3D", "contourf3D",
    "quiver", "barbs",
    "add_patch", "add_collection",
}

CONFIG_METHODS = {
    "set_title", "set_xlabel", "set_ylabel", "set_zlabel",
    "set_xlim", "set_ylim", "set_zlim", "set_xticks", "set_yticks",
    "set_xticklabels", "set_yticklabels",
    "grid", "legend", "colorbar", "text", "annotate",
    "tight_layout", "suptitle", "view_init",
    "set_proj_type", "set_theta_zero_location", "set_rlabel_position",
    "set_theta_offset", "set_rorigin", "set_rlim",
    "tick_params", "set_aspect", "set_facecolor",
    "set_xscale", "set_yscale",
    "subplots_adjust",
}

ALL_PLOT_METHODS = DRAW_METHODS | CONFIG_METHODS

FIGURE_CREATORS = {"subplots", "figure", "subplot", "subplot_mosaic",
                   "add_subplot", "add_axes"}


# -- Safe expression evaluator --
_SAFE_NAMES = {
    "pi": math.pi, "e": math.e, "tau": math.tau,
    "inf": math.inf, "nan": math.nan,
}
_SAFE_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
    ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos,
}
_SAFE_FUNCS = {"sqrt": math.sqrt, "abs": abs, "sin": math.sin,
               "cos": math.cos, "log": math.log, "exp": math.exp}


def _safe_eval_node(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.Name):
        if node.id in _SAFE_NAMES:
            return _SAFE_NAMES[node.id]
        raise ValueError(f"Unknown name: {node.id!r}")
    if isinstance(node, ast.BinOp):
        op_fn = _SAFE_OPS.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"Unsupported binary op: {type(node.op).__name__}")
        return op_fn(_safe_eval_node(node.left), _safe_eval_node(node.right))
    if isinstance(node, ast.UnaryOp):
        op_fn = _SAFE_OPS.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"Unsupported unary op: {type(node.op).__name__}")
        return op_fn(_safe_eval_node(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        fn = _SAFE_FUNCS.get(node.func.id)
        if fn is None:
            raise ValueError(f"Unsupported function: {node.func.id!r}")
        args = [_safe_eval_node(a) for a in node.args]
        return fn(*args)
    raise ValueError(f"Unsupported expression: {ast.dump(node)}")


def parse_val(s):
    """Parse a numeric expression string safely (no arbitrary code execution)."""
    s = s.strip()
    tree = ast.parse(s, mode="eval")
    return _safe_eval_node(tree.body)


# -- AST scan result --
@dataclass
class AxesInfo:
    var_name: str
    is_3d: bool = False
    subplot_spec: str = ""
    creation_source: str = ""


@dataclass
class ASTInfo:
    fig_node: ast.stmt | None = None
    show_node: ast.stmt | None = None
    first_draw_node: ast.stmt | None = None
    last_draw_node: ast.stmt | None = None
    node_assigns: dict = field(default_factory=dict)
    node_has_draw: dict = field(default_factory=dict)
    node_has_config: dict = field(default_factory=dict)
    node_is_show: dict = field(default_factory=dict)
    node_is_fig_creation: dict = field(default_factory=dict)
    ax_info: list = field(default_factory=list)
    all_names: set = field(default_factory=set)


def _get_assigned_names(target_node):
    names = set()
    for n in ast.walk(target_node):
        if isinstance(n, ast.Name):
            names.add(n.id)
    return names


def _node_uses_names(node):
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def _check_3d_projection(call_node):
    for kw in call_node.keywords:
        if kw.arg == "projection":
            if isinstance(kw.value, ast.Constant) and kw.value.value == "3d":
                return True
        if kw.arg == "subplot_kw":
            if isinstance(kw.value, ast.Dict):
                for k, v in zip(kw.value.keys, kw.value.values):
                    if (isinstance(k, ast.Constant) and k.value == "projection"
                            and isinstance(v, ast.Constant) and v.value == "3d"):
                        return True
    return False


def _extract_subplot_spec(call_node):
    parts = []
    for arg in call_node.args:
        if isinstance(arg, ast.Constant):
            parts.append(repr(arg.value))
    return ", ".join(parts)


def scan_ast(src):
    """Phase 1: Parse source and extract structural information."""
    tree = ast.parse(src)
    info = ASTInfo()

    info.all_names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}

    for stmt in tree.body:
        stmt_id = id(stmt)
        info.node_assigns[stmt_id] = set()
        info.node_has_draw[stmt_id] = False
        info.node_has_config[stmt_id] = False
        info.node_is_show[stmt_id] = False
        info.node_is_fig_creation[stmt_id] = False

        for node in ast.walk(stmt):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    info.node_assigns[stmt_id] |= _get_assigned_names(target)
            elif isinstance(node, ast.AugAssign):
                info.node_assigns[stmt_id] |= _get_assigned_names(node.target)
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                info.node_assigns[stmt_id] |= _get_assigned_names(node.target)

            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                attr = node.func.attr

                if attr == "show":
                    info.node_is_show[stmt_id] = True
                    if info.show_node is None:
                        info.show_node = stmt

                if attr in DRAW_METHODS:
                    info.node_has_draw[stmt_id] = True
                    if info.first_draw_node is None:
                        info.first_draw_node = stmt
                    info.last_draw_node = stmt

                if attr in CONFIG_METHODS:
                    info.node_has_config[stmt_id] = True

                if attr in FIGURE_CREATORS:
                    info.node_is_fig_creation[stmt_id] = True
                    if info.fig_node is None:
                        info.fig_node = stmt

                    is_3d = _check_3d_projection(node)

                    # Only register axes targets for axes-creating methods, not plt.figure()
                    _AXES_CREATORS = {"subplots", "subplot", "subplot_mosaic",
                                      "add_subplot", "add_axes"}
                    if attr in _AXES_CREATORS and isinstance(stmt, ast.Assign):
                        for target in stmt.targets:
                            for elt in ast.walk(target):
                                if isinstance(elt, ast.Name) and elt.id != "fig":
                                    ax = AxesInfo(
                                        var_name=elt.id,
                                        is_3d=is_3d,
                                        creation_source=ast.get_source_segment(src, stmt) or "",
                                    )
                                    if attr == "add_subplot":
                                        ax.subplot_spec = _extract_subplot_spec(node)
                                    info.ax_info.append(ax)

    return tree, info


# -- Dependency tracking --
def build_deps(tree, var):
    """Phase 2: Build transitive dependency graph for `var`.

    var may be a single variable name (str) or a list of root variable names.
    Returns the union of all transitively dependent variable names.
    """
    roots = [var] if isinstance(var, str) else list(var)
    rhs_uses: dict[str, set[str]] = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            used = _node_uses_names(node.value)
            for target in node.targets:
                for n in ast.walk(target):
                    if isinstance(n, ast.Name):
                        rhs_uses.setdefault(n.id, set()).update(used)
        elif isinstance(node, ast.AugAssign):
            used = _node_uses_names(node.value)
            if isinstance(node.target, ast.Name):
                name = node.target.id
                rhs_uses.setdefault(name, set()).update(used | {name})
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            used = _node_uses_names(node.value)
            if isinstance(node.target, ast.Name):
                rhs_uses.setdefault(node.target.id, set()).update(used)

    found, queue = set(), set(roots)
    while queue:
        cur = queue.pop()
        for k, v in rhs_uses.items():
            if cur in v and k not in found:
                found.add(k)
                queue.add(k)
    return found


# -- Statement partitioning --
def _get_stmt_source(src, stmt):
    seg = ast.get_source_segment(src, stmt)
    if seg is not None:
        return seg
    lines = src.splitlines()
    start = stmt.lineno - 1
    end = getattr(stmt, "end_lineno", stmt.lineno)
    return "\n".join(lines[start:end])


def partition(src, tree, info, dep_vars, var):
    """Phase 3: Partition top-level statements into static/dynamic/plot/show.

    var may be a single variable name (str) or a list of animated variable names.
    """
    vars_set = {var} if isinstance(var, str) else set(var)
    static_stmts = []
    dynamic_stmts = []
    plot_stmts = []

    fig_found = False
    # Variables assigned by plot/config statements (e.g. CS = ax.contour(...))
    # — any later statement using these must also go to plot_stmts
    plot_assigned: set = set()

    for stmt in tree.body:
        sid = id(stmt)
        text = _get_stmt_source(src, stmt)

        if info.node_is_show[sid]:
            continue

        # Function/class definitions always go to static — their bodies must
        # not influence classification (ast.walk would otherwise find draw/
        # config calls inside them and misclassify the def statement).
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            static_stmts.append(text)
            continue

        assigned = info.node_assigns[sid]
        has_draw = info.node_has_draw[sid]
        has_config = info.node_has_config[sid]
        is_fig = info.node_is_fig_creation[sid]

        if is_fig:
            fig_found = True

        stmt_names = _node_uses_names(stmt)

        if is_fig:
            static_stmts.append(text)
            continue

        # Skip pure assignment of animated variable(s) — we generate our own
        if assigned and assigned <= vars_set and not has_draw and not has_config:
            continue

        depends_on_var = bool(assigned & (dep_vars | vars_set)) or bool(stmt_names & (vars_set | dep_vars))
        uses_plot_assigned = bool(stmt_names & plot_assigned)

        if not fig_found:
            if depends_on_var:
                dynamic_stmts.append(text)
            else:
                static_stmts.append(text)
        else:
            if has_draw or has_config or uses_plot_assigned:
                plot_stmts.append(text)
                plot_assigned.update(assigned)  # track what this plot stmt assigned
            elif depends_on_var:
                dynamic_stmts.append(text)
            else:
                static_stmts.append(text)

    return static_stmts, dynamic_stmts, plot_stmts


# -- Axes clearing code generation --
def _gen_clear_lines(ax_info_list):
    if not ax_info_list:
        return ["plt.gca().clear()"]

    out = []
    for ax in ax_info_list:
        v = ax.var_name
        if not ax.is_3d:
            out += [
                f"_axs = list({v}.values()) if hasattr({v},'values') else "
                f"(list({v}.flat) if hasattr({v},'flat') else "
                f"(list({v}) if hasattr({v},'__iter__') else [{v}]))",
                "[_a.clear() for _a in _axs]",
            ]

    return out if out else ["plt.gca().clear()"]


# -- Agg backend injection --
def _inject_agg(static_stmts):
    result = []
    agg_done = False
    for stmt in static_stmts:
        if "matplotlib.use" in stmt:
            continue
        if not agg_done and ("import matplotlib" in stmt or "pyplot" in stmt):
            result.append("import matplotlib; matplotlib.use('Agg')")
            agg_done = True
        result.append(stmt)
    if not agg_done:
        result.insert(0, "import matplotlib; matplotlib.use('Agg')")
    return result


# -- Indentation helper --
def _ind(lst, n=4):
    pad = " " * n
    if isinstance(lst, str):
        lines = lst.splitlines()
    else:
        # Flatten list items that may contain embedded newlines
        lines = []
        for item in lst:
            lines.extend(item.splitlines())
    return "\n".join(pad + line for line in lines)


# -- Code generation template --
TEMPLATE = '''\
"""Auto-generated by mpl-animator v0.1.5 from <<<SOURCE>>>
   <<<VAR_SUMMARY>>> over <<<FRAMES>>> frames @ <<<FPS>>>fps
"""
<<<STATIC>>>
import multiprocessing, os, subprocess, tempfile, shutil, time

# -- frame update ----------------------------------------------------
def update(_frame):
<<<UPDATE_BODY>>>

# -- output helpers --------------------------------------------------
def _stitch_gif(png_paths, out_gif, interval, loop):
    from PIL import Image
    imgs = [Image.open(p).convert("RGBA") for p in png_paths]
    imgs[0].save(out_gif, save_all=True, append_images=imgs[1:],
                 loop=loop, duration=interval, optimize=False)

def _encode_mp4(png_dir, out_mp4, fps):
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", os.path.join(png_dir, "frame_%05d.png"),
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        out_mp4,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\\n{result.stderr}")

def _save_output(png_paths, out_file, fps, interval, loop):
    # Ensure output directory exists (handles nested paths on any OS)
    out_dir = os.path.dirname(os.path.abspath(out_file))
    os.makedirs(out_dir, exist_ok=True)
    if out_file.endswith(".mp4"):
        _encode_mp4(os.path.dirname(png_paths[0]), out_file, fps)
    else:
        _stitch_gif(png_paths, out_file, interval, loop)

# -- config ----------------------------------------------------------
_HAS_3D   = <<<HAS_3D>>>
_OUT_FILE = <<<OUTFILE>>>
_FPS      = <<<FPS>>>
_INTERVAL = <<<INTERVAL>>>
_LOOP     = <<<LOOP>>>
_FRAMES   = <<<FRAMES>>>
_PING_PONG = <<<PING_PONG>>>

def _frame_sequence():
    fwd = list(range(_FRAMES))
    return fwd + list(reversed(fwd[1:-1])) if _PING_PONG else fwd

# -- sequential renderer ---------------------------------------------
def render_sequential():
    frames = _frame_sequence()
    total = len(frames)
    tmpdir = tempfile.mkdtemp(prefix="anim_")
    try:
        paths = []
        for idx, _i in enumerate(frames):
            print(f"  Frame {idx+1}/{total}", end="\\r")
            update(_i)
            _path = os.path.join(tmpdir, f"frame_{idx:05d}.png")
            plt.savefig(_path, dpi=<<<DPI>>>, bbox_inches="tight")
            if _HAS_3D:
                plt.close("all")
            paths.append(_path)
        if not _HAS_3D:
            plt.close("all")
        print()
        _save_output(paths, _OUT_FILE, _FPS, _INTERVAL, _LOOP)
        print("  Saved ->", _OUT_FILE)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

# -- parallel worker (one PNG per frame) -----------------------------
def _render_one(job):
    _frame, _idx, _tmpdir = job
<<<WORKER_BODY>>>
    _path = os.path.join(_tmpdir, f"frame_{_idx:05d}.png")
    plt.savefig(_path, dpi=<<<DPI>>>, bbox_inches="tight")
    plt.close("all")
    return _path

def render_parallel(n_workers):
    frames = _frame_sequence()
    total = len(frames)
    tmpdir = tempfile.mkdtemp(prefix="anim_")
    chunk = max(1, total // (n_workers * 4))
    print(f"  Rendering {total} frames on {n_workers} workers...")
    try:
        jobs = [(_frame, _idx, tmpdir) for _idx, _frame in enumerate(frames)]
        with multiprocessing.Pool(n_workers) as pool:
            paths = list(pool.imap(_render_one, jobs, chunksize=chunk))
        paths.sort()
        print(f"  Stitching {len(paths)} frames...")
        _save_output(paths, _OUT_FILE, _FPS, _INTERVAL, _LOOP)
        print("  Saved ->", _OUT_FILE)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

# -- main ------------------------------------------------------------
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


def _check_ffmpeg():
    """Return True if ffmpeg is available on PATH."""
    return shutil.which("ffmpeg") is not None


def _normalize_var_range(var, range_str):
    """Normalize var and range_str to lists. Accepts str or list[str]."""
    if isinstance(var, str):
        var = [var]
    if isinstance(range_str, str):
        range_str = [range_str]
    return list(var), list(range_str)


# -- Main animate function --
def animate(src, var="t", range_str="0,1", frames=120, fps=25,
            workers=0, dpi=100, out=None, fmt="gif", loop=0,
            reverse=False, ping_pong=False, source_name="<script>"):
    """Convert a static matplotlib script source to an animated script.

    Args:
        src: Source code of the static matplotlib script.
        var: Variable name(s) to animate. str or list[str].
        range_str: "start,end" range per variable. str or list[str].
            Each entry can use math expressions, e.g. "0,2*pi".
        frames: Number of animation frames.
        fps: Frames per second.
        workers: Parallel workers (0 = auto = cpu_count).
        dpi: DPI for rendered frames.
        out: Output filename (None = auto from source_name).
        fmt: Output format: "gif" (default) or "mp4" (requires ffmpeg).
        loop: GIF loop count (0 = loop forever, 1 = play once, N = N times).
        reverse: If True, sweep each range end -> start instead of start -> end.
        ping_pong: If True, play forward then backward for smooth looping.
        source_name: Name of the source script (for messages/docstring).

    Returns:
        Generated Python script as a string.
    """
    assert isinstance(src, str) and len(src.strip()) > 0, \
        "Source code must be a non-empty string"
    assert frames > 0, f"frames must be positive, got {frames}"
    assert fps > 0, f"fps must be positive, got {fps}"
    assert dpi > 0, f"dpi must be positive, got {dpi}"
    assert fmt in ("gif", "mp4"), f"fmt must be 'gif' or 'mp4', got {fmt!r}"
    assert isinstance(loop, int) and loop >= 0, \
        f"loop must be a non-negative integer, got {loop!r}"

    # -- Normalize var / range_str to lists --
    vars_list, ranges_list = _normalize_var_range(var, range_str)

    assert len(vars_list) >= 1, "At least one variable must be provided"
    assert len(vars_list) == len(ranges_list), (
        f"len(var) ({len(vars_list)}) != len(range_str) ({len(ranges_list)}): "
        f"each variable needs exactly one range"
    )
    assert len(set(vars_list)) == len(vars_list), \
        f"Duplicate variable names: {vars_list}"
    for v in vars_list:
        assert isinstance(v, str) and v.isidentifier(), \
            f"Variable name must be a valid identifier, got {v!r}"

    # Parse all ranges
    range_pairs = []
    for rs in ranges_list:
        parts = rs.split(",")
        assert len(parts) == 2, f"Range must be 'start,end', got {rs!r}"
        s = parse_val(parts[0].strip())
        e = parse_val(parts[1].strip())
        assert s != e, f"Range start and end must differ, got {s}"
        range_pairs.append((s, e))

    if reverse:
        range_pairs = [(e, s) for s, e in range_pairs]

    steps = [(e - s) / frames for s, e in range_pairs]

    if fmt == "mp4" and not _check_ffmpeg():
        warnings.warn(
            "ffmpeg not found on PATH. The generated script will fail when run. "
            "Install ffmpeg: https://ffmpeg.org/download.html",
            stacklevel=2,
        )

    if out:
        out_file = out
        if out_file.endswith(".mp4"):
            fmt = "mp4"
        elif out_file.endswith(".gif"):
            fmt = "gif"
    else:
        out_file = Path(source_name).stem + f"_animated.{fmt}"
    interval = 1000 // fps

    # -- Phase 1: AST scan --
    tree, info = scan_ast(src)

    for v in vars_list:
        assert v in info.all_names, \
            f"Variable {v!r} not found in script. Available names: " \
            f"{sorted(info.all_names)}"

    if info.fig_node is None:
        warnings.warn(
            "No explicit figure creation found (plt.subplots, plt.figure, etc.). "
            "The script may use implicit figure creation via plt.plot(). "
            "Animation may not work correctly with implicit figures.",
            stacklevel=2,
        )
    if info.first_draw_node is None:
        warnings.warn(
            "No drawing methods found (plot, scatter, bar, etc.). "
            "The generated animation may produce blank frames.",
            stacklevel=2,
        )

    # Detect which animated vars had integer original values (for int-cast in generated code)
    int_vars = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                for n in ast.walk(target):
                    if isinstance(n, ast.Name) and n.id in vars_list:
                        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, int):
                            int_vars.add(n.id)

    # -- Phase 2: Dependency tracking --
    dep_vars = build_deps(tree, vars_list)

    # -- Phase 3: Partition --
    static_stmts, dynamic_stmts, plot_stmts = partition(
        src, tree, info, dep_vars, vars_list
    )

    # -- Phase 4: Axes clearing --
    clear = _gen_clear_lines(info.ax_info)

    # -- Phase 5: Agg injection --
    final_static = _inject_agg(static_stmts)

    # -- Phase 6: Build sections --
    static_block = "\n".join(final_static)

    has_3d = False
    ax_3d_sources = []
    ax_3d_vars = {"fig"}
    for _ax in info.ax_info:
        if _ax.is_3d:
            has_3d = True
            ax_3d_vars.add(_ax.var_name)
            if _ax.creation_source:
                ax_3d_sources.append(_ax.creation_source)

    fig_recreation = ""
    global_decl = ""
    if has_3d and info.fig_node is not None:
        fig_source = _get_stmt_source(src, info.fig_node)
        # Filter out ax creation sources identical to the fig statement
        ax_3d_sources = [s for s in ax_3d_sources if s != fig_source]
        # Detect whether `fig` is actually a named variable in this script
        fig_is_named = "fig" in info.all_names and any(
            "fig" in info.node_assigns.get(id(s), set())
            for s in tree.body
        )
        if fig_is_named:
            fig_recreation = (
                _ind("fig.clear()") + "\n"
                + _ind(fig_source) + "\n"
                + (_ind(ax_3d_sources) + "\n" if ax_3d_sources else "")
            )
        else:
            # Inline fig (e.g. plt.figure().add_subplot()); just re-run the statement
            fig_recreation = (
                _ind("plt.close('all')") + "\n"
                + _ind(fig_source) + "\n"
                + (_ind(ax_3d_sources) + "\n" if ax_3d_sources else "")
            )
            ax_3d_vars.discard("fig")  # fig not a real variable here
        global_decl = _ind(f"global {', '.join(sorted(ax_3d_vars))}") + "\n" if ax_3d_vars else ""

    # One assignment line per animated variable
    # Variables that were originally int literals are cast back to int
    def _var_assign(v, s, step):
        expr = f"{s!r} + _frame * {step!r}"
        return f"    {v} = int({expr})" if v in int_vars else f"    {v} = {expr}"

    var_assignments = "\n".join(
        _var_assign(v, s, step)
        for v, (s, _e), step in zip(vars_list, range_pairs, steps)
    )

    update_body = (
        var_assignments + "\n"
        + global_decl
        + (_ind(dynamic_stmts) + "\n" if dynamic_stmts else "")
        + (fig_recreation if has_3d else (_ind(clear) + "\n"))
        + _ind(plot_stmts)
    )

    # Worker receives (_frame, _idx, _tmpdir) -- use _frame for data, _idx for filename
    var_assignments_worker = "\n".join(
        _var_assign(v, s, step)
        for v, (s, _e), step in zip(vars_list, range_pairs, steps)
    )
    worker_body = (
        _ind(final_static, 4) + "\n"
        + var_assignments_worker + "\n"
        + (_ind(dynamic_stmts, 4) + "\n" if dynamic_stmts else "")
        + _ind(plot_stmts, 4)
    )

    # Docstring summary: single var uses classic form, multi-var lists all
    if len(vars_list) == 1:
        var_summary = f"{vars_list[0]} sweeps {range_pairs[0][0]!r} -> {range_pairs[0][1]!r}"
    else:
        parts_summary = [f"{v}({s!r}->{e!r})" for v, (s, e) in zip(vars_list, range_pairs)]
        var_summary = "vars: " + ", ".join(parts_summary)

    # -- Phase 7: Assemble --
    result = (TEMPLATE
        .replace("<<<SOURCE>>>",      source_name)
        .replace("<<<STATIC>>>",      static_block)
        .replace("<<<VAR_SUMMARY>>>", var_summary)
        .replace("<<<UPDATE_BODY>>>", update_body)
        .replace("<<<WORKER_BODY>>>", worker_body)
        .replace("<<<FRAMES>>>",      str(frames))
        .replace("<<<INTERVAL>>>",    str(interval))
        .replace("<<<FPS>>>",         str(fps))
        .replace("<<<OUTFILE>>>",     repr(out_file))
        .replace("<<<DPI>>>",         str(dpi))
        .replace("<<<WORKERS>>>",     str(workers))
        .replace("<<<HAS_3D>>>",      str(has_3d))
        .replace("<<<LOOP>>>",        str(loop))
        .replace("<<<PING_PONG>>>",   str(ping_pong))
    )

    return result


# -- CLI entry point --
def main():
    import argparse

    p = argparse.ArgumentParser(
        description="Convert a static matplotlib script to an animation."
    )
    p.add_argument("script",           help="Input matplotlib script to animate")
    p.add_argument("--var",   nargs="+", default=["t"],       metavar="VAR",
                   help="Variable(s) to animate, e.g. --var t  or  --var t alpha")
    p.add_argument("--range", nargs="+", default=["0,1"],    metavar="RANGE",
                   help="start,end per variable, e.g. --range 0,2*pi  or  --range 0,6.28 0,1")
    p.add_argument("--frames",         default=120,    type=int, help="Number of frames")
    p.add_argument("--fps",            default=25,     type=int, help="Frames per second")
    p.add_argument("--workers",        default=0,      type=int, help="0=auto=cpu_count")
    p.add_argument("--dpi",            default=100,    type=int, help="DPI for output")
    p.add_argument("--out",            default=None,   help="Output filename")
    p.add_argument("--format",         default="gif",  choices=["gif", "mp4"], help="Output format (default: gif)")
    p.add_argument("--loop",           default=0,      type=int, help="GIF loop count (0=forever, default: 0)")
    p.add_argument("--reverse",        action="store_true", help="Sweep range end->start")
    p.add_argument("--ping-pong",      action="store_true", help="Play forward then backward")
    args = p.parse_args()

    try:
        src = Path(args.script).read_text(encoding="utf-8")
    except FileNotFoundError:
        p.error(f"Script not found: {args.script}")
    except PermissionError:
        p.error(f"Permission denied reading: {args.script}")
    except UnicodeDecodeError:
        p.error(f"Could not read {args.script} as UTF-8 text")

    # Unwrap single-element lists for backward-compat display
    var_arg   = args.var[0]   if len(args.var)   == 1 else args.var
    range_arg = args.range[0] if len(args.range) == 1 else args.range

    try:
        result = animate(
            src,
            var=var_arg,
            range_str=range_arg,
            frames=args.frames,
            fps=args.fps,
            workers=args.workers,
            dpi=args.dpi,
            out=args.out,
            fmt=args.format,
            loop=args.loop,
            reverse=args.reverse,
            ping_pong=args.ping_pong,
            source_name=args.script,
        )
    except (AssertionError, ValueError) as exc:
        p.error(str(exc))

    out_script = Path(args.script).stem + "_animated.py"
    Path(out_script).write_text(result, encoding="utf-8")
    var_display = ", ".join(args.var)
    print(f"Written  -> {out_script}")
    print(f"   Variables: {var_display}")
    print(f"   Format   : {args.format}{' (ping-pong)' if args.ping_pong else ''}{' (reversed)' if args.reverse else ''}")
    print(f"   Loop     : {'forever' if args.loop == 0 else args.loop}")
    print(f"   Workers  : {args.workers or 'auto (cpu_count)'}")
    print(f"   Run      : python {out_script}")
    print(f"   Seq only : python {out_script} --sequential")


if __name__ == "__main__":
    main()
