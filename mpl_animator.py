"""
mpl_animator.py -- convert ANY static matplotlib script to an animation.
Author: Basem Rajjoub
Uses only matplotlib + multiprocessing + Pillow. No celluloid needed.

Usage (CLI):
    python mpl_animator.py plot.py --var f --range "3,60"
    python mpl_animator.py plot.py --var t --range "0,2*pi" --frames 120 --fps 30
    python mpl_animator.py plot.py --var alpha --range "0,1" --workers 8
    (run generated script with --sequential to skip parallel)

Usage (library):
    from mpl_animator import animate
    code = animate(open("plot.py").read(), var="f", range_str="3,60")
"""

import ast
import math
import operator
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
}

CONFIG_METHODS = {
    "set_title", "set_xlabel", "set_ylabel", "set_zlabel",
    "set_xlim", "set_ylim", "set_zlim", "set_xticks", "set_yticks",
    "set_xticklabels", "set_yticklabels",
    "grid", "legend", "colorbar", "text", "annotate",
    "tight_layout", "suptitle", "view_init",
    "set_proj_type", "set_theta_zero_location", "set_rlabel_position",
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
    """Recursively evaluate an AST expression using only safe operations."""
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
    subplot_spec: str = ""       # e.g. "1, 1, 1" for add_subplot(1,1,1)
    creation_source: str = ""    # full source line that creates this axes


@dataclass
class ASTInfo:
    fig_node: ast.stmt | None = None
    show_node: ast.stmt | None = None
    first_draw_node: ast.stmt | None = None
    last_draw_node: ast.stmt | None = None
    node_assigns: dict = field(default_factory=dict)       # node -> set of var names
    node_has_draw: dict = field(default_factory=dict)       # node -> bool
    node_has_config: dict = field(default_factory=dict)     # node -> bool
    node_is_show: dict = field(default_factory=dict)        # node -> bool
    node_is_fig_creation: dict = field(default_factory=dict)  # node -> bool
    ax_info: list = field(default_factory=list)             # list of AxesInfo
    all_names: set = field(default_factory=set)             # all Name nodes in script


def _get_assigned_names(target_node):
    """Extract all variable names from an assignment target (handles tuples)."""
    names = set()
    for n in ast.walk(target_node):
        if isinstance(n, ast.Name):
            names.add(n.id)
    return names


def _node_uses_names(node):
    """Get all Name references in an AST node."""
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def _check_3d_projection(call_node):
    """Check if a call has projection='3d' keyword argument."""
    for kw in call_node.keywords:
        if kw.arg == "projection":
            if isinstance(kw.value, ast.Constant) and kw.value.value == "3d":
                return True
        if kw.arg == "subplot_kw":
            # subplot_kw={'projection': '3d'} in plt.subplots()
            if isinstance(kw.value, ast.Dict):
                for k, v in zip(kw.value.keys, kw.value.values):
                    if (isinstance(k, ast.Constant) and k.value == "projection"
                            and isinstance(v, ast.Constant) and v.value == "3d"):
                        return True
    return False


def _extract_subplot_spec(call_node):
    """Extract subplot spec args as a string, e.g. '1, 1, 1'."""
    parts = []
    for arg in call_node.args:
        if isinstance(arg, ast.Constant):
            parts.append(repr(arg.value))
    return ", ".join(parts)


def scan_ast(src):
    """Phase 1: Parse source and extract structural information."""
    tree = ast.parse(src)
    info = ASTInfo()

    # Collect all names used in the script
    info.all_names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}

    # Analyze each top-level statement
    for stmt in tree.body:
        stmt_id = id(stmt)
        info.node_assigns[stmt_id] = set()
        info.node_has_draw[stmt_id] = False
        info.node_has_config[stmt_id] = False
        info.node_is_show[stmt_id] = False
        info.node_is_fig_creation[stmt_id] = False

        # Walk all nodes in this statement
        for node in ast.walk(stmt):
            # Track assignments (Assign, AugAssign, AnnAssign)
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    info.node_assigns[stmt_id] |= _get_assigned_names(target)
            elif isinstance(node, ast.AugAssign):
                info.node_assigns[stmt_id] |= _get_assigned_names(node.target)
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                info.node_assigns[stmt_id] |= _get_assigned_names(node.target)

            # Track method calls
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

                    # Extract axes variables from assignment targets
                    if isinstance(stmt, ast.Assign):
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
    """Phase 2: Build transitive dependency graph for `var`."""
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

    # Transitive closure: find all variables that depend on `var`
    found, queue = set(), {var}
    while queue:
        cur = queue.pop()
        for k, v in rhs_uses.items():
            if cur in v and k not in found:
                found.add(k)
                queue.add(k)
    return found


# -- Statement partitioning --
def _get_stmt_source(src, stmt):
    """Extract full source text for a statement, handling multi-line."""
    seg = ast.get_source_segment(src, stmt)
    if seg is not None:
        return seg
    # Fallback: use line numbers
    lines = src.splitlines()
    start = stmt.lineno - 1
    end = getattr(stmt, "end_lineno", stmt.lineno)
    return "\n".join(lines[start:end])


def partition(src, tree, info, dep_vars, var):
    """Phase 3: Partition top-level statements into static/dynamic/plot/show."""
    static_stmts = []    # run once (setup)
    dynamic_stmts = []   # recalculated per frame (depend on animated var)
    plot_stmts = []      # drawing commands (run per frame after dynamic)

    fig_found = False

    for stmt in tree.body:
        sid = id(stmt)
        text = _get_stmt_source(src, stmt)

        # Skip show() calls entirely
        if info.node_is_show[sid]:
            continue

        assigned = info.node_assigns[sid]
        has_draw = info.node_has_draw[sid]
        has_config = info.node_has_config[sid]
        is_fig = info.node_is_fig_creation[sid]

        if is_fig:
            fig_found = True

        # Names used in this statement
        stmt_names = _node_uses_names(stmt)

        # If it's figure creation, always static
        if is_fig:
            static_stmts.append(text)
            continue

        # If it assigns the animated variable itself, skip it
        # (we generate var = start + frame * step)
        if var in assigned and not (assigned - {var}):
            # Pure assignment of the animated variable -- skip
            # But only if it's not also a draw/config call
            if not has_draw and not has_config:
                continue

        depends_on_var = bool(assigned & (dep_vars | {var})) or (var in stmt_names)

        if not fig_found:
            # Before figure creation: static or dynamic setup
            if depends_on_var:
                dynamic_stmts.append(text)
            else:
                static_stmts.append(text)
        else:
            # After figure creation: plot/config or dynamic
            if has_draw or has_config:
                plot_stmts.append(text)
            elif depends_on_var:
                dynamic_stmts.append(text)
            else:
                static_stmts.append(text)

    return static_stmts, dynamic_stmts, plot_stmts


# -- Axes clearing code generation --
def _gen_clear_lines(ax_info_list):
    """Generate clearing code for axes, handling 2D and 3D differently."""
    if not ax_info_list:
        return ["plt.gca().clear()"]

    out = []
    has_3d = any(ax.is_3d for ax in ax_info_list)

    if has_3d:
        # For 3D axes, we need to recreate them to preserve projection
        # Use fig.clear() then recreate all axes from the figure creation line
        # This is handled at a higher level -- we include the figure creation
        # in the update body and skip clear entirely
        pass

    for ax in ax_info_list:
        v = ax.var_name
        if ax.is_3d:
            # 3D axes: clear and re-set projection type
            out += [
                f"_axs = list({v}) if hasattr({v},'__iter__') else [{v}]",
                f"[_a.clear() for _a in _axs]",
            ]
        else:
            out += [
                f"_axs = list({v}) if hasattr({v},'__iter__') else [{v}]",
                f"[_a.clear() for _a in _axs]",
            ]

    return out if out else ["plt.gca().clear()"]


# -- Agg backend injection --
def _inject_agg(static_stmts):
    """Inject matplotlib.use('Agg') before first matplotlib import."""
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
    """Indent a list of strings (or a single multi-line string) by n spaces."""
    pad = " " * n
    if isinstance(lst, str):
        lst = lst.splitlines()
    return "\n".join(pad + line for line in lst)


# -- Code generation template --
TEMPLATE = '''\
"""Auto-generated by mpl_animator.py from <<<SOURCE>>>
   <<<VAR>>> sweeps <<<START>>> -> <<<END>>> over <<<FRAMES>>> frames @ <<<FPS>>>fps
"""
<<<STATIC>>>
import matplotlib.animation as animation
import multiprocessing, os, tempfile, shutil, time
from PIL import Image

# -- frame update ----------------------------------------------------
def update(_frame):
<<<UPDATE_BODY>>>

# -- sequential (FuncAnimation) --------------------------------------
def render_sequential():
    ani = animation.FuncAnimation(
        fig, update, frames=<<<FRAMES>>>, interval=<<<INTERVAL>>>, blit=False)
    ani.save("<<<OUTGIF>>>", writer="pillow", fps=<<<FPS>>>)
    print("  Saved ->", "<<<OUTGIF>>>")

# -- parallel worker (one PNG per frame) -----------------------------
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


# -- Main animate function --
def animate(src, var="t", range_str="0,1", frames=120, fps=25,
            workers=0, dpi=100, out=None, source_name="<script>"):
    """Convert a static matplotlib script source to an animated script.

    Args:
        src: Source code of the static matplotlib script.
        var: Variable name to animate.
        range_str: "start,end" range (math expressions OK, e.g. "0,2*pi").
        frames: Number of animation frames.
        fps: Frames per second.
        workers: Parallel workers (0 = auto = cpu_count).
        dpi: DPI for rendered frames.
        out: Output GIF filename (None = auto from source_name).
        source_name: Name of the source script (for messages/docstring).

    Returns:
        Generated Python script as a string.

    Raises:
        SyntaxError: If src is not valid Python.
        ValueError: If the animation variable is not found in the script.
    """
    # -- Validate input --
    assert isinstance(src, str) and len(src.strip()) > 0, \
        "Source code must be a non-empty string"
    assert isinstance(var, str) and var.isidentifier(), \
        f"Variable name must be a valid identifier, got {var!r}"
    assert frames > 0, f"frames must be positive, got {frames}"
    assert fps > 0, f"fps must be positive, got {fps}"
    assert dpi > 0, f"dpi must be positive, got {dpi}"

    # Parse range
    parts = range_str.split(",")
    assert len(parts) == 2, \
        f"Range must be 'start,end', got {range_str!r}"
    start_val = parse_val(parts[0])
    end_val = parse_val(parts[1])
    assert start_val != end_val, \
        f"Range start and end must differ, got {start_val}"

    step = (end_val - start_val) / frames
    out_gif = out or (Path(source_name).stem + "_animated.gif")
    interval = 1000 // fps

    # -- Phase 1: AST scan --
    tree, info = scan_ast(src)

    # Validate the animation variable exists
    assert var in info.all_names, \
        f"Variable {var!r} not found in script. Available names: " \
        f"{sorted(info.all_names)}"

    # Warnings for unusual cases
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

    # -- Phase 2: Dependency tracking --
    dep_vars = build_deps(tree, var)

    # -- Phase 3: Partition --
    static_stmts, dynamic_stmts, plot_stmts = partition(
        src, tree, info, dep_vars, var
    )

    # -- Phase 4: Axes clearing --
    clear = _gen_clear_lines(info.ax_info)

    # -- Phase 5: Agg injection --
    final_static = _inject_agg(static_stmts)

    # -- Phase 6: Build sections --
    static_block = "\n".join(final_static)

    # For 3D plots, include figure creation in the update body
    has_3d = any(ax.is_3d for ax in info.ax_info)
    fig_recreation = ""
    if has_3d and info.fig_node is not None:
        fig_source = _get_stmt_source(src, info.fig_node)
        fig_recreation = f"    fig.clear()\n    {fig_source}\n"

    # update() body
    update_body = (
        f"    {var} = {start_val!r} + _frame * {step!r}\n"
        + (_ind(dynamic_stmts) + "\n" if dynamic_stmts else "")
        + (fig_recreation if has_3d else (_ind(clear) + "\n"))
        + _ind(plot_stmts)
    )

    # worker body: re-create fig inside worker process
    worker_body = (
        _ind(final_static, 4) + "\n"
        + f"    {var} = {start_val!r} + _frame * {step!r}\n"
        + (_ind(dynamic_stmts, 4) + "\n" if dynamic_stmts else "")
        + _ind(plot_stmts, 4)
    )

    # -- Phase 7: Assemble --
    result = (TEMPLATE
        .replace("<<<SOURCE>>>",      source_name)
        .replace("<<<STATIC>>>",      static_block)
        .replace("<<<VAR>>>",         var)
        .replace("<<<START>>>",       repr(start_val))
        .replace("<<<END>>>",         repr(end_val))
        .replace("<<<STEP>>>",        repr(step))
        .replace("<<<UPDATE_BODY>>>", update_body)
        .replace("<<<WORKER_BODY>>>", worker_body)
        .replace("<<<FRAMES>>>",      str(frames))
        .replace("<<<INTERVAL>>>",    str(interval))
        .replace("<<<FPS>>>",         str(fps))
        .replace("<<<OUTGIF>>>",      out_gif)
        .replace("<<<DPI>>>",         str(dpi))
        .replace("<<<WORKERS>>>",     str(workers))
    )

    return result


# -- CLI entry point --
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="Convert a static matplotlib script to an animation."
    )
    p.add_argument("script", help="Input matplotlib script to animate")
    p.add_argument("--var",     default="t",   help="Variable to animate (default: t)")
    p.add_argument("--range",   default="0,1", help="start,end (math ok: 2*pi)")
    p.add_argument("--frames",  default=120,   type=int, help="Number of frames")
    p.add_argument("--fps",     default=25,    type=int, help="Frames per second")
    p.add_argument("--workers", default=0,     type=int, help="0=auto=cpu_count")
    p.add_argument("--dpi",     default=100,   type=int, help="DPI for output")
    p.add_argument("--out",     default=None,  help="Output GIF filename")
    args = p.parse_args()

    src = Path(args.script).read_text(encoding="utf-8")

    result = animate(
        src,
        var=args.var,
        range_str=args.range,
        frames=args.frames,
        fps=args.fps,
        workers=args.workers,
        dpi=args.dpi,
        out=args.out,
        source_name=args.script,
    )

    out_script = Path(args.script).stem + "_animated.py"
    Path(out_script).write_text(result)
    print(f"Written  -> {out_script}")
    print(f"   Variable : {args.var}")
    print(f"   Workers  : {args.workers or 'auto (cpu_count)'}")
    print(f"   Run      : python {out_script}")
    print(f"   Seq only : python {out_script} --sequential")
