"""Comprehensive test suite for mpl_animator.py."""

import ast
import os
import subprocess
import sys
import textwrap
import warnings

import pytest

from mpl_animator import (
    ALL_PLOT_METHODS,
    CONFIG_METHODS,
    DRAW_METHODS,
    AxesInfo,
    animate,
    build_deps,
    parse_val,
    partition,
    scan_ast,
    _inject_agg,
    _gen_clear_lines,
    _normalize_var_range,
    _normalize_values,
)

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


def _read_fixture(name):
    """Read a fixture file and return its source."""
    path = os.path.join(FIXTURES_DIR, name)
    with open(path) as f:
        return f.read()


# ===============================================================
# parse_val - safe expression evaluator
# ===============================================================
class TestParseVal:
    def test_integer(self):
        assert parse_val("42") == 42

    def test_float(self):
        assert parse_val("3.14") == pytest.approx(3.14)

    def test_negative(self):
        assert parse_val("-5") == -5

    def test_named_constants(self):
        import math
        assert parse_val("pi") == pytest.approx(math.pi)
        assert parse_val("tau") == pytest.approx(math.tau)
        assert parse_val("e") == pytest.approx(math.e)

    def test_arithmetic(self):
        assert parse_val("2*pi") == pytest.approx(2 * 3.141592653589793)
        assert parse_val("pi/2") == pytest.approx(1.5707963267948966)
        assert parse_val("2*pi + 1") == pytest.approx(2 * 3.141592653589793 + 1)

    def test_functions(self):
        assert parse_val("sqrt(4)") == pytest.approx(2.0)
        assert parse_val("sin(pi)") == pytest.approx(0.0, abs=1e-10)

    def test_rejects_unsafe_input(self):
        with pytest.raises(ValueError, match="Unknown name"):
            parse_val("foo")
        with pytest.raises((ValueError, SyntaxError)):
            parse_val("__import__('os')")
        with pytest.raises((ValueError, SyntaxError)):
            parse_val("os.system('echo hi')")
        with pytest.raises(ValueError, match="Unsupported"):
            parse_val("eval('1')")

    def test_whitespace_stripped(self):
        assert parse_val("  3  ") == 3


# ===============================================================
# scan_ast - AST scanning
# ===============================================================
class TestScanAST:
    def test_finds_figure_show_draw(self):
        src = textwrap.dedent("""\
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            ax.plot([1,2,3], [1,2,3])
            ax.scatter([1,2], [3,4])
            plt.show()
        """)
        _, info = scan_ast(src)
        assert info.fig_node is not None and info.fig_node.lineno == 2
        assert info.show_node is not None and info.show_node.lineno == 5
        assert info.first_draw_node.lineno == 3
        assert info.last_draw_node.lineno == 4

    def test_config_vs_draw_classification(self):
        src = textwrap.dedent("""\
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            ax.set_title("hello")
            ax.plot([1,2,3], [1,2,3])
            plt.show()
        """)
        tree, info = scan_ast(src)
        stmt3 = tree.body[2]  # ax.set_title
        assert info.node_has_config[id(stmt3)] is True
        assert info.node_has_draw[id(stmt3)] is False

    def test_detects_axes_variables(self):
        src = textwrap.dedent("""\
            import matplotlib.pyplot as plt
            fig, (ax1, ax2) = plt.subplots(1, 2)
            ax1.plot([1], [1])
            plt.show()
        """)
        _, info = scan_ast(src)
        ax_names = {a.var_name for a in info.ax_info}
        assert ax_names == {"ax1", "ax2"}
        assert "fig" not in ax_names

    def test_detects_3d_projection(self):
        src = textwrap.dedent("""\
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
            ax.plot_surface([], [], [])
            plt.show()
        """)
        _, info = scan_ast(src)
        assert info.ax_info[0].is_3d is True

        src2 = textwrap.dedent("""\
            import matplotlib.pyplot as plt
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            plt.show()
        """)
        _, info2 = scan_ast(src2)
        assert any(a.is_3d for a in info2.ax_info)

    def test_augmented_and_annotated_assign(self):
        src = textwrap.dedent("""\
            import matplotlib.pyplot as plt
            import numpy as np
            t: float = 1.0
            x = np.linspace(0, 10, 100)
            y = np.sin(x)
            y += t
            fig, ax = plt.subplots()
            ax.plot(x, y)
            plt.show()
        """)
        tree, info = scan_ast(src)
        stmt_t = tree.body[2]  # t: float = 1.0
        assert "t" in info.node_assigns[id(stmt_t)]
        stmt_aug = tree.body[5]  # y += t
        assert "y" in info.node_assigns[id(stmt_aug)]
        assert "y" in info.node_aug_assigns[id(stmt_aug)]

    def test_all_names_collected(self):
        src = textwrap.dedent("""\
            import matplotlib.pyplot as plt
            x = 1
            y = x + 2
            fig, ax = plt.subplots()
            ax.plot([x], [y])
            plt.show()
        """)
        _, info = scan_ast(src)
        for name in ("x", "y", "fig", "ax", "plt"):
            assert name in info.all_names

    def test_no_figure_no_show(self):
        src = textwrap.dedent("""\
            import matplotlib.pyplot as plt
            plt.plot([1,2,3], [1,2,3])
        """)
        _, info = scan_ast(src)
        assert info.fig_node is None
        assert info.show_node is None


# ===============================================================
# build_deps - dependency tracking
# ===============================================================
class TestBuildDeps:
    def test_direct_and_transitive(self):
        src = textwrap.dedent("""\
            t = 1.0
            a = t + 1
            b = a * 2
            c = b - 1
        """)
        tree = ast.parse(src)
        deps = build_deps(tree, "t")
        assert {"a", "b", "c"} <= deps

    def test_no_false_positives(self):
        src = textwrap.dedent("""\
            t = 1.0
            a = t + 1
            x = 5
            z = x * 2
        """)
        tree = ast.parse(src)
        deps = build_deps(tree, "t")
        assert "a" in deps
        assert "x" not in deps and "z" not in deps

    def test_augassign_and_annassign(self):
        src = textwrap.dedent("""\
            t = 1.0
            y = 0
            y += t
            z: float = t * 2
        """)
        tree = ast.parse(src)
        deps = build_deps(tree, "t")
        assert "y" in deps and "z" in deps

    def test_empty_script(self):
        tree = ast.parse("t = 1.0\n")
        assert build_deps(tree, "t") == set()


# ===============================================================
# partition - statement classification
# ===============================================================
class TestPartition:
    def test_simple_partition(self):
        src = textwrap.dedent("""\
            import matplotlib.pyplot as plt
            import numpy as np
            t = 1.0
            x = np.linspace(0, 10, 100)
            y = np.sin(t * x)
            fig, ax = plt.subplots()
            ax.plot(x, y)
            plt.show()
        """)
        tree, info = scan_ast(src)
        dep_vars = build_deps(tree, "t")
        static, dynamic, plot, _aug = partition(src, tree, info, dep_vars, "t")

        assert any("import matplotlib" in s for s in static)
        assert any("subplots" in s for s in static)
        assert any("y = " in s or "y =" in s for s in dynamic)
        assert any("ax.plot" in s for s in plot)
        all_text = "\n".join(static + dynamic + plot)
        assert "plt.show()" not in all_text

    def test_animated_var_assignment_skipped(self):
        src = textwrap.dedent("""\
            import matplotlib.pyplot as plt
            t = 5.0
            fig, ax = plt.subplots()
            ax.plot([t], [t])
            plt.show()
        """)
        tree, info = scan_ast(src)
        dep_vars = build_deps(tree, "t")
        static_s, dynamic_s, _, _aug = partition(src, tree, info, dep_vars, "t")
        assert not any(s.strip() == "t = 5.0" for s in static_s + dynamic_s)

    def test_multiline_statement_intact(self):
        src = _read_fixture("multiline_stmt.py")
        tree, info = scan_ast(src)
        dep_vars = build_deps(tree, "t")
        _, dynamic, _, _aug = partition(src, tree, info, dep_vars, "t")
        dynamic_text = "\n".join(dynamic)
        assert "np.sin(" in dynamic_text

    def test_config_after_figure_goes_to_plot(self):
        src = textwrap.dedent("""\
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            ax.set_title("Hello")
            ax.plot([1,2], [3,4])
            plt.show()
        """)
        tree, info = scan_ast(src)
        dep_vars = build_deps(tree, "t")
        _, _, plot, _aug = partition(src, tree, info, dep_vars, "t")
        assert any("set_title" in s for s in plot)

    def test_augmented_assign_tracked(self):
        """partition() reports augmented-assigned vars; none for non-aug scripts."""
        src = _read_fixture("augmented_assign.py")
        tree, info = scan_ast(src)
        dep_vars = build_deps(tree, "t")
        _, _, _, aug = partition(src, tree, info, dep_vars, "t")
        assert "y" in aug

        src2 = _read_fixture("simple_line.py")
        tree2, info2 = scan_ast(src2)
        dep_vars2 = build_deps(tree2, "t")
        _, _, _, aug2 = partition(src2, tree2, info2, dep_vars2, "t")
        assert len(aug2) == 0


# ===============================================================
# _inject_agg - Agg backend injection
# ===============================================================
class TestInjectAgg:
    def test_injects_before_matplotlib_import(self):
        stmts = ["import matplotlib.pyplot as plt", "x = 1"]
        result = _inject_agg(stmts)
        assert result[0] == "import matplotlib; matplotlib.use('Agg')"

    def test_injects_at_start_if_no_mpl_import(self):
        result = _inject_agg(["import numpy as np", "x = 1"])
        assert result[0] == "import matplotlib; matplotlib.use('Agg')"

    def test_removes_existing_use_call(self):
        stmts = ["import matplotlib; matplotlib.use('TkAgg')",
                  "import matplotlib.pyplot as plt"]
        result = _inject_agg(stmts)
        assert not any("TkAgg" in s for s in result)
        assert any("Agg" in s for s in result)


# ===============================================================
# _gen_clear_lines - axes clearing
# ===============================================================
class TestGenClearLines:
    def test_no_axes_uses_gca(self):
        result = _gen_clear_lines([])
        assert any("clear" in line for line in result)

    def test_2d_axes_handles_dict_flat_iter(self):
        """Clear code must handle dict (mosaic), 2D array (.flat), and iterable."""
        result = _gen_clear_lines([AxesInfo(var_name="axs")])
        text = "\n".join(result)
        assert "values" in text and "flat" in text and "clear" in text


# ===============================================================
# Code generation - output structure
# ===============================================================
class TestCodeGeneration:
    def test_contains_required_functions(self):
        src = _read_fixture("simple_line.py")
        result = animate(src, var="t", range_str="0,6.28")
        ast.parse(result)
        assert "def update(_frame):" in result
        assert "def _render_one(job):" in result
        assert "matplotlib.use('Agg')" in result
        assert "plt.show()" not in result

    def test_animated_var_in_update(self):
        src = _read_fixture("simple_line.py")
        result = animate(src, var="t", range_str="0,6.28")
        assert "t = 0 + _frame *" in result or "t = 0.0 + _frame *" in result

    def test_custom_output_name(self):
        src = _read_fixture("simple_line.py")
        result = animate(src, var="t", range_str="0,6.28", out="custom.gif")
        assert "custom.gif" in result

    def test_frames_parameter_used(self):
        src = _read_fixture("simple_line.py")
        result = animate(src, var="t", range_str="0,6.28", frames=60)
        assert "_FRAMES" in result and "60" in result

    def test_per_frame_error_handling_in_template(self):
        """Sequential and parallel renderers have try/except, None filtering, RuntimeError."""
        src = _read_fixture("simple_line.py")
        result = animate(src, var="t", range_str="0.5,5", frames=10)
        seq = result.split("def render_sequential():")[1].split("def _render_one")[0]
        worker = result.split("def _render_one(job):")[1].split("def render_parallel")[0]
        par = result.split("def render_parallel(")[1]
        assert "except Exception" in seq and "WARNING: frame" in seq
        assert "except Exception" in worker and "return None" in worker
        assert "if p]" in par or "if p is not None" in par
        assert 'RuntimeError("All frames failed' in result

    def test_worker_body_indented_inside_try(self):
        """Worker content should be at 8-space indent (inside try:)."""
        src = _read_fixture("simple_line.py")
        result = animate(src, var="t", range_str="0.5,5", frames=10)
        worker = result.split("def _render_one(job):")[1].split("def render_parallel")[0]
        in_try = False
        for line in worker.strip().split("\n"):
            if "try:" in line:
                in_try = True
                continue
            if "except" in line:
                break
            if in_try and line.strip():
                assert len(line) - len(line.lstrip()) >= 8, f"Undent: {line!r}"

    def test_aug_assign_gets_global_not_animated_var(self):
        """y += ... produces `global y` but animated var t does NOT get global."""
        src = _read_fixture("augmented_assign.py")
        result = animate(src, var="t", range_str="0.1,3", frames=10)
        ast.parse(result)
        assert "global y" in result
        assert "global t" not in result

    def test_aug_assign_loop_gets_global(self):
        src = _read_fixture("augmented_assign_loop.py")
        result = animate(src, var="n", range_str="1,20", frames=10)
        ast.parse(result)
        assert "global y" in result


# ===============================================================
# Input validation
# ===============================================================
class TestValidation:
    def test_missing_variable_raises(self):
        src = "import matplotlib.pyplot as plt\nfig,ax=plt.subplots()\nax.plot([1],[1])\nplt.show()\n"
        with pytest.raises(AssertionError, match="not found"):
            animate(src, var="nonexistent", range_str="0,1")

    def test_invalid_python_raises(self):
        with pytest.raises(SyntaxError):
            animate("def foo(:", var="t", range_str="0,1")

    def test_empty_source_raises(self):
        with pytest.raises(AssertionError):
            animate("", var="t", range_str="0,1")
        with pytest.raises(AssertionError):
            animate("   \n  \n  ", var="t", range_str="0,1")

    def test_invalid_var_name_raises(self):
        with pytest.raises(AssertionError, match="valid identifier"):
            animate("t = 1", var="123bad", range_str="0,1")

    def test_zero_frames_raises(self):
        with pytest.raises(AssertionError, match="positive"):
            animate("t = 1", var="t", range_str="0,1", frames=0)

    def test_equal_range_raises(self):
        src = "import matplotlib.pyplot as plt\nt = 1\nplt.plot([t],[t])\nplt.show()\n"
        with pytest.raises(AssertionError, match="differ"):
            animate(src, var="t", range_str="5,5")

    def test_bad_range_format_raises(self):
        with pytest.raises(AssertionError, match="start,end"):
            animate("t = 1", var="t", range_str="1,2,3")

    def test_invalid_fmt_and_loop_raise(self):
        src = "import matplotlib.pyplot as plt\nt=1\nfig,ax=plt.subplots()\nax.plot([t],[t])\nplt.show()\n"
        with pytest.raises(AssertionError, match="fmt must be"):
            animate(src, var="t", range_str="0,1", fmt="avi")
        with pytest.raises(AssertionError, match="loop"):
            animate(src, var="t", range_str="0,1", loop=-1)

    def test_no_figure_warns(self):
        src = textwrap.dedent("""\
            import matplotlib.pyplot as plt
            t = 1.0
            plt.plot([t], [t])
            plt.show()
        """)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            animate(src, var="t", range_str="0,5")
            assert any("No explicit figure creation" in str(x.message) for x in w)

    def test_no_draw_methods_warns(self):
        src = textwrap.dedent("""\
            import matplotlib.pyplot as plt
            t = 1.0
            fig, ax = plt.subplots()
            ax.set_title(f"t = {t}")
            plt.show()
        """)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            animate(src, var="t", range_str="0,5")
            assert any("No drawing methods" in str(x.message) for x in w)

    def test_ffmpeg_missing_warns(self, monkeypatch):
        import mpl_animator
        monkeypatch.setattr(mpl_animator, "_check_ffmpeg", lambda: False)
        src = "import matplotlib.pyplot as plt\nt=1\nfig,ax=plt.subplots()\nax.plot([t],[t])\nplt.show()\n"
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            animate(src, var="t", range_str="0,1", fmt="mp4")
            assert any("ffmpeg" in str(x.message) for x in w)


# ===============================================================
# Features - loop, reverse, ping-pong, output naming
# ===============================================================
class TestFeatures:
    def _simple_src(self):
        return textwrap.dedent("""\
            import numpy as np
            import matplotlib.pyplot as plt
            t = 1.0
            x = np.linspace(0, 10, 100)
            y = np.sin(x + t)
            fig, ax = plt.subplots()
            ax.plot(x, y)
            plt.show()
        """)

    def test_loop_values(self):
        r0 = animate(self._simple_src(), var="t", range_str="0,1")
        assert "_LOOP     = 0" in r0
        r3 = animate(self._simple_src(), var="t", range_str="0,1", loop=3)
        assert "_LOOP     = 3" in r3

    def test_reverse_swaps_start_end(self):
        result = animate(self._simple_src(), var="t", range_str="0,5", reverse=True)
        assert "t = 5" in result or "t = 5.0" in result

    def test_ping_pong(self):
        off = animate(self._simple_src(), var="t", range_str="0,1")
        on = animate(self._simple_src(), var="t", range_str="0,1", ping_pong=True)
        assert "_PING_PONG = False" in off
        assert "_PING_PONG = True" in on
        assert "reversed" in on

    def test_output_naming(self):
        gif = animate(self._simple_src(), var="t", range_str="0,1",
                      fmt="gif", source_name="myplot.py")
        assert "myplot_animated.gif" in gif
        mp4 = animate(self._simple_src(), var="t", range_str="0,1",
                      fmt="mp4", source_name="myplot.py")
        assert "myplot_animated.mp4" in mp4

    def test_out_extension_overrides_fmt(self):
        result = animate(self._simple_src(), var="t", range_str="0,1",
                         out="custom.mp4", fmt="gif")
        assert "custom.mp4" in result

    def test_set_theta_offset_goes_to_plot_stmts(self):
        src = textwrap.dedent("""\
            import numpy as np
            import matplotlib.pyplot as plt
            angle = 0.0
            theta = np.linspace(0, 2*np.pi, 100)
            r = np.ones(100)
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1, projection='polar')
            ax.plot(theta, r)
            ax.set_theta_offset(angle)
            plt.show()
        """)
        tree, info = scan_ast(src)
        dep_vars = build_deps(tree, "angle")
        _, dynamic, plot, _aug = partition(src, tree, info, dep_vars, "angle")
        assert any("set_theta_offset" in s for s in plot)
        assert not any("set_theta_offset" in s for s in dynamic)


# ===============================================================
# Method categories
# ===============================================================
class TestMethodCategories:
    def test_draw_and_config_are_disjoint(self):
        assert DRAW_METHODS & CONFIG_METHODS == set()
        assert ALL_PLOT_METHODS == DRAW_METHODS | CONFIG_METHODS

    def test_core_methods_present(self):
        for m in ("plot", "scatter", "bar", "imshow", "hist", "contour",
                  "plot_surface", "quiver", "add_patch", "add_collection",
                  "stackplot", "tripcolor", "voxels"):
            assert m in DRAW_METHODS, f"{m} missing from DRAW_METHODS"
        for m in ("set_title", "set_xlim", "grid", "legend",
                  "tick_params", "set_aspect", "set_facecolor",
                  "set_xscale", "set_yscale", "subplots_adjust"):
            assert m in CONFIG_METHODS, f"{m} missing from CONFIG_METHODS"

    def test_seaborn_methods_in_draw(self):
        for m in ("heatmap", "lineplot", "scatterplot", "barplot",
                  "kdeplot", "regplot", "pairplot", "clustermap"):
            assert m in DRAW_METHODS, f"{m} missing from DRAW_METHODS"

    def test_pandas_methods_in_draw(self):
        for m in ("scatter_matrix", "parallel_coordinates",
                  "andrews_curves", "radviz"):
            assert m in DRAW_METHODS, f"{m} missing from DRAW_METHODS"


# ===============================================================
# End-to-end - animate() on every fixture
# ===============================================================
FIXTURE_CONFIGS = {
    "simple_line.py":       {"var": "t", "range_str": "0,6.28"},
    "multi_subplot.py":     {"var": "t", "range_str": "0.5,4"},
    "surface_3d.py":        {"var": "t", "range_str": "0.5,3"},
    "scatter_3d.py":        {"var": "t", "range_str": "0,6.28"},
    "polar.py":             {"var": "t", "range_str": "0.5,5"},
    "heatmap.py":           {"var": "t", "range_str": "0.2,3"},
    "multiline_stmt.py":    {"var": "t", "range_str": "0.5,5"},
    "implicit_figure.py":   {"var": "t", "range_str": "0.5,3"},
    "add_subplot_3d.py":    {"var": "t", "range_str": "0.5,2"},
    "augmented_assign.py":  {"var": "t", "range_str": "0,3"},
    "bar_chart.py":         {"var": "t", "range_str": "0.5,5"},
    "contour.py":           {"var": "t", "range_str": "0.5,4"},
    "histogram.py":         {"var": "t", "range_str": "0.5,3"},
    "rotate_3d_surface.py":   {"var": "azim",  "range_str": "0,360"},
    "rotate_3d_scatter.py":   {"var": "azim",  "range_str": "0,360"},
    "rotate_2d_polar.py":     {"var": "angle", "range_str": "0,2*pi"},
    "latex_titles.py":        {"var": "t",     "range_str": "0.5,4"},
    "moving_annotation.py":   {"var": "t",     "range_str": "0,6.28"},
    "dashboard_4panel.py":    {"var": "t",     "range_str": "0.5,3"},
    "changing_text.py":       {"var": "t",     "range_str": "0,6.28"},
    "mixed_3d_2d.py":         {"var": "t",     "range_str": "0,6.28"},
    # multi-variable fixtures
    "camera_cinematic.py":    {"var": ["azim", "elev"],  "range_str": ["0,360", "20,60"]},
    "multi_var_2d.py":        {"var": ["freq", "amp"],   "range_str": ["1,5", "0.3,1.5"]},
    "zoom_and_rotate.py":     {"var": ["azim", "zoom"],  "range_str": ["0,360", "0.5,1.5"]},
    # edge-case fixtures
    "func_def_in_script.py":  {"var": "t",  "range_str": "0,6.28"},
    "conditional_style.py":   {"var": "t",  "range_str": "0,1"},
    "for_loop_draw.py":       {"var": "t",  "range_str": "0.1,3"},
    "add_patch_circle.py":    {"var": "t",  "range_str": "0,1.5"},
    "twin_axes.py":           {"var": "t",  "range_str": "0.1,3"},
    "tick_params_spines.py":  {"var": "t",  "range_str": "0.1,3"},
    "tuple_unpack.py":        {"var": "t",  "range_str": "0,6.28"},
    "list_comprehension.py":  {"var": "t",  "range_str": "0.1,3"},
    "nested_tuple_axes.py":   {"var": "t",  "range_str": "0.1,3"},
    "style_use.py":           {"var": "t",  "range_str": "0.1,3"},
    "rcparams.py":            {"var": "t",  "range_str": "0.1,3"},
    "subplot_mosaic.py":      {"var": "t",  "range_str": "0.1,3"},
    "axes_2d_indexing.py":    {"var": "t",  "range_str": "0.1,3"},
    "multi_assign.py":        {"var": "t",  "range_str": "0,6.28"},
    "starred_assign.py":      {"var": "t",  "range_str": "0.1,3"},
    "lambda_capture.py":      {"var": "t",  "range_str": "0,6.28"},
    "try_except.py":          {"var": "t",  "range_str": "0,6.28"},
    "multi_3d_subplots.py":   {"var": "t",  "range_str": "0.5,2"},
    "from_mpl_import.py":     {"var": "t",  "range_str": "0,6.28"},
    "walrus_operator.py":     {"var": "t",  "range_str": "0,6.28"},
    "log_scale.py":           {"var": "t",  "range_str": "0.1,2"},
    "fill_between_where.py":  {"var": "t",  "range_str": "0.1,3"},
    "quiver_plot.py":         {"var": "t",  "range_str": "0.1,3"},
    # real-world fixtures
    "real_world_01.py":       {"var": "freq",        "range_str": "1,10"},
    "real_world_02.py":       {"var": "scale",       "range_str": "0.5,2"},
    "real_world_03.py":       {"var": "error_scale", "range_str": "0.5,3"},
    "real_world_04.py":       {"var": "width",       "range_str": "0.2,0.8"},
    "real_world_05.py":       {"var": "delta",       "range_str": "0.01,0.1"},
    "real_world_06.py":       {"var": "elev",        "range_str": "10,60"},
    "real_world_07.py":       {"var": "alpha",       "range_str": "0.1,0.9"},
    "real_world_08.py":       {"var": "freq",        "range_str": "1,5"},
    "real_world_09.py":       {"var": "scale",       "range_str": "0.5,2"},
    "real_world_10.py":       {"var": "startangle",  "range_str": "0,360"},
    # augmented assign / third-party
    "augmented_assign_loop.py": {"var": "n",  "range_str": "1,20"},
    "seaborn_heatmap.py":     {"var": "n",  "range_str": "5,20"},
    "pandas_plot.py":         {"var": "n",  "range_str": "10,100"},
}


class TestEndToEnd:
    @pytest.mark.parametrize("fixture_name", list(FIXTURE_CONFIGS.keys()))
    def test_fixture_generates_valid_python(self, fixture_name):
        """Each fixture should produce a valid Python script with required functions."""
        src = _read_fixture(fixture_name)
        cfg = FIXTURE_CONFIGS[fixture_name]
        result = animate(src, frames=10, **cfg)
        try:
            ast.parse(result)
        except SyntaxError as e:
            pytest.fail(f"Generated code for {fixture_name} has syntax error: {e}")
        assert "def update(_frame):" in result
        assert "def _render_one(job):" in result
        assert "plt.show()" not in result

    def test_wave_static_backward_compatible(self):
        """The original wave_static.py example should still work."""
        wave_path = os.path.join(os.path.dirname(__file__), "..", "examples", "wave_static.py")
        with open(wave_path, encoding="utf-8") as f:
            src = f.read()
        result = animate(src, var="f", range_str="3,60", frames=10)
        ast.parse(result)
        assert "def update(_frame):" in result


# ===============================================================
# Specific edge-case behavior (beyond "generates valid python")
# ===============================================================
class TestEdgeCases:
    def test_func_def_worker_has_indented_body(self):
        """Worker body must indent all lines of a function definition."""
        src = _read_fixture("func_def_in_script.py")
        result = animate(src, var="t", range_str="0,6.28", frames=10)
        worker = result.split("def _render_one(job):")[1].split("def render_parallel")[0]
        for line in worker.splitlines():
            stripped = line.lstrip()
            if stripped.startswith("return np.sin"):
                indent = len(line) - len(stripped)
                assert indent >= 8
                break
        else:
            pytest.fail("'return np.sin' not found in worker body")

    def test_add_patch_in_plot_stmts(self):
        """ax.add_patch(c) where c depends on t must be in plot_stmts."""
        src = _read_fixture("add_patch_circle.py")
        tree, info = scan_ast(src)
        dep_vars = build_deps(tree, "t")
        _, _, plot, _aug = partition(src, tree, info, dep_vars, "t")
        assert any("add_patch" in s for s in plot)

    def test_tick_params_and_log_scale_in_plot_stmts(self):
        """tick_params and set_xscale/set_yscale after fig go to plot_stmts."""
        src = _read_fixture("log_scale.py")
        tree, info = scan_ast(src)
        dep_vars = build_deps(tree, "t")
        _, _, plot, _aug = partition(src, tree, info, dep_vars, "t")
        assert any("set_xscale" in s for s in plot)
        assert any("set_yscale" in s for s in plot)

    def test_tuple_unpack_deps(self):
        """x, y = sin(t), cos(t) => both depend on t."""
        src = _read_fixture("tuple_unpack.py")
        tree, _ = scan_ast(src)
        deps = build_deps(tree, "t")
        assert "x_val" in deps and "y_val" in deps

    def test_nested_tuple_axes_all_detected(self):
        """((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2) => 4 axes detected."""
        src = _read_fixture("nested_tuple_axes.py")
        _, info = scan_ast(src)
        assert {a.var_name for a in info.ax_info} == {"ax1", "ax2", "ax3", "ax4"}

    def test_style_use_in_static(self):
        """plt.style.use() before fig should be in static stmts."""
        src = _read_fixture("style_use.py")
        tree, info = scan_ast(src)
        dep_vars = build_deps(tree, "t")
        static, _, _, _aug = partition(src, tree, info, dep_vars, "t")
        assert any("style.use" in s for s in static)

    def test_seaborn_heatmap_in_update(self):
        """sns.heatmap() should be in plot_stmts (inside update), not static."""
        src = _read_fixture("seaborn_heatmap.py")
        result = animate(src, var="n", range_str="5,20", frames=10)
        assert "heatmap" in result.split("def update(_frame):")[1].split("def _stitch_gif")[0]

    def test_pandas_plot_in_update(self):
        """df.plot() should be in plot_stmts (inside update)."""
        src = _read_fixture("pandas_plot.py")
        result = animate(src, var="n", range_str="10,100", frames=10)
        assert "df.plot" in result.split("def update(_frame):")[1].split("def _stitch_gif")[0]

    def test_class_def_in_script(self):
        """Class definition in script should go to static, produce valid output."""
        src = textwrap.dedent("""\
            import numpy as np
            import matplotlib.pyplot as plt

            class Wave:
                def __init__(self, freq):
                    self.freq = freq
                def eval(self, x):
                    return np.sin(self.freq * x)

            t = 0.5
            w = Wave(t)
            x = np.linspace(0, 10, 100)
            y = w.eval(x)
            fig, ax = plt.subplots()
            ax.plot(x, y)
            plt.show()
        """)
        result = animate(src, var="t", range_str="0.1,3", frames=10)
        ast.parse(result)

    def test_ind_helpers(self):
        """_ind on strings and lists with embedded newlines."""
        from mpl_animator import _ind
        assert _ind("def foo():\n    return 1", 4) == "    def foo():\n        return 1"
        items = ["x = 1", "if True:\n    y = 2", "z = 3"]
        assert _ind(items, 4) == "    x = 1\n    if True:\n        y = 2\n    z = 3"

    def test_funcanimation_stripped(self):
        """Scripts with existing FuncAnimation should have it stripped."""
        src = textwrap.dedent("""\
            import numpy as np
            import matplotlib.pyplot as plt
            from matplotlib.animation import FuncAnimation

            fig, ax = plt.subplots()
            x = np.linspace(0, 2*np.pi, 100)
            amp = 1.0
            line, = ax.plot(x, amp * np.sin(x))
            ax.set_ylim(-3, 3)

            def animate_inner(frame):
                line.set_ydata(amp * np.sin(x + frame/10))
                return line,

            ani = FuncAnimation(fig, animate_inner, frames=100, interval=50, blit=True)
            plt.show()
        """)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = animate(src, var="amp", range_str="0.5,3", frames=10)
            assert any("FuncAnimation detected" in str(x.message) for x in w)
        ast.parse(result)
        assert "def animate_inner" not in result
        assert "ani = FuncAnimation" not in result
        assert "def update(_frame):" in result

    def test_dynamic_stmt_using_axes_goes_to_plot_stmts(self):
        """A function call that uses axes AND depends on animated var must run after clear."""
        src = textwrap.dedent("""\
            import numpy as np
            import matplotlib.pyplot as plt

            phi = 0.5
            ts = np.cumsum(np.random.randn(200) * phi)

            fig, axes = plt.subplots(1, 2)

            def my_acf(data, ax):
                ax.bar(range(10), data[:10])

            my_acf(ts, ax=axes[1])
            axes[0].plot(ts)
            axes[0].set_title(f'phi={phi}')
            plt.show()
        """)
        result = animate(src, var="phi", range_str="0.1,0.9", frames=10)
        ast.parse(result)
        body = result.split("def update(_frame):")[1].split("def _stitch_gif")[0]
        lines = body.strip().split("\n")
        clear_idx = next(i for i, l in enumerate(lines) if "clear" in l)
        acf_idx = next(i for i, l in enumerate(lines) if "my_acf(ts" in l)
        assert acf_idx > clear_idx, "axes-using dynamic stmt must be after clear"

    def test_non_standard_pyplot_alias(self):
        """import matplotlib.pyplot as MPL must still produce valid code."""
        src = textwrap.dedent("""\
            import matplotlib.pyplot as MPL
            import numpy as np
            t = 1.0
            x = np.linspace(0, 10, 100)
            y = np.sin(t * x)
            fig, ax = MPL.subplots()
            ax.plot(x, y)
            MPL.show()
        """)
        result = animate(src, var="t", range_str="0.5,5", frames=5)
        ast.parse(result)
        assert "import matplotlib.pyplot as plt" in result

    def test_main_guard_unwrapped(self):
        """Code inside if __name__ == '__main__': must be analyzed for draw calls."""
        src = textwrap.dedent("""\
            import matplotlib.pyplot as plt
            import numpy as np
            if __name__ == '__main__':
                t = 1.0
                x = np.linspace(0, 10, 100)
                y = np.sin(t * x)
                fig, ax = plt.subplots()
                ax.plot(x, y)
                plt.show()
        """)
        result = animate(src, var="t", range_str="0.5,5", frames=5)
        ast.parse(result)
        body = result.split("def update(_frame):")[1].split("def _stitch_gif")[0]
        assert "ax.plot" in body

    def test_wildcard_import_rewritten(self):
        """from matplotlib.pyplot import * must be rewritten to avoid worker crash."""
        src = textwrap.dedent("""\
            from matplotlib.pyplot import *
            import numpy as np
            t = 1.0
            x = np.linspace(0, 10, 100)
            y = np.sin(t * x)
            fig, ax = subplots()
            ax.plot(x, y)
            show()
        """)
        result = animate(src, var="t", range_str="0.5,5", frames=5)
        ast.parse(result)
        # Wildcard stays at module level for update() path
        assert "from matplotlib.pyplot import *" in result.split("def update")[0]
        # Worker body must NOT have wildcard (illegal inside function)
        worker = result.split("def _render_one(job):")[1].split("def render_parallel")[0]
        assert "from matplotlib.pyplot import *" not in worker
        assert "import matplotlib.pyplot as plt" in worker
        assert "from matplotlib.pyplot import" in worker  # explicit names

    def test_style_context_unwrapped(self):
        """with plt.style.context(...) must be unwrapped for draw detection."""
        src = textwrap.dedent("""\
            import numpy as np
            import matplotlib.pyplot as plt
            freq = 5.0
            t = np.linspace(0, 1, 500)
            signal = np.sin(2 * np.pi * freq * t)
            with plt.style.context('ggplot'):
                fig, ax = plt.subplots()
                ax.plot(t, signal)
                ax.set_title(f'freq={freq}')
                plt.show()
        """)
        result = animate(src, var="freq", range_str="1,20", frames=5)
        ast.parse(result)
        body = result.split("def update(_frame):")[1].split("def _stitch_gif")[0]
        assert "ax.plot" in body
        assert "signal" in body


# ===============================================================
# Multi-variable animation
# ===============================================================
class TestMultiVar:
    SIMPLE_SRC = textwrap.dedent("""\
        import numpy as np
        import matplotlib.pyplot as plt

        freq = 1.0
        amp  = 1.0
        x = np.linspace(0, 2 * np.pi, 200)
        y = amp * np.sin(freq * x)

        fig, ax = plt.subplots()
        ax.plot(x, y, lw=2)
        ax.set_ylim(-2, 2)
        ax.set_title(f'freq={freq:.2f} amp={amp:.2f}')
        plt.tight_layout()
        plt.show()
    """)

    def test_normalize_var_range(self):
        v, r = _normalize_var_range("t", "0,1")
        assert v == ["t"] and r == ["0,1"]
        v, r = _normalize_var_range(["freq", "amp"], ["1,5", "0.3,1.5"])
        assert v == ["freq", "amp"] and r == ["1,5", "0.3,1.5"]

    def test_mismatched_var_range_raises(self):
        with pytest.raises(AssertionError, match="each variable needs exactly one range"):
            animate(self.SIMPLE_SRC, var=["freq", "amp"], range_str="1,5")

    def test_duplicate_vars_raises(self):
        with pytest.raises(AssertionError, match="Duplicate variable names"):
            animate(self.SIMPLE_SRC, var=["freq", "freq"], range_str=["1,5", "1,5"])

    def test_unknown_var_raises(self):
        with pytest.raises(AssertionError, match="not found in script"):
            animate(self.SIMPLE_SRC, var=["freq", "nonexistent"], range_str=["1,5", "0,1"])

    def test_two_vars_in_update_and_worker(self):
        result = animate(self.SIMPLE_SRC, var=["freq", "amp"], range_str=["1,5", "0.3,1.5"])
        ast.parse(result)
        update = result.split("def update(_frame):")[1].split("def _render_one")[0]
        worker = result.split("def _render_one(job):")[1].split("def render_parallel")[0]
        for section in (update, worker):
            assert "freq = " in section and "amp = " in section

    def test_original_var_assignment_skipped(self):
        tree, info = scan_ast(self.SIMPLE_SRC)
        deps = build_deps(tree, ["freq", "amp"])
        static, dynamic, _, _aug = partition(self.SIMPLE_SRC, tree, info, deps, ["freq", "amp"])
        combined = "\n".join(static + dynamic)
        assert "freq = 1.0" not in combined and "amp  = 1.0" not in combined

    def test_step_values_are_correct(self):
        result = animate(self.SIMPLE_SRC, var=["freq", "amp"],
                         range_str=["1,5", "0.3,1.5"], frames=100)
        assert repr((5.0 - 1.0) / 100) in result
        assert repr((1.5 - 0.3) / 100) in result

    def test_build_deps_multi_var(self):
        tree, _ = scan_ast(self.SIMPLE_SRC)
        deps = build_deps(tree, ["freq", "amp"])
        assert "y" in deps

    @pytest.mark.slow
    def test_multi_var_2d_produces_gif(self, tmp_path):
        src = _read_fixture("multi_var_2d.py")
        gif_name = "multi_var_2d_animated.gif"
        result = animate(src, var=["freq", "amp"], range_str=["1,5", "0.3,1.5"],
                         frames=6, fps=6, out=gif_name, workers=1)
        script_path = tmp_path / "mv2d.py"
        script_path.write_text(result, encoding="utf-8")
        proc = subprocess.run(
            [sys.executable, str(script_path), "--sequential"],
            capture_output=True, text=True, timeout=60, cwd=str(tmp_path),
        )
        assert proc.returncode == 0, f"multi_var_2d failed:\n{proc.stderr}"
        assert (tmp_path / gif_name).exists()


# ===============================================================
# Explicit values (--values / values=)
# ===============================================================
class TestExplicitValues:
    SIMPLE_SRC = textwrap.dedent("""\
        import matplotlib.pyplot as plt
        import numpy as np
        n = 10
        x = np.linspace(0, 1, n)
        y = np.sin(np.pi * x)
        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.set_title(f'n={n}')
        plt.show()
    """)

    def test_normalize_values(self):
        assert _normalize_values("1,5,10") == [[1.0, 5.0, 10.0]]
        assert _normalize_values([1, 5, 10]) == [[1.0, 5.0, 10.0]]
        assert _normalize_values(["1,5,10", "0,1,2"]) == [[1.0, 5.0, 10.0], [0.0, 1.0, 2.0]]
        assert _normalize_values([[1, 5, 10], [0, 1, 2]]) == [[1.0, 5.0, 10.0], [0.0, 1.0, 2.0]]
        import math
        result = _normalize_values("0,pi,2*pi")
        assert result[0][1] == pytest.approx(math.pi)

    def test_values_overrides_frames(self):
        result = animate(self.SIMPLE_SRC, var="n", values="5,10,20,50,100", fps=10)
        assert "_FRAMES   = 5" in result

    def test_values_generates_valid_code(self):
        result = animate(self.SIMPLE_SRC, var="n", values="5,10,20,50,100", fps=10)
        ast.parse(result)
        assert "_VALUES_n = " in result
        update = result.split("def update(_frame):")[1].split("def _stitch_gif")[0]
        assert "_VALUES_n[_frame]" in update

    def test_values_int_cast(self):
        """n was int literal => int(_VALUES_n[_frame]); float var => no cast."""
        result = animate(self.SIMPLE_SRC, var="n", values="5,10,20,50,100", fps=10)
        assert "int(_VALUES_n[_frame])" in result

        src_float = textwrap.dedent("""\
            import matplotlib.pyplot as plt
            alpha = 0.5
            fig, ax = plt.subplots()
            ax.plot([1,2,3], [1,2,3], alpha=alpha)
            plt.show()
        """)
        result_f = animate(src_float, var="alpha", values="0.1,0.3,0.5,0.7,0.9", fps=10)
        assert "int(_VALUES_alpha" not in result_f

    def test_values_backwards_compat_range_still_works(self):
        result = animate(self.SIMPLE_SRC, var="n", range_str="5,100", frames=10, fps=10)
        assert "_VALUES_n" not in result

    def test_values_as_list_of_numbers(self):
        result = animate(self.SIMPLE_SRC, var="n", values=[5, 10, 20, 50], fps=10)
        ast.parse(result)
        assert "_FRAMES   = 4" in result

    def test_values_validation(self):
        src2 = textwrap.dedent("""\
            import matplotlib.pyplot as plt
            import numpy as np
            a = 1.0
            b = 2.0
            fig, ax = plt.subplots()
            ax.plot([a, b], [a, b])
            plt.show()
        """)
        with pytest.raises(AssertionError, match="same length"):
            animate(src2, var=["a", "b"], values=["1,2,3", "10,20"], fps=10)
        with pytest.raises(AssertionError, match="at least 2"):
            animate(self.SIMPLE_SRC, var="n", values="5", fps=10)
        with pytest.raises(AssertionError, match=r"len\(values\)"):
            animate(self.SIMPLE_SRC, var="n", values=["5,10,20", "1,2,3"], fps=10)

    def test_values_multivar(self):
        src = textwrap.dedent("""\
            import matplotlib.pyplot as plt
            import numpy as np
            a = 1.0
            b = 2.0
            x = np.linspace(0, 1, 50)
            y = a * x + b
            fig, ax = plt.subplots()
            ax.plot(x, y)
            plt.show()
        """)
        result = animate(src, var=["a", "b"], values=["1,2,3", "10,20,30"], fps=10)
        ast.parse(result)
        assert "_VALUES_a = " in result and "_VALUES_b = " in result
        assert "_FRAMES   = 3" in result

    def test_values_worker_body_uses_index(self):
        result = animate(self.SIMPLE_SRC, var="n", values="5,10,20,50,100", fps=10)
        worker = result.split("def _render_one(job):")[1].split("def render_parallel")[0]
        assert "_VALUES_n[_frame]" in worker


# ===============================================================
# Slow tests - actually run the generated scripts
# ===============================================================
@pytest.mark.slow
class TestSlowExecution:
    """Tests that actually execute the generated animated scripts."""

    @pytest.mark.parametrize("fixture_name", list(FIXTURE_CONFIGS.keys()))
    def test_generated_script_produces_gif(self, fixture_name, tmp_path):
        if fixture_name == "implicit_figure.py":
            pytest.xfail("no explicit figure creation")
        if fixture_name == "mixed_3d_2d.py":
            pytest.xfail("mixed 2D/3D subplots known limitation")
        _OPTIONAL_LIBS = {"seaborn_heatmap.py": "seaborn", "pandas_plot.py": "pandas"}
        if fixture_name in _OPTIONAL_LIBS:
            pytest.importorskip(_OPTIONAL_LIBS[fixture_name])
        src = _read_fixture(fixture_name)
        cfg = FIXTURE_CONFIGS[fixture_name]
        gif_name = fixture_name.replace(".py", "_animated.gif")
        result = animate(src, frames=5, fps=5, out=gif_name, workers=1, **cfg)

        script_path = tmp_path / "animated.py"
        script_path.write_text(result, encoding="utf-8")
        proc = subprocess.run(
            [sys.executable, str(script_path), "--sequential"],
            capture_output=True, text=True, timeout=60, cwd=str(tmp_path),
        )
        assert proc.returncode == 0, \
            f"Script failed:\nstdout: {proc.stdout}\nstderr: {proc.stderr}"
        gif_path = tmp_path / gif_name
        assert gif_path.exists() and gif_path.stat().st_size > 0

        import shutil
        out_dir = os.path.join(os.path.dirname(__file__), "output")
        os.makedirs(out_dir, exist_ok=True)
        shutil.copy(gif_path, os.path.join(out_dir, gif_name))

    def test_ping_pong_produces_more_frames(self, tmp_path):
        src = _read_fixture("simple_line.py")
        normal = animate(src, var="t", range_str="0,6.28", frames=5,
                         fps=5, out="normal.gif", workers=1)
        pp = animate(src, var="t", range_str="0,6.28", frames=5,
                     fps=5, out="pp.gif", ping_pong=True, workers=1)

        for code, name in [(normal, "normal.gif"), (pp, "pp.gif")]:
            p = tmp_path / name.replace(".gif", ".py")
            p.write_text(code, encoding="utf-8")
            proc = subprocess.run(
                [sys.executable, str(p), "--sequential"],
                capture_output=True, text=True, timeout=60, cwd=str(tmp_path),
            )
            assert proc.returncode == 0

        from PIL import Image
        assert Image.open(tmp_path / "pp.gif").n_frames > \
               Image.open(tmp_path / "normal.gif").n_frames
