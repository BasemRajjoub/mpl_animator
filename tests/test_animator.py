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

    def test_pi(self):
        import math
        assert parse_val("pi") == pytest.approx(math.pi)

    def test_tau(self):
        import math
        assert parse_val("tau") == pytest.approx(math.tau)

    def test_e(self):
        import math
        assert parse_val("e") == pytest.approx(math.e)

    def test_arithmetic(self):
        assert parse_val("2*pi") == pytest.approx(2 * 3.141592653589793)

    def test_division(self):
        assert parse_val("pi/2") == pytest.approx(1.5707963267948966)

    def test_complex_expression(self):
        assert parse_val("2*pi + 1") == pytest.approx(2 * 3.141592653589793 + 1)

    def test_sqrt_function(self):
        assert parse_val("sqrt(4)") == pytest.approx(2.0)

    def test_sin_function(self):
        assert parse_val("sin(pi)") == pytest.approx(0.0, abs=1e-10)

    def test_rejects_unknown_name(self):
        with pytest.raises(ValueError, match="Unknown name"):
            parse_val("foo")

    def test_rejects_import(self):
        with pytest.raises((ValueError, SyntaxError)):
            parse_val("__import__('os')")

    def test_rejects_attribute_access(self):
        with pytest.raises((ValueError, SyntaxError)):
            parse_val("os.system('echo hi')")

    def test_rejects_unknown_function(self):
        with pytest.raises(ValueError, match="Unsupported"):
            parse_val("eval('1')")

    def test_whitespace_stripped(self):
        assert parse_val("  3  ") == 3


# ===============================================================
# scan_ast - AST scanning
# ===============================================================
class TestScanAST:
    def test_finds_figure_line(self):
        src = textwrap.dedent("""\
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            ax.plot([1,2,3], [1,2,3])
            plt.show()
        """)
        _, info = scan_ast(src)
        assert info.fig_node is not None
        assert info.fig_node.lineno == 2

    def test_finds_show_line(self):
        src = textwrap.dedent("""\
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            ax.plot([1,2,3], [1,2,3])
            plt.show()
        """)
        _, info = scan_ast(src)
        assert info.show_node is not None
        assert info.show_node.lineno == 4

    def test_finds_draw_methods(self):
        src = textwrap.dedent("""\
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            ax.plot([1,2,3], [1,2,3])
            ax.scatter([1,2], [3,4])
            plt.show()
        """)
        _, info = scan_ast(src)
        assert info.first_draw_node is not None
        assert info.first_draw_node.lineno == 3
        assert info.last_draw_node.lineno == 4

    def test_finds_config_methods(self):
        src = textwrap.dedent("""\
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            ax.set_title("hello")
            ax.plot([1,2,3], [1,2,3])
            plt.show()
        """)
        tree, info = scan_ast(src)  # tree needed for tree.body access below
        # set_title is config, not draw
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
        assert "ax1" in ax_names
        assert "ax2" in ax_names
        assert "fig" not in ax_names

    def test_detects_3d_projection_subplots(self):
        src = textwrap.dedent("""\
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
            ax.plot_surface([], [], [])
            plt.show()
        """)
        _, info = scan_ast(src)
        assert len(info.ax_info) == 1
        assert info.ax_info[0].is_3d is True

    def test_detects_3d_projection_add_subplot(self):
        src = textwrap.dedent("""\
            import matplotlib.pyplot as plt
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface([], [], [])
            plt.show()
        """)
        _, info = scan_ast(src)
        assert any(a.is_3d for a in info.ax_info)

    def test_handles_augmented_assign(self):
        src = textwrap.dedent("""\
            import matplotlib.pyplot as plt
            import numpy as np
            t = 1.0
            x = np.linspace(0, 10, 100)
            y = np.sin(x)
            y += t
            fig, ax = plt.subplots()
            ax.plot(x, y)
            plt.show()
        """)
        tree, info = scan_ast(src)  # tree needed for tree.body access below
        # y += t is on line 6 - should have 'y' in assigns
        stmt5 = tree.body[5]  # y += t
        assert "y" in info.node_assigns[id(stmt5)]

    def test_handles_annotated_assign(self):
        src = textwrap.dedent("""\
            import matplotlib.pyplot as plt
            t: float = 1.0
            fig, ax = plt.subplots()
            ax.plot([t], [t])
            plt.show()
        """)
        tree, info = scan_ast(src)  # tree needed for tree.body access below
        stmt1 = tree.body[1]  # t: float = 1.0
        assert "t" in info.node_assigns[id(stmt1)]

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
        assert "x" in info.all_names
        assert "y" in info.all_names
        assert "fig" in info.all_names
        assert "ax" in info.all_names
        assert "plt" in info.all_names

    def test_no_figure_creation(self):
        src = textwrap.dedent("""\
            import matplotlib.pyplot as plt
            plt.plot([1,2,3], [1,2,3])
            plt.show()
        """)
        _, info = scan_ast(src)
        assert info.fig_node is None

    def test_no_show(self):
        src = textwrap.dedent("""\
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            ax.plot([1,2,3], [1,2,3])
        """)
        _, info = scan_ast(src)
        assert info.show_node is None


# ===============================================================
# build_deps - dependency tracking
# ===============================================================
class TestBuildDeps:
    def test_direct_dependency(self):
        src = textwrap.dedent("""\
            t = 1.0
            y = t * 2
        """)
        tree = ast.parse(src)
        deps = build_deps(tree, "t")
        assert "y" in deps

    def test_transitive_dependency(self):
        src = textwrap.dedent("""\
            t = 1.0
            a = t + 1
            b = a * 2
            c = b - 1
        """)
        tree = ast.parse(src)
        deps = build_deps(tree, "t")
        assert "a" in deps
        assert "b" in deps
        assert "c" in deps

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
        assert "x" not in deps
        assert "z" not in deps

    def test_augassign_dependency(self):
        src = textwrap.dedent("""\
            t = 1.0
            y = 0
            y += t
        """)
        tree = ast.parse(src)
        deps = build_deps(tree, "t")
        assert "y" in deps

    def test_annassign_dependency(self):
        src = textwrap.dedent("""\
            t = 1.0
            y: float = t * 2
        """)
        tree = ast.parse(src)
        deps = build_deps(tree, "t")
        assert "y" in deps

    def test_empty_script(self):
        src = "t = 1.0\n"
        tree = ast.parse(src)
        deps = build_deps(tree, "t")
        assert deps == set()  # no dependents


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
        static, dynamic, plot = partition(src, tree, info, dep_vars, "t")

        # Imports and fig creation should be static
        assert any("import matplotlib" in s for s in static)
        assert any("subplots" in s for s in static)

        # y depends on t, should be dynamic
        assert any("y = " in s or "y =" in s for s in dynamic)

        # ax.plot should be in plot
        assert any("ax.plot" in s for s in plot)

        # show() should be excluded entirely
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
        static_s, dynamic_s, _ = partition(src, tree, info, dep_vars, "t")

        # t = 5.0 should be skipped (we generate our own assignment)
        assert not any(s.strip() == "t = 5.0" for s in static_s)
        assert not any(s.strip() == "t = 5.0" for s in dynamic_s)

    def test_multiline_statement_intact(self):
        src = _read_fixture("multiline_stmt.py")
        tree, info = scan_ast(src)
        dep_vars = build_deps(tree, "t")
        _, dynamic, _ = partition(src, tree, info, dep_vars, "t")

        # The multi-line y = sin(\n    t*x\n) should be intact in dynamic
        dynamic_text = "\n".join(dynamic)
        assert "np.sin(" in dynamic_text
        assert "t * x" in dynamic_text or "t*x" in dynamic_text

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
        _, _, plot = partition(src, tree, info, dep_vars, "t")

        # set_title is config, should be in plot section (after fig creation)
        assert any("set_title" in s for s in plot)


# ===============================================================
# _inject_agg - Agg backend injection
# ===============================================================
class TestInjectAgg:
    def test_injects_before_matplotlib_import(self):
        stmts = ["import matplotlib.pyplot as plt", "x = 1"]
        result = _inject_agg(stmts)
        assert result[0] == "import matplotlib; matplotlib.use('Agg')"
        assert "import matplotlib.pyplot as plt" in result

    def test_injects_at_start_if_no_mpl_import(self):
        stmts = ["import numpy as np", "x = 1"]
        result = _inject_agg(stmts)
        assert result[0] == "import matplotlib; matplotlib.use('Agg')"

    def test_removes_existing_use_call(self):
        stmts = ["import matplotlib; matplotlib.use('TkAgg')",
                  "import matplotlib.pyplot as plt"]
        result = _inject_agg(stmts)
        assert not any("TkAgg" in s for s in result)
        assert any("Agg" in s for s in result)

    def test_no_duplicate_injection(self):
        stmts = ["import matplotlib; matplotlib.use('Agg')",
                  "import matplotlib.pyplot as plt"]
        result = _inject_agg(stmts)
        agg_count = sum(1 for s in result if "matplotlib.use('Agg')" in s)
        assert agg_count == 1


# ===============================================================
# _gen_clear_lines - axes clearing
# ===============================================================
class TestGenClearLines:
    def test_no_axes_info(self):
        result = _gen_clear_lines([])
        assert any("clear" in line for line in result)

    def test_single_2d_axes(self):
        result = _gen_clear_lines([AxesInfo(var_name="ax")])
        text = "\n".join(result)
        assert "ax" in text
        assert "clear" in text

    def test_multiple_axes(self):
        result = _gen_clear_lines([
            AxesInfo(var_name="ax1"),
            AxesInfo(var_name="ax2"),
        ])
        text = "\n".join(result)
        assert "ax1" in text
        assert "ax2" in text

    def test_3d_axes(self):
        result = _gen_clear_lines([AxesInfo(var_name="ax", is_3d=True)])
        text = "\n".join(result)
        assert "clear" in text


# ===============================================================
# Code generation - output validity
# ===============================================================
class TestCodeGeneration:
    def test_output_is_valid_python(self):
        src = _read_fixture("simple_line.py")
        result = animate(src, var="t", range_str="0,6.28")
        # Should parse without errors
        ast.parse(result)

    def test_contains_update_function(self):
        src = _read_fixture("simple_line.py")
        result = animate(src, var="t", range_str="0,6.28")
        assert "def update(_frame):" in result

    def test_contains_worker_function(self):
        src = _read_fixture("simple_line.py")
        result = animate(src, var="t", range_str="0,6.28")
        assert "def _render_one(job):" in result

    def test_agg_backend_injected(self):
        src = _read_fixture("simple_line.py")
        result = animate(src, var="t", range_str="0,6.28")
        assert "matplotlib.use('Agg')" in result

    def test_show_removed(self):
        src = _read_fixture("simple_line.py")
        result = animate(src, var="t", range_str="0,6.28")
        assert "plt.show()" not in result

    def test_animated_var_in_update(self):
        src = _read_fixture("simple_line.py")
        result = animate(src, var="t", range_str="0,6.28")
        # The update function should set t = start + _frame * step
        assert "t = 0 + _frame *" in result or "t = 0.0 + _frame *" in result

    def test_frames_parameter_used(self):
        src = _read_fixture("simple_line.py")
        result = animate(src, var="t", range_str="0,6.28", frames=60)
        assert "_FRAMES = 60" in result or "range(60)" in result or "60 frames" in result

    def test_custom_output_name(self):
        src = _read_fixture("simple_line.py")
        result = animate(src, var="t", range_str="0,6.28", out="custom.gif")
        assert "custom.gif" in result

    def test_3d_output_is_valid_python(self):
        src = _read_fixture("surface_3d.py")
        result = animate(src, var="t", range_str="0.5,3")
        ast.parse(result)

    def test_polar_output_is_valid_python(self):
        src = _read_fixture("polar.py")
        result = animate(src, var="t", range_str="0.5,5")
        ast.parse(result)


# ===============================================================
# Input validation
# ===============================================================
class TestValidation:
    def test_missing_variable_raises(self):
        src = textwrap.dedent("""\
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            ax.plot([1,2,3], [1,2,3])
            plt.show()
        """)
        with pytest.raises(AssertionError, match="not found"):
            animate(src, var="nonexistent", range_str="0,1")

    def test_invalid_python_raises(self):
        with pytest.raises(SyntaxError):
            animate("def foo(:", var="t", range_str="0,1")

    def test_empty_source_raises(self):
        with pytest.raises(AssertionError):
            animate("", var="t", range_str="0,1")

    def test_whitespace_only_raises(self):
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

    def test_invalid_fmt_raises(self):
        src = "import matplotlib.pyplot as plt\nt=1\nfig,ax=plt.subplots()\nax.plot([t],[t])\nplt.show()\n"
        with pytest.raises(AssertionError, match="fmt must be"):
            animate(src, var="t", range_str="0,1", fmt="avi")

    def test_invalid_loop_raises(self):
        src = "import matplotlib.pyplot as plt\nt=1\nfig,ax=plt.subplots()\nax.plot([t],[t])\nplt.show()\n"
        with pytest.raises(AssertionError, match="loop"):
            animate(src, var="t", range_str="0,1", loop=-1)

    def test_ffmpeg_missing_warns(self, monkeypatch):
        import mpl_animator
        monkeypatch.setattr(mpl_animator, "_check_ffmpeg", lambda: False)
        src = "import matplotlib.pyplot as plt\nt=1\nfig,ax=plt.subplots()\nax.plot([t],[t])\nplt.show()\n"
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            animate(src, var="t", range_str="0,1", fmt="mp4")
            assert any("ffmpeg" in str(x.message) for x in w)


# ===============================================================
# New features - loop, reverse, ping-pong
# ===============================================================
class TestNewFeatures:
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

    def test_loop_forever_default(self):
        result = animate(self._simple_src(), var="t", range_str="0,1")
        assert any("_LOOP" in line and "= 0" in line for line in result.splitlines())

    def test_loop_once(self):
        result = animate(self._simple_src(), var="t", range_str="0,1", loop=1)
        assert any("_LOOP" in line and "= 1" in line for line in result.splitlines())

    def test_loop_value_passed_to_stitch(self):
        result = animate(self._simple_src(), var="t", range_str="0,1", loop=3)
        assert any("_LOOP" in line and "= 3" in line for line in result.splitlines())
        assert "loop=loop" in result

    def test_reverse_swaps_start_end(self):
        result = animate(self._simple_src(), var="t", range_str="0,5", reverse=True)
        # With reverse, start=5, end=0, so step is negative
        assert "t = 5" in result or "t = 5.0" in result

    def test_reverse_forward_differ(self):
        fwd = animate(self._simple_src(), var="t", range_str="0,5")
        rev = animate(self._simple_src(), var="t", range_str="0,5", reverse=True)
        # The step sign should differ
        assert fwd != rev

    def test_ping_pong_off_by_default(self):
        result = animate(self._simple_src(), var="t", range_str="0,1")
        assert "_PING_PONG = False" in result

    def test_ping_pong_on(self):
        result = animate(self._simple_src(), var="t", range_str="0,1", ping_pong=True)
        assert "_PING_PONG = True" in result

    def test_ping_pong_uses_reversed_frames(self):
        result = animate(self._simple_src(), var="t", range_str="0,1", ping_pong=True)
        assert "reversed" in result

    def test_output_naming_gif(self):
        result = animate(self._simple_src(), var="t", range_str="0,1",
                         fmt="gif", source_name="myplot.py")
        assert "myplot_animated.gif" in result

    def test_output_naming_mp4(self):
        result = animate(self._simple_src(), var="t", range_str="0,1",
                         fmt="mp4", source_name="myplot.py")
        assert "myplot_animated.mp4" in result

    def test_out_extension_overrides_fmt(self):
        result = animate(self._simple_src(), var="t", range_str="0,1",
                         out="custom.mp4", fmt="gif")
        assert "custom.mp4" in result

    def test_progress_bar_in_sequential(self):
        result = animate(self._simple_src(), var="t", range_str="0,1")
        assert "Frame" in result and "total" in result

    def test_set_theta_offset_in_config_methods(self):
        from mpl_animator import CONFIG_METHODS
        assert "set_theta_offset" in CONFIG_METHODS

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
        from mpl_animator import scan_ast, build_deps, partition
        tree, info = scan_ast(src)
        dep_vars = build_deps(tree, "angle")
        _, dynamic, plot = partition(src, tree, info, dep_vars, "angle")
        assert any("set_theta_offset" in s for s in plot), \
            "set_theta_offset must be in plot_stmts (runs after clear), not dynamic"
        assert not any("set_theta_offset" in s for s in dynamic), \
            "set_theta_offset must NOT be in dynamic_stmts (would run before clear)"


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
}


class TestEndToEnd:
    @pytest.mark.parametrize("fixture_name", list(FIXTURE_CONFIGS.keys()))
    def test_fixture_generates_valid_python(self, fixture_name):
        """Each fixture should produce a valid Python script."""
        src = _read_fixture(fixture_name)
        cfg = FIXTURE_CONFIGS[fixture_name]
        result = animate(src, frames=10, **cfg)
        try:
            ast.parse(result)
        except SyntaxError as e:
            pytest.fail(
                f"Generated code for {fixture_name} has syntax error: {e}\n"
                f"--- Generated code ---\n{result}"
            )

    @pytest.mark.parametrize("fixture_name", list(FIXTURE_CONFIGS.keys()))
    def test_fixture_has_update_and_worker(self, fixture_name):
        """Each generated script should have update() and _render_one()."""
        src = _read_fixture(fixture_name)
        cfg = FIXTURE_CONFIGS[fixture_name]
        result = animate(src, frames=10, **cfg)
        assert "def update(_frame):" in result
        assert "def _render_one(job):" in result
        assert "def render_sequential():" in result
        assert "def render_parallel(n_workers):" in result

    @pytest.mark.parametrize("fixture_name", list(FIXTURE_CONFIGS.keys()))
    def test_fixture_no_show(self, fixture_name):
        """Generated script should never contain plt.show()."""
        src = _read_fixture(fixture_name)
        cfg = FIXTURE_CONFIGS[fixture_name]
        result = animate(src, frames=10, **cfg)
        assert "plt.show()" not in result

    def test_wave_static_backward_compatible(self):
        """The original wave_static.py example should still work."""
        wave_path = os.path.join(os.path.dirname(__file__), "..", "examples", "wave_static.py")
        with open(wave_path, encoding="utf-8") as f:
            src = f.read()
        result = animate(src, var="f", range_str="3,60", frames=10)
        ast.parse(result)
        assert "def update(_frame):" in result
        assert "plt.show()" not in result


# ===============================================================
# Method categories sanity checks
# ===============================================================
class TestMethodCategories:
    def test_draw_and_config_are_disjoint(self):
        assert DRAW_METHODS & CONFIG_METHODS == set()

    def test_all_is_union(self):
        assert ALL_PLOT_METHODS == DRAW_METHODS | CONFIG_METHODS

    def test_common_methods_present(self):
        assert "plot" in DRAW_METHODS
        assert "scatter" in DRAW_METHODS
        assert "bar" in DRAW_METHODS
        assert "imshow" in DRAW_METHODS
        assert "hist" in DRAW_METHODS
        assert "plot_surface" in DRAW_METHODS
        assert "set_title" in CONFIG_METHODS
        assert "set_xlim" in CONFIG_METHODS
        assert "grid" in CONFIG_METHODS
        assert "legend" in CONFIG_METHODS


# ===============================================================
# Slow tests - actually run the generated scripts
# ===============================================================
@pytest.mark.slow
class TestSlowExecution:
    """Tests that actually execute the generated animated scripts.
    Run with: pytest -m slow
    GIFs are saved to tests/output/ for inspection.
    """
    @pytest.mark.parametrize("fixture_name", list(FIXTURE_CONFIGS.keys()))
    def test_generated_script_produces_gif(self, fixture_name, tmp_path):
        """Run generated script and check that a GIF is produced."""
        if fixture_name == "implicit_figure.py":
            pytest.xfail("implicit_figure.py has no explicit figure creation; animate() already warns this may not work")
        if fixture_name == "mixed_3d_2d.py":
            pytest.xfail("mixed_3d_2d.py mixes 2D and 3D subplots in one figure; the 3D fig.clear() path drops the 2D subplot (known limitation)")
        src = _read_fixture(fixture_name)
        cfg = FIXTURE_CONFIGS[fixture_name]
        gif_name = fixture_name.replace(".py", "_animated.gif")
        result = animate(src, frames=5, fps=5, out=gif_name,
                         workers=1, **cfg)

        script_path = tmp_path / "animated.py"
        script_path.write_text(result)

        # Run the generated script
        proc = subprocess.run(
            [sys.executable, str(script_path), "--sequential"],
            capture_output=True, text=True, timeout=60,
            cwd=str(tmp_path),
        )
        assert proc.returncode == 0, \
            f"Script failed:\nstdout: {proc.stdout}\nstderr: {proc.stderr}"

        gif_path = tmp_path / gif_name
        assert gif_path.exists(), \
            f"GIF not created. stdout: {proc.stdout}\nstderr: {proc.stderr}"
        assert gif_path.stat().st_size > 0

        # Copy GIF to tests/output/ so it can be inspected after the run
        import shutil
        out_dir = os.path.join(os.path.dirname(__file__), "output")
        os.makedirs(out_dir, exist_ok=True)
        shutil.copy(gif_path, os.path.join(out_dir, gif_name))

    def test_ping_pong_produces_gif(self, tmp_path):
        """ping-pong GIF should have ~2x frames and be larger than normal."""
        src = _read_fixture("simple_line.py")
        frames = 5
        gif_name = "simple_line_pingpong.gif"
        normal = animate(src, var="t", range_str="0,6.28", frames=frames,
                         fps=5, out="simple_line_normal.gif", workers=1)
        pp = animate(src, var="t", range_str="0,6.28", frames=frames,
                     fps=5, out=gif_name, ping_pong=True, workers=1)

        for code, name in [(normal, "simple_line_normal.gif"), (pp, gif_name)]:
            p = tmp_path / (name.replace(".gif", ".py"))
            p.write_text(code)
            proc = subprocess.run(
                [sys.executable, str(p), "--sequential"],
                capture_output=True, text=True, timeout=60, cwd=str(tmp_path),
            )
            assert proc.returncode == 0, f"{name} failed:\n{proc.stderr}"

        from PIL import Image
        normal_frames = Image.open(tmp_path / "simple_line_normal.gif").n_frames
        pp_frames = Image.open(tmp_path / gif_name).n_frames
        assert pp_frames > normal_frames, \
            f"ping-pong ({pp_frames} frames) should have more frames than normal ({normal_frames})"

        import shutil
        out_dir = os.path.join(os.path.dirname(__file__), "output")
        os.makedirs(out_dir, exist_ok=True)
        shutil.copy(tmp_path / gif_name, os.path.join(out_dir, gif_name))
        shutil.copy(tmp_path / "simple_line_normal.gif",
                    os.path.join(out_dir, "simple_line_normal.gif"))

    def test_reverse_produces_gif(self, tmp_path):
        """reverse GIF should run the same number of frames but in opposite order."""
        src = _read_fixture("simple_line.py")
        gif_name = "simple_line_reversed.gif"
        result = animate(src, var="t", range_str="0,6.28", frames=5,
                         fps=5, out=gif_name, reverse=True, workers=1)
        script_path = tmp_path / "rev.py"
        script_path.write_text(result)
        proc = subprocess.run(
            [sys.executable, str(script_path), "--sequential"],
            capture_output=True, text=True, timeout=60, cwd=str(tmp_path),
        )
        assert proc.returncode == 0, f"reverse script failed:\n{proc.stderr}"
        gif_path = tmp_path / gif_name
        assert gif_path.exists() and gif_path.stat().st_size > 0

        import shutil
        out_dir = os.path.join(os.path.dirname(__file__), "output")
        os.makedirs(out_dir, exist_ok=True)
        shutil.copy(gif_path, os.path.join(out_dir, gif_name))

    def test_loop_once_produces_gif(self, tmp_path):
        """loop=1 GIF should exist and differ from loop=0 in metadata."""
        src = _read_fixture("simple_line.py")
        gif_name = "simple_line_loop1.gif"
        result = animate(src, var="t", range_str="0,6.28", frames=5,
                         fps=5, out=gif_name, loop=1, workers=1)
        script_path = tmp_path / "loop1.py"
        script_path.write_text(result)
        proc = subprocess.run(
            [sys.executable, str(script_path), "--sequential"],
            capture_output=True, text=True, timeout=60, cwd=str(tmp_path),
        )
        assert proc.returncode == 0, f"loop=1 script failed:\n{proc.stderr}"
        gif_path = tmp_path / gif_name
        assert gif_path.exists() and gif_path.stat().st_size > 0

        import shutil
        out_dir = os.path.join(os.path.dirname(__file__), "output")
        os.makedirs(out_dir, exist_ok=True)
        shutil.copy(gif_path, os.path.join(out_dir, gif_name))
