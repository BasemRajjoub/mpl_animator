"""Discovery script: test mpl_animator against all matplotlib gallery examples."""
import ast
import json
import os
import sys
import re

# Add parent dir to path so we can import mpl_animator
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mpl_animator import animate

GALLERY_DIR = os.path.join(os.path.dirname(__file__), "gallery_examples")

PREFERRED_VARS = [
    "n", "t", "x", "freq", "alpha", "scale", "angle", "delta", "sigma",
    "r", "k", "a", "b", "c", "width", "size", "N", "theta", "phi",
    "amplitude", "offset", "factor", "threshold", "level", "ratio",
    "density", "spacing", "margin", "padding", "radius", "height",
    "length", "depth", "count", "step", "rate", "speed", "dt", "dx",
    "dy", "dz", "mu", "std", "mean", "var", "lw", "linewidth",
    "fontsize", "pad", "gap", "shift", "rotation", "elevation",
    "azimuth", "roll", "zoom",
]


def preprocess(src):
    """Remove savefig lines, truncate after last plt.show(), strip RST blocks."""
    lines = [l for l in src.splitlines() if "savefig" not in l]
    src2 = "\n".join(lines)
    if "plt.show()" in src2:
        idx = src2.rfind("plt.show()")
        src2 = src2[:idx + len("plt.show()")]
    else:
        src2 += "\nplt.show()\n"
    return src2


def should_skip(filepath, src):
    """Return (should_skip: bool, reason: str)."""
    basename = os.path.basename(filepath)

    if "sgskip" in basename:
        return True, "sgskip in filename"

    if "FuncAnimation" in src or "ArtistAnimation" in src:
        return True, "uses FuncAnimation/ArtistAnimation"

    if "matplotlib.widgets" in src or "from matplotlib.widgets" in src:
        return True, "uses matplotlib widgets"

    # Event handling scripts with connect()
    if ".mpl_connect(" in src or "fig.canvas.mpl_connect" in src:
        return True, "uses event handling (mpl_connect)"

    # Scripts that use plt.connect or button callbacks
    if "Button(" in src or "Slider(" in src or "RadioButtons(" in src:
        return True, "uses interactive widgets"

    if "plt.ginput" in src:
        return True, "uses ginput (interactive)"

    # Scripts that define classes (complex OOP patterns)
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return True, "SyntaxError in source"

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            return True, "contains class definition (complex pattern)"

    return False, ""


def _try_eval_numeric(node):
    """Try to evaluate an AST node as a numeric constant.

    Handles: literals, -literal, simple np.pi/math.pi expressions,
    and basic arithmetic of constants.
    Returns float/int or None if not evaluable.
    """
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        return None

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = _try_eval_numeric(node.operand)
        if inner is not None:
            return -inner
        return None

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd):
        return _try_eval_numeric(node.operand)

    # Handle np.pi, math.pi, etc.
    if isinstance(node, ast.Attribute):
        if isinstance(node.value, ast.Name) and node.attr == "pi":
            if node.value.id in ("np", "numpy", "math"):
                return 3.141592653589793
        if isinstance(node.value, ast.Name) and node.attr == "e":
            if node.value.id in ("np", "numpy", "math"):
                return 2.718281828459045
        return None

    # Handle simple BinOps like 2 * np.pi
    if isinstance(node, ast.BinOp):
        left = _try_eval_numeric(node.left)
        right = _try_eval_numeric(node.right)
        if left is not None and right is not None:
            try:
                if isinstance(node.op, ast.Add):
                    return left + right
                elif isinstance(node.op, ast.Sub):
                    return left - right
                elif isinstance(node.op, ast.Mult):
                    return left * right
                elif isinstance(node.op, ast.Div) and right != 0:
                    return left / right
                elif isinstance(node.op, ast.Pow):
                    return left ** right
            except Exception:
                pass
        return None

    return None


def find_animatable_var(src):
    """Find a suitable numeric variable to animate.

    Returns (var_name, value) or (None, None).
    """
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None, None

    candidates = {}  # name -> value

    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                val = _try_eval_numeric(node.value)
                if val is not None:
                    candidates[target.id] = val

    if candidates:
        # Prefer variables from our preferred list
        for pref in PREFERRED_VARS:
            if pref in candidates:
                return pref, candidates[pref]

        # Fallback: pick any numeric variable (prefer float, then int)
        float_vars = {k: v for k, v in candidates.items() if isinstance(v, float)}
        if float_vars:
            name = next(iter(float_vars))
            return name, float_vars[name]

        int_vars = {k: v for k, v in candidates.items() if isinstance(v, int) and v != 0}
        if int_vars:
            name = next(iter(int_vars))
            return name, int_vars[name]

        # Any numeric candidate
        name = next(iter(candidates))
        return name, candidates[name]

    # Fallback: look for preferred variable names that exist in the script
    # even if they're not assigned to simple numeric literals.
    # These may be assigned to arrays, function calls, etc. -- the animator
    # can still handle them since it only needs the name to exist in the AST.
    all_names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    for pref in PREFERRED_VARS:
        if pref in all_names:
            return pref, 1.0  # default value, range will be 0.5,3

    return None, None


def make_range_str(val):
    """Create a sensible range string for a given numeric value."""
    if val == 0:
        return "0.1,2"
    if abs(val) > 100:
        lo = val * 0.8
        hi = val * 1.2
        if lo == hi:
            return f"{val - 1},{val + 1}"
        return f"{lo},{hi}"
    lo = val * 0.5
    hi = val * 3 if val > 0 else val * 0.3
    if lo == hi:
        return f"{val - 1},{val + 1}"
    # Make sure lo != hi
    if lo == hi:
        return "0.1,2"
    return f"{lo},{hi}"


def run_discovery():
    results = []
    total = 0
    passed = 0
    skipped = 0
    failed = 0

    for root, dirs, files in os.walk(GALLERY_DIR):
        dirs.sort()
        for fname in sorted(files):
            if not fname.endswith(".py"):
                continue
            total += 1
            filepath = os.path.join(root, fname)
            rel_path = os.path.relpath(filepath, GALLERY_DIR)
            # Normalize path separators
            rel_path = rel_path.replace("\\", "/")

            try:
                with open(filepath, encoding="utf-8") as f:
                    raw_src = f.read()
            except Exception as e:
                results.append({
                    "file": rel_path,
                    "var": None,
                    "range": None,
                    "status": "SKIP",
                    "error": f"Could not read file: {e}"
                })
                skipped += 1
                continue

            skip, reason = should_skip(filepath, raw_src)
            if skip:
                results.append({
                    "file": rel_path,
                    "var": None,
                    "range": None,
                    "status": "SKIP",
                    "error": reason
                })
                skipped += 1
                continue

            src = preprocess(raw_src)

            var_name, var_val = find_animatable_var(src)
            if var_name is None:
                results.append({
                    "file": rel_path,
                    "var": None,
                    "range": None,
                    "status": "SKIP",
                    "error": "no suitable numeric variable found"
                })
                skipped += 1
                continue

            range_str = make_range_str(var_val)

            try:
                result_code = animate(
                    src,
                    var=var_name,
                    range_str=range_str,
                    frames=5,
                )
            except AssertionError as e:
                err_msg = str(e)
                if "not found" in err_msg:
                    status = "SKIP"
                    skipped += 1
                elif "no drawing" in err_msg.lower():
                    status = "SKIP"
                    skipped += 1
                else:
                    status = "FAIL"
                    failed += 1
                results.append({
                    "file": rel_path,
                    "var": var_name,
                    "range": range_str,
                    "status": status,
                    "error": f"AssertionError: {err_msg}"
                })
                continue
            except Exception as e:
                results.append({
                    "file": rel_path,
                    "var": var_name,
                    "range": range_str,
                    "status": "FAIL",
                    "error": f"{type(e).__name__}: {e}"
                })
                failed += 1
                continue

            # Verify the generated code is valid Python
            try:
                ast.parse(result_code)
            except SyntaxError as e:
                results.append({
                    "file": rel_path,
                    "var": var_name,
                    "range": range_str,
                    "status": "FAIL",
                    "error": f"SyntaxError in generated code: {e}"
                })
                failed += 1
                continue

            # Check for update function
            if "def update(_frame):" not in result_code:
                results.append({
                    "file": rel_path,
                    "var": var_name,
                    "range": range_str,
                    "status": "FAIL",
                    "error": "Missing update function"
                })
                failed += 1
                continue

            results.append({
                "file": rel_path,
                "var": var_name,
                "range": range_str,
                "status": "PASS",
                "error": None
            })
            passed += 1

    # Save results
    results_file = os.path.join(os.path.dirname(__file__), "gallery_test_results.json")
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Gallery Discovery Results")
    print(f"{'='*60}")
    print(f"Total scripts:  {total}")
    print(f"Skipped:        {skipped}")
    print(f"PASS:           {passed}")
    print(f"FAIL:           {failed}")
    print(f"{'='*60}")

    # Show failure breakdown
    fail_results = [r for r in results if r["status"] == "FAIL"]
    if fail_results:
        print(f"\nFailure breakdown:")
        error_types = {}
        for r in fail_results:
            # Categorize error
            err = r["error"]
            if "SyntaxError" in err:
                key = "SyntaxError in generated code"
            elif "AssertionError" in err:
                key = "AssertionError"
            else:
                # Extract error class
                key = err.split(":")[0] if ":" in err else err[:60]
            error_types.setdefault(key, []).append(r)

        for etype, items in sorted(error_types.items()):
            print(f"\n  {etype}: {len(items)} files")
            for item in items[:5]:  # Show first 5 of each type
                print(f"    - {item['file']} (var={item['var']})")
                print(f"      {item['error'][:120]}")
            if len(items) > 5:
                print(f"    ... and {len(items)-5} more")

    return results


if __name__ == "__main__":
    run_discovery()
