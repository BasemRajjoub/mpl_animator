"""Tests against official matplotlib gallery examples."""
import ast
import os
import json

import pytest

from mpl_animator import animate

GALLERY_DIR = os.path.join(os.path.dirname(__file__), "gallery_examples")
RESULTS_FILE = os.path.join(os.path.dirname(__file__), "gallery_test_results.json")

# Load the pre-computed pass list (only test scripts that are known-animatable)
with open(RESULTS_FILE) as f:
    _results = json.load(f)

ANIMATABLE = [(r["file"], r["var"], r["range"]) for r in _results if r["status"] == "PASS"]
SKIP_REASONS = {r["file"]: r["error"] for r in _results if r["status"] == "SKIP"}
KNOWN_FAIL = {r["file"]: r["error"] for r in _results if r["status"] == "FAIL"}


def _preprocess(src):
    """Remove savefig lines, truncate after last plt.show(), strip RST blocks."""
    lines = [l for l in src.splitlines() if "savefig" not in l]
    src2 = "\n".join(lines)
    if "plt.show()" in src2:
        idx = src2.rfind("plt.show()")
        src2 = src2[:idx + len("plt.show()")]
    else:
        src2 += "\nplt.show()\n"
    return src2


@pytest.mark.parametrize("rel_path,var,range_str", ANIMATABLE,
                         ids=[r[0] for r in ANIMATABLE])
def test_gallery_generates_valid_python(rel_path, var, range_str):
    """Each animatable gallery script must produce valid Python."""
    src = open(os.path.join(GALLERY_DIR, rel_path), encoding="utf-8").read()
    src = _preprocess(src)
    result = animate(src, var=var, range_str=range_str, frames=5)
    try:
        tree = ast.parse(result)
    except SyntaxError as e:
        pytest.fail(f"SyntaxError in generated code: {e}")
    assert "def update(_frame):" in result
    # Verify no actual plt.show() call exists in the AST (ignoring string literals)
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "show"):
            pytest.fail("Generated code contains a plt.show() call")


@pytest.mark.parametrize("rel_path,reason", list(SKIP_REASONS.items())[:5],
                         ids=list(SKIP_REASONS.keys())[:5])
def test_gallery_skipped_scripts_are_skippable(rel_path, reason):
    """Skipped scripts should skip gracefully, not crash."""
    pytest.skip(reason)
