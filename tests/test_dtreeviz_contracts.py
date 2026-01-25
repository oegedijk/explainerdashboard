import inspect

import dtreeviz.trees as trees


def test_dtreeviz_view_signature_includes_x():
    sig = inspect.signature(trees.DTreeVizAPI.view)
    assert "x" in sig.parameters
