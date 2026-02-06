import numpy as np
import pandas as pd
import pytest

from explainerdashboard import ClassifierExplainer
from explainerdashboard.explainer_plots import (
    plotly_actual_vs_col,
    plotly_preds_vs_col,
    plotly_residuals_vs_col,
    plotly_shap_violin_plot,
)


class NoClassesModel:
    """Classifier-like model without classes_ attribute."""

    def predict_proba(self, X):
        n = len(X)
        p = np.full(n, 0.6)
        return np.c_[1 - p, p]

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)


@pytest.mark.parametrize(
    "plot_fn,args",
    [
        (
            plotly_shap_violin_plot,
            (
                pd.Series(["a", 1, np.nan, "b"], name="mixed_cat", dtype=object),
                np.array([0.1, -0.2, 0.0, 0.3]),
            ),
        ),
        (
            plotly_residuals_vs_col,
            (
                np.array([1.0, 2.0, 3.0, 4.0]),
                np.array([1.1, 1.9, 2.8, 4.2]),
                pd.Series(["a", 1, np.nan, "b"], name="mixed_cat", dtype=object),
            ),
        ),
        (
            plotly_actual_vs_col,
            (
                np.array([1.0, 2.0, 3.0, 4.0]),
                np.array([1.1, 1.9, 2.8, 4.2]),
                pd.Series(["a", 1, np.nan, "b"], name="mixed_cat", dtype=object),
            ),
        ),
        (
            plotly_preds_vs_col,
            (
                np.array([1.0, 2.0, 3.0, 4.0]),
                np.array([1.1, 1.9, 2.8, 4.2]),
                pd.Series(["a", 1, np.nan, "b"], name="mixed_cat", dtype=object),
            ),
        ),
    ],
)
def test_plot_functions_do_not_crash_on_mixed_categorical_values(plot_fn, args):
    try:
        plot_fn(*args)
    except TypeError as exc:
        pytest.fail(f"Mixed categorical sorting should not crash: {exc}")


def test_classifier_explainer_without_classes_handles_mixed_target_labels():
    X = pd.DataFrame({"a": [1, 2, 3, 4], "b": [0, 1, 0, 1]})
    y = pd.Series([0, "1", 0, "1"], dtype=object)

    try:
        explainer = ClassifierExplainer(NoClassesModel(), X, y, shap="kernel")
    except TypeError as exc:
        pytest.fail(f"Mixed target-label sorting should not crash: {exc}")

    assert len(explainer.labels) == 2
