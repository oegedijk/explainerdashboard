import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.metrics import roc_auc_score
from sklearn.metrics import r2_score

from explainerdashboard.explainer_methods import (
    align_categorical_dtypes,
    get_pdp_df,
    make_one_vs_all_scorer,
    permutation_importances,
)


class DtypeCheckingModel(RegressorMixin, BaseEstimator):
    def __init__(self, expected_dtypes: pd.Series):
        self.expected_dtypes = expected_dtypes
        self.feature_names_in_ = expected_dtypes.index.to_numpy()

    def predict(self, X):
        if isinstance(X, np.ndarray):
            raise ValueError("Expected DataFrame input to preserve dtypes")
        if not X.dtypes.equals(self.expected_dtypes):
            raise ValueError("Dtype mismatch")
        return np.zeros(len(X))


class DtypeCheckingClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, expected_dtypes: pd.Series):
        self.expected_dtypes = expected_dtypes
        self.feature_names_in_ = expected_dtypes.index.to_numpy()

    def predict_proba(self, X):
        if isinstance(X, np.ndarray):
            raise ValueError("Expected DataFrame input to preserve dtypes")
        if not X.dtypes.equals(self.expected_dtypes):
            raise ValueError("Dtype mismatch")
        return np.tile(np.array([0.2, 0.8]), (len(X), 1))


class DataFrameProbaClassifier(ClassifierMixin, BaseEstimator):
    def predict_proba(self, X):
        probs = np.tile(np.array([0.2, 0.8]), (len(X), 1))
        return pd.DataFrame(probs, index=getattr(X, "index", None), columns=[0, 1])


def test_permutation_importances_preserves_categorical_dtypes():
    X = pd.DataFrame(
        {
            "cat": pd.Series(["a", "b", "a", "b"], dtype="category"),
            "num": pd.Series([1, 2, 3, 4], dtype="int64"),
        }
    )
    y = pd.Series([0.1, 0.2, 0.3, 0.4])
    model = DtypeCheckingModel(X.dtypes)

    importances = permutation_importances(model, X, y, metric=r2_score, n_repeats=1)

    assert importances["Importance"].notnull().all()


def test_get_pdp_df_handles_boolean_onehot():
    X = pd.DataFrame(
        {
            "color_red": pd.Series([True, False, True], dtype="bool"),
            "color_blue": pd.Series([False, True, False], dtype="bool"),
        }
    )
    model = DtypeCheckingClassifier(X.dtypes)

    pdp_df = get_pdp_df(
        model=model,
        X_sample=X,
        feature=["color_red", "color_blue"],
        grid_values=["color_red", "color_blue"],
        is_classifier=True,
    )

    assert isinstance(pdp_df, pd.DataFrame)
    assert list(pdp_df.columns) == ["color_red", "color_blue"]


def test_align_categorical_dtypes_matches_reference():
    reference = pd.DataFrame({"cat": pd.Series(["a", "b", "a"], dtype="category")})
    target = pd.DataFrame({"cat": pd.Series(["a", "b", "a"], dtype="object")})

    aligned = align_categorical_dtypes(target, reference)

    assert aligned["cat"].dtype == reference["cat"].dtype


def test_make_one_vs_all_scorer_accepts_dataframe_predict_proba():
    scorer = make_one_vs_all_scorer(roc_auc_score, pos_label=1)
    clf = DataFrameProbaClassifier()
    X = pd.DataFrame({"feature": [0, 1, 2, 3]})
    y = np.array([0, 1, 0, 1])

    score = scorer(clf, X, y)

    assert isinstance(score, float)
