import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from explainerdashboard import ClassifierExplainer


class CategoricalModelWrapper:
    def __init__(self, model) -> None:
        self._model = model

    def _preprocessor(self, X):
        return X.drop(["Name"], axis=1)

    def predict(self, X):
        X = self._preprocessor(X)
        return self._model.predict(X)

    def predict_proba(self, X):
        X = self._preprocessor(X)
        return self._model.predict_proba(X)


class MixedCategoricalModelWrapper:
    def __init__(self, model) -> None:
        self._model = model

    def _preprocessor(self, X):
        return X.drop(["MixedCat"], axis=1)

    def predict(self, X):
        X = self._preprocessor(X)
        return self._model.predict(X)

    def predict_proba(self, X):
        X = self._preprocessor(X)
        return self._model.predict_proba(X)


@pytest.fixture(scope="module")
def categorical_nan_explainer(classifier_data):
    X_train, y_train, X_test, y_test = classifier_data
    X_train = X_train.copy()
    X_test = X_test.copy()

    X_train["Name"] = X_train.index
    X_test["Name"] = X_test.index
    X_test.iloc[:5, X_test.columns.get_loc("Name")] = np.nan

    model = RandomForestClassifier(n_estimators=5, max_depth=2, random_state=42)
    model.fit(X_train.drop(["Name"], axis=1), y_train)

    wrapper = CategoricalModelWrapper(model)
    explainer = ClassifierExplainer(
        wrapper, X_test, y_test, cats=["Sex", "Deck", "Embarked"]
    )
    return explainer, X_test


def test_categorical_dict_adds_nan_option(categorical_nan_explainer):
    explainer, _ = categorical_nan_explainer
    assert "NaN" in explainer.categorical_dict["Name"]


def test_get_row_from_input_preserves_nan_merged(categorical_nan_explainer):
    explainer, X_test = categorical_nan_explainer

    row = explainer.X_merged.iloc[0].copy()
    row["Name"] = np.nan
    input_row = explainer.get_row_from_input(row.values.tolist())
    assert pd.isna(input_row.loc[0, "Name"])

    row = explainer.X_merged.iloc[0].copy()
    row["Name"] = "NaN"
    input_row = explainer.get_row_from_input(row.values.tolist())
    assert pd.isna(input_row.loc[0, "Name"])


def test_get_row_from_input_preserves_nan_unmerged(categorical_nan_explainer):
    explainer, _ = categorical_nan_explainer

    row = explainer.X.iloc[0].copy()
    row["Name"] = np.nan
    input_row = explainer.get_row_from_input(row.values.tolist())
    assert pd.isna(input_row.loc[0, "Name"])

    row = explainer.X.iloc[0].copy()
    row["Name"] = "NaN"
    input_row = explainer.get_row_from_input(row.values.tolist())
    assert pd.isna(input_row.loc[0, "Name"])


def test_mixed_type_categorical_with_nan_does_not_crash_sorting(classifier_data):
    X_train, y_train, X_test, y_test = classifier_data
    X_train = X_train.copy()
    X_test = X_test.copy()

    X_train["MixedCat"] = pd.Series(
        ["alpha"] * len(X_train), index=X_train.index, dtype=object
    )
    X_test["MixedCat"] = pd.Series(
        ["alpha"] * len(X_test), index=X_test.index, dtype=object
    )

    # Simulate mixed-type categorical values from upstream preprocessing artifacts.
    X_train.loc[X_train.index[:8], "MixedCat"] = 1
    X_train.loc[X_train.index[8:16], "MixedCat"] = np.nan
    X_test.loc[X_test.index[:4], "MixedCat"] = 0
    X_test.loc[X_test.index[4:8], "MixedCat"] = np.nan

    model = RandomForestClassifier(n_estimators=5, max_depth=2, random_state=42)
    model.fit(X_train.drop(["MixedCat"], axis=1), y_train)
    wrapper = MixedCategoricalModelWrapper(model)

    try:
        ClassifierExplainer(
            wrapper,
            X_test,
            y_test,
            cats=["Sex", "Deck", "Embarked"],
            shap="kernel",
        )
    except TypeError as exc:
        pytest.fail(
            f"Categorical sorting should not crash on mixed values + NaN: {exc}"
        )
