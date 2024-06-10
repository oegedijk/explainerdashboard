import pytest

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from torch import nn

from skorch import NeuralNetClassifier, NeuralNetRegressor

from explainerdashboard.explainers import RegressionExplainer, ClassifierExplainer


@pytest.fixture(scope="session")
def skorch_regressor():
    X, y = make_regression(100, 5, n_informative=3, random_state=0)
    X = X.astype(np.float32)
    y = y / np.std(y)
    y = y.reshape(-1, 1).astype(np.float32)

    X_df = pd.DataFrame(X, columns=["col" + str(i) for i in range(X.shape[1])])

    class MyModule(nn.Module):
        def __init__(
            skorch_classifier_explainer, input_units=5, num_units=5, nonlin=nn.ReLU()
        ):
            super(MyModule, skorch_classifier_explainer).__init__()

            skorch_classifier_explainer.dense0 = nn.Linear(input_units, num_units)
            skorch_classifier_explainer.nonlin = nonlin
            skorch_classifier_explainer.dense1 = nn.Linear(num_units, num_units)
            skorch_classifier_explainer.output = nn.Linear(num_units, 1)

        def forward(skorch_classifier_explainer, X, **kwargs):
            X = skorch_classifier_explainer.nonlin(
                skorch_classifier_explainer.dense0(X)
            )
            X = skorch_classifier_explainer.nonlin(
                skorch_classifier_explainer.dense1(X)
            )
            X = skorch_classifier_explainer.output(X)
            return X

    model = NeuralNetRegressor(
        MyModule,
        max_epochs=20,
        lr=0.2,
        iterator_train__shuffle=True,
    )

    model.fit(X_df.values, y)
    return model, X_df, y


@pytest.fixture(scope="session")
def skorch_classifier():
    X, y = make_classification(200, 5, n_informative=3, random_state=0)
    X = X.astype(np.float32)
    y = y.astype(np.int64)

    X_df = pd.DataFrame(X, columns=["col" + str(i) for i in range(X.shape[1])])

    class MyModule(nn.Module):
        def __init__(
            skorch_classifier_explainer, input_units=5, num_units=5, nonlin=nn.ReLU()
        ):
            super(MyModule, skorch_classifier_explainer).__init__()

            skorch_classifier_explainer.dense0 = nn.Linear(input_units, num_units)
            skorch_classifier_explainer.nonlin = nonlin
            skorch_classifier_explainer.dense1 = nn.Linear(num_units, num_units)
            skorch_classifier_explainer.output = nn.Linear(num_units, 2)
            skorch_classifier_explainer.softmax = nn.Softmax(dim=-1)

        def forward(skorch_classifier_explainer, X, **kwargs):
            X = skorch_classifier_explainer.nonlin(
                skorch_classifier_explainer.dense0(X)
            )
            X = skorch_classifier_explainer.nonlin(
                skorch_classifier_explainer.dense1(X)
            )
            X = skorch_classifier_explainer.softmax(
                skorch_classifier_explainer.output(X)
            )
            return X

    model = NeuralNetClassifier(
        MyModule,
        max_epochs=20,
        lr=0.1,
    )

    model.fit(X_df.values, y)
    return model, X_df, y


@pytest.fixture(scope="session")
def skorch_regressor_explainer(skorch_regressor):
    model, X, y = skorch_regressor
    return RegressionExplainer(model, X, y, shap_kwargs=dict(check_additivity=False))


@pytest.fixture(scope="session")
def skorch_classifier_explainer(skorch_classifier):
    model, X, y = skorch_classifier
    return ClassifierExplainer(model, X, y, shap_kwargs=dict(check_additivity=False))


def test_preds(skorch_regressor_explainer):
    assert isinstance(skorch_regressor_explainer.preds, np.ndarray)


def test_permutation_importances(skorch_regressor_explainer):
    assert isinstance(
        skorch_regressor_explainer.get_permutation_importances_df(), pd.DataFrame
    )


def test_shap_base_value(skorch_regressor_explainer):
    assert isinstance(
        skorch_regressor_explainer.shap_base_value(), (np.floating, float)
    )


def test_shap_values_shape(skorch_regressor_explainer):
    assert skorch_regressor_explainer.get_shap_values_df().shape == (
        len(skorch_regressor_explainer),
        len(skorch_regressor_explainer.merged_cols),
    )


def test_shap_values(skorch_regressor_explainer):
    assert isinstance(skorch_regressor_explainer.get_shap_values_df(), pd.DataFrame)


def test_mean_abs_shap(skorch_regressor_explainer):
    assert isinstance(skorch_regressor_explainer.get_mean_abs_shap_df(), pd.DataFrame)


def test_calculate_properties(skorch_regressor_explainer):
    skorch_regressor_explainer.calculate_properties(include_interactions=False)


def test_pdp_df(skorch_regressor_explainer):
    assert isinstance(skorch_regressor_explainer.pdp_df("col1"), pd.DataFrame)


def test_preds(skorch_classifier_explainer):
    assert isinstance(skorch_classifier_explainer.preds, np.ndarray)


def test_pred_probas(skorch_classifier_explainer):
    assert isinstance(skorch_classifier_explainer.pred_probas(), np.ndarray)


def test_permutation_importances(skorch_classifier_explainer):
    assert isinstance(
        skorch_classifier_explainer.get_permutation_importances_df(), pd.DataFrame
    )


def test_shap_base_value(skorch_classifier_explainer):
    assert isinstance(
        skorch_classifier_explainer.shap_base_value(), (np.floating, float)
    )


def test_shap_values_shape(skorch_classifier_explainer):
    assert skorch_classifier_explainer.get_shap_values_df().shape == (
        len(skorch_classifier_explainer),
        len(skorch_classifier_explainer.merged_cols),
    )


def test_shap_values(skorch_classifier_explainer):
    assert isinstance(skorch_classifier_explainer.get_shap_values_df(), pd.DataFrame)


def test_mean_abs_shap(skorch_classifier_explainer):
    assert isinstance(skorch_classifier_explainer.get_mean_abs_shap_df(), pd.DataFrame)


def test_calculate_properties(skorch_classifier_explainer):
    skorch_classifier_explainer.calculate_properties(include_interactions=False)


def test_pdp_df(skorch_classifier_explainer):
    assert isinstance(skorch_classifier_explainer.pdp_df("col1"), pd.DataFrame)


def test_metrics(skorch_classifier_explainer):
    assert isinstance(skorch_classifier_explainer.metrics(), dict)


def test_precision_df(skorch_classifier_explainer):
    assert isinstance(skorch_classifier_explainer.get_precision_df(), pd.DataFrame)


def test_lift_curve_df(skorch_classifier_explainer):
    assert isinstance(skorch_classifier_explainer.get_liftcurve_df(), pd.DataFrame)


def test_prediction_result_df(skorch_classifier_explainer):
    assert isinstance(skorch_classifier_explainer.prediction_result_df(0), pd.DataFrame)
