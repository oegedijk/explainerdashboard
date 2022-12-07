import pytest

import pandas as pd

from explainerdashboard.explainers import ClassifierExplainer, RegressionExplainer
from explainerdashboard.datasets import titanic_survive, titanic_fare


@pytest.fixture(scope="module")
def classifier_explainer_with_cv(fitted_rf_classifier_model):
    _, _, X_test, y_test = titanic_survive()
    return ClassifierExplainer(
        fitted_rf_classifier_model, 
        X_test, y_test, 
        cats=[{'Gender': ['Sex_female', 'Sex_male', 'Sex_nan']}, 'Deck', 'Embarked'],
        cv=3
    )

@pytest.fixture(scope="module")
def regression_explainer_with_cv(fitted_rf_regression_model):
    _, _, X_test, y_test = titanic_fare()
    return RegressionExplainer(
        fitted_rf_regression_model, 
        X_test, y_test, 
        cats=[{'Gender': ['Sex_female', 'Sex_male', 'Sex_nan']}, 'Deck', 'Embarked'],
        cv=3
    )

def test_clas_cv_permutation_importances(classifier_explainer_with_cv):
    assert isinstance(classifier_explainer_with_cv.permutation_importances(), pd.DataFrame)
    assert isinstance(classifier_explainer_with_cv.permutation_importances(pos_label=0), pd.DataFrame)

def test_clas_cv_metrics(classifier_explainer_with_cv):
    assert isinstance(classifier_explainer_with_cv.metrics(), dict)
    assert isinstance(classifier_explainer_with_cv.metrics(pos_label=0), dict)


def test_reg_cv_permutation_importances(regression_explainer_with_cv):
    assert isinstance(regression_explainer_with_cv.permutation_importances(), pd.DataFrame)
    assert isinstance(regression_explainer_with_cv.permutation_importances(pos_label=0), pd.DataFrame)

def test_reg_cv_metrics(regression_explainer_with_cv):
    assert isinstance(regression_explainer_with_cv.metrics(), dict)


