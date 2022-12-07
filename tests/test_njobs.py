import pandas as pd

from sklearn.metrics import roc_auc_score

from explainerdashboard.explainers import ClassifierExplainer
from explainerdashboard.datasets import titanic_survive


def test_permutation_importances_njobs_5(fitted_rf_classifier_model):
    _, _, X_test, y_test = titanic_survive()
    explainer = ClassifierExplainer(fitted_rf_classifier_model, X_test, y_test, roc_auc_score, n_jobs=5)
    assert isinstance(explainer.get_permutation_importances_df(), pd.DataFrame)


def test_permutation_importances_njobs_minus1(fitted_rf_classifier_model):
    _, _, X_test, y_test = titanic_survive()
    explainer = ClassifierExplainer(fitted_rf_classifier_model, X_test, y_test, roc_auc_score, n_jobs=-1)
    assert isinstance(explainer.get_permutation_importances_df(), pd.DataFrame)