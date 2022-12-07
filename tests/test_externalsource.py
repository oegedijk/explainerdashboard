import pytest

import pandas as pd

from explainerdashboard.explainers import ClassifierExplainer, RegressionExplainer
from explainerdashboard.datasets import titanic_survive, titanic_fare


@pytest.fixture(scope='session')
def classifier_explainer_with_external_data(fitted_rf_classifier_model):
    _, _, X_test, y_test = titanic_survive()
    X_test.reset_index(drop=True, inplace=True)
    X_test.index = X_test.index.astype(str)

    X_test1, y_test1 = X_test.iloc[:100], y_test.iloc[:100]
    X_test2, y_test2 = X_test.iloc[100:], y_test.iloc[100:]

    explainer = ClassifierExplainer(fitted_rf_classifier_model, X_test1, y_test1, cats=['Sex', 'Deck'])    

    def index_exists_func(index):
        return index in X_test2.index

    def index_list_func():
        # only returns first 50 indexes
        return list(X_test2.index[:50])

    def y_func(index):
        idx = X_test2.index.get_loc(index)
        return y_test2.iloc[[idx]]

    def X_func(index):
        idx = X_test2.index.get_loc(index)
        return X_test2.iloc[[idx]]

    explainer.set_index_exists_func(index_exists_func)
    explainer.set_index_list_func(index_list_func)
    explainer.set_X_row_func(X_func)
    explainer.set_y_func(y_func)
    return explainer


@pytest.fixture(scope='session')
def classifier_explainer_with_external_data_methods(fitted_rf_classifier_model):
    _, _, X_test, y_test = titanic_survive()
    X_test.reset_index(drop=True, inplace=True)
    X_test.index = X_test.index.astype(str)

    X_test1, y_test1 = X_test.iloc[:100], y_test.iloc[:100]
    X_test2, y_test2 = X_test.iloc[100:], y_test.iloc[100:]

    explainer = ClassifierExplainer(fitted_rf_classifier_model, X_test1, y_test1, cats=['Sex', 'Deck'])    

    def index_exists_func(self, index):
        assert self.is_classifier
        return index in X_test2.index

    def index_list_func(self):
        assert self.is_classifier
        # only returns first 50 indexes
        return list(X_test2.index[:50])

    def y_func(self, index):
        assert self.is_classifier
        idx = X_test2.index.get_loc(index)
        return y_test2.iloc[[idx]]

    def X_func(self, index):
        assert self.is_classifier
        idx = X_test2.index.get_loc(index)
        return X_test2.iloc[[idx]]

    explainer.set_index_exists_func(index_exists_func)
    explainer.set_index_list_func(index_list_func)
    explainer.set_X_row_func(X_func)
    explainer.set_y_func(y_func)
    return explainer

@pytest.fixture(scope='session')
def regression_explainer_with_external_data(fitted_rf_regression_model):
    _, _, X_test, y_test = titanic_fare()

    X_test.reset_index(drop=True, inplace=True)
    X_test.index = X_test.index.astype(str)

    X_test1, y_test1 = X_test.iloc[:100], y_test.iloc[:100]
    X_test2, y_test2 = X_test.iloc[100:], y_test.iloc[100:]

    explainer = RegressionExplainer(fitted_rf_regression_model, X_test1, y_test1, cats=['Sex', 'Deck'])    

    def index_exists_func(index):
        return index in X_test2.index

    def index_list_func():
        # only returns first 50 indexes
        return list(X_test2.index[:50])

    def y_func(index):
        idx = X_test2.index.get_loc(index)
        return y_test2.iloc[[idx]]

    def X_func(index):
        idx = X_test2.index.get_loc(index)
        return X_test2.iloc[[idx]]

    explainer.set_index_exists_func(index_exists_func)
    explainer.set_index_list_func(index_list_func)
    explainer.set_X_row_func(X_func)
    explainer.set_y_func(y_func)
    return explainer

def test_clas_externalsource_reset_index_list(classifier_explainer_with_external_data):
    classifier_explainer_with_external_data.reset_index_list()
    index_list = classifier_explainer_with_external_data.get_index_list()
    assert ('100' in index_list)
    assert (not '160'in index_list)

def test_clas_externalsource_get_X_row(classifier_explainer_with_external_data):
    assert isinstance(classifier_explainer_with_external_data.get_X_row(0), pd.DataFrame)
    assert isinstance(classifier_explainer_with_external_data.get_X_row("0"), pd.DataFrame)
    assert isinstance(classifier_explainer_with_external_data.get_X_row("120"), pd.DataFrame)
    assert isinstance(classifier_explainer_with_external_data.get_X_row("150"), pd.DataFrame)

def test_clas_externalsource_get_shap_row(classifier_explainer_with_external_data):
    assert isinstance(classifier_explainer_with_external_data.get_shap_row(0), pd.DataFrame)
    assert isinstance(classifier_explainer_with_external_data.get_shap_row("0"), pd.DataFrame)
    assert isinstance(classifier_explainer_with_external_data.get_shap_row("120"), pd.DataFrame)
    assert isinstance(classifier_explainer_with_external_data.get_shap_row("150"), pd.DataFrame)

def test_clas_externalsource_get_y(classifier_explainer_with_external_data):
    assert isinstance(classifier_explainer_with_external_data.get_y(0), int)
    assert isinstance(classifier_explainer_with_external_data.get_y("0"), int)
    assert isinstance(classifier_explainer_with_external_data.get_y("120"), int)
    assert isinstance(classifier_explainer_with_external_data.get_y("150"), int)

def test_clas_externalsource_index_list(classifier_explainer_with_external_data):
    index_list = classifier_explainer_with_external_data.get_index_list()
    assert ('100' in index_list)
    assert (not '160'in index_list)

def test_clas_externalsource_index_exists(classifier_explainer_with_external_data):
    assert (classifier_explainer_with_external_data.index_exists("0"))
    assert (classifier_explainer_with_external_data.index_exists("100"))
    assert (classifier_explainer_with_external_data.index_exists("160"))
    assert (classifier_explainer_with_external_data.index_exists(0))

    assert (not classifier_explainer_with_external_data.index_exists(-1))
    assert (not classifier_explainer_with_external_data.index_exists(120))
    assert (not classifier_explainer_with_external_data.index_exists("wrong index"))

def test_clas_externalsource_methods_get_X_row(classifier_explainer_with_external_data_methods):
    assert isinstance(classifier_explainer_with_external_data_methods.get_X_row(0), pd.DataFrame)
    assert isinstance(classifier_explainer_with_external_data_methods.get_X_row("0"), pd.DataFrame)
    assert isinstance(classifier_explainer_with_external_data_methods.get_X_row("120"), pd.DataFrame)
    assert isinstance(classifier_explainer_with_external_data_methods.get_X_row("150"), pd.DataFrame)

def test_clas_externalsource_methods_get_shap_row(classifier_explainer_with_external_data_methods):
    assert isinstance(classifier_explainer_with_external_data_methods.get_shap_row(0), pd.DataFrame)
    assert isinstance(classifier_explainer_with_external_data_methods.get_shap_row("0"), pd.DataFrame)
    assert isinstance(classifier_explainer_with_external_data_methods.get_shap_row("120"), pd.DataFrame)
    assert isinstance(classifier_explainer_with_external_data_methods.get_shap_row("150"), pd.DataFrame)

def test_clas_externalsource_methods_get_y(classifier_explainer_with_external_data_methods):
    assert isinstance(classifier_explainer_with_external_data_methods.get_y(0), int)
    assert isinstance(classifier_explainer_with_external_data_methods.get_y("0"), int)
    assert isinstance(classifier_explainer_with_external_data_methods.get_y("120"), int)
    assert isinstance(classifier_explainer_with_external_data_methods.get_y("150"), int)

def test_clas_externalsource_methods_index_list(classifier_explainer_with_external_data_methods):
    index_list = classifier_explainer_with_external_data_methods.get_index_list()
    assert ('100' in index_list)
    assert (not '160'in index_list)

def test_clas_externalsource_methods_index_exists(classifier_explainer_with_external_data_methods):
    assert (classifier_explainer_with_external_data_methods.index_exists("0"))
    assert (classifier_explainer_with_external_data_methods.index_exists("100"))
    assert (classifier_explainer_with_external_data_methods.index_exists("160"))
    assert (classifier_explainer_with_external_data_methods.index_exists(0))

    assert (not classifier_explainer_with_external_data_methods.index_exists(-1))
    assert (not classifier_explainer_with_external_data_methods.index_exists(120))
    assert (not classifier_explainer_with_external_data_methods.index_exists("wrong index"))

def test_reg_externalsource_get_X_row(regression_explainer_with_external_data):
    assert isinstance(regression_explainer_with_external_data.get_X_row(0), pd.DataFrame)
    assert isinstance(regression_explainer_with_external_data.get_X_row("0"), pd.DataFrame)
    assert isinstance(regression_explainer_with_external_data.get_X_row("120"), pd.DataFrame)
    assert isinstance(regression_explainer_with_external_data.get_X_row("150"), pd.DataFrame)

def test_reg_externalsource_get_shap_row(regression_explainer_with_external_data):
    assert isinstance(regression_explainer_with_external_data.get_shap_row(0), pd.DataFrame)
    assert isinstance(regression_explainer_with_external_data.get_shap_row("0"), pd.DataFrame)
    assert isinstance(regression_explainer_with_external_data.get_shap_row("120"), pd.DataFrame)
    assert isinstance(regression_explainer_with_external_data.get_shap_row("150"), pd.DataFrame)

def test_reg_externalsource_get_y(regression_explainer_with_external_data):
    assert isinstance(regression_explainer_with_external_data.get_y(0), float)
    assert isinstance(regression_explainer_with_external_data.get_y("0"), float)
    assert isinstance(regression_explainer_with_external_data.get_y("120"), float)
    assert isinstance(regression_explainer_with_external_data.get_y("150"), float)

def test_reg_externalsource_index_list(regression_explainer_with_external_data):
    index_list = regression_explainer_with_external_data.get_index_list()
    assert ('100' in index_list)
    assert (not '160' in index_list)

def test_reg_externalsource_index_exists(regression_explainer_with_external_data):
    assert (regression_explainer_with_external_data.index_exists("0"))
    assert (regression_explainer_with_external_data.index_exists("100"))
    assert (regression_explainer_with_external_data.index_exists("160"))
    assert (regression_explainer_with_external_data.index_exists(0))

    assert (not regression_explainer_with_external_data.index_exists(-1))
    assert (not regression_explainer_with_external_data.index_exists(120))
    assert (not regression_explainer_with_external_data.index_exists("wrong index"))


