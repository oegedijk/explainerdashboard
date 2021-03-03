import pandas as pd
import unittest

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from explainerdashboard.explainers import ClassifierExplainer, RegressionExplainer
from explainerdashboard.datasets import titanic_survive, titanic_fare


class ExternalSourceClassifierTests(unittest.TestCase):
    def setUp(self):
        X_train, y_train, X_test, y_test = titanic_survive()
        model = RandomForestClassifier(n_estimators=50, max_depth=4).fit(X_train, y_train)

        X_test.reset_index(drop=True, inplace=True)
        X_test.index = X_test.index.astype(str)

        X_test1, y_test1 = X_test.iloc[:100], y_test.iloc[:100]
        X_test2, y_test2 = X_test.iloc[100:], y_test.iloc[100:]

        self.explainer = ClassifierExplainer(model, X_test1, y_test1, cats=['Sex', 'Deck'])    

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

        self.explainer.set_index_exists_func(index_exists_func)
        self.explainer.set_index_list_func(index_list_func)
        self.explainer.set_X_row_func(X_func)
        self.explainer.set_y_func(y_func)

    def test_get_X_row(self):
        self.assertIsInstance(self.explainer.get_X_row(0), pd.DataFrame)
        self.assertIsInstance(self.explainer.get_X_row("0"), pd.DataFrame)
        self.assertIsInstance(self.explainer.get_X_row("120"), pd.DataFrame)
        self.assertIsInstance(self.explainer.get_X_row("150"), pd.DataFrame)

    def test_get_shap_row(self):
        self.assertIsInstance(self.explainer.get_shap_row(0), pd.DataFrame)
        self.assertIsInstance(self.explainer.get_shap_row("0"), pd.DataFrame)
        self.assertIsInstance(self.explainer.get_shap_row("120"), pd.DataFrame)
        self.assertIsInstance(self.explainer.get_shap_row("150"), pd.DataFrame)

    def test_get_y(self):
        self.assertIsInstance(self.explainer.get_y(0), int)
        self.assertIsInstance(self.explainer.get_y("0"), int)
        self.assertIsInstance(self.explainer.get_y("120"), int)
        self.assertIsInstance(self.explainer.get_y("150"), int)

    def test_index_list(self):
        index_list = self.explainer.get_index_list()
        self.assertIn('100', index_list)
        self.assertNotIn('160', index_list)

    def test_index_exists(self):
        self.assertTrue(self.explainer.index_exists("0"))
        self.assertTrue(self.explainer.index_exists("100"))
        self.assertTrue(self.explainer.index_exists("160"))
        self.assertTrue(self.explainer.index_exists(0))

        self.assertFalse(self.explainer.index_exists(-1))
        self.assertFalse(self.explainer.index_exists(120))
        self.assertFalse(self.explainer.index_exists("wrong index"))


class ExternalSourceRegressionTests(unittest.TestCase):
    def setUp(self):
        X_train, y_train, X_test, y_test = titanic_fare()
        model = RandomForestRegressor(n_estimators=50, max_depth=4).fit(X_train, y_train)

        X_test.reset_index(drop=True, inplace=True)
        X_test.index = X_test.index.astype(str)

        X_test1, y_test1 = X_test.iloc[:100], y_test.iloc[:100]
        X_test2, y_test2 = X_test.iloc[100:], y_test.iloc[100:]

        self.explainer = RegressionExplainer(model, X_test1, y_test1, cats=['Sex', 'Deck'])    

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

        self.explainer.set_index_exists_func(index_exists_func)
        self.explainer.set_index_list_func(index_list_func)
        self.explainer.set_X_row_func(X_func)
        self.explainer.set_y_func(y_func)

    def test_get_X_row(self):
        self.assertIsInstance(self.explainer.get_X_row(0), pd.DataFrame)
        self.assertIsInstance(self.explainer.get_X_row("0"), pd.DataFrame)
        self.assertIsInstance(self.explainer.get_X_row("120"), pd.DataFrame)
        self.assertIsInstance(self.explainer.get_X_row("150"), pd.DataFrame)

    def test_get_shap_row(self):
        self.assertIsInstance(self.explainer.get_shap_row(0), pd.DataFrame)
        self.assertIsInstance(self.explainer.get_shap_row("0"), pd.DataFrame)
        self.assertIsInstance(self.explainer.get_shap_row("120"), pd.DataFrame)
        self.assertIsInstance(self.explainer.get_shap_row("150"), pd.DataFrame)

    def test_get_y(self):
        self.assertIsInstance(self.explainer.get_y(0), float)
        self.assertIsInstance(self.explainer.get_y("0"), float)
        self.assertIsInstance(self.explainer.get_y("120"), float)
        self.assertIsInstance(self.explainer.get_y("150"), float)

    def test_index_list(self):
        index_list = self.explainer.get_index_list()
        self.assertIn('100', index_list)
        self.assertNotIn('160', index_list)

    def test_index_exists(self):
        self.assertTrue(self.explainer.index_exists("0"))
        self.assertTrue(self.explainer.index_exists("100"))
        self.assertTrue(self.explainer.index_exists("160"))
        self.assertTrue(self.explainer.index_exists(0))

        self.assertFalse(self.explainer.index_exists(-1))
        self.assertFalse(self.explainer.index_exists(120))
        self.assertFalse(self.explainer.index_exists("wrong index"))


