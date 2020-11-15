import unittest
from pathlib import Path
import pickle 

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor

from explainerdashboard.explainers import ClassifierExplainer, RegressionExplainer
from explainerdashboard.datasets import titanic_survive, titanic_fare, titanic_names


class TestRFClassifierExplainerPicklable(unittest.TestCase):
    def setUp(self):
        X_train, y_train, X_test, y_test = titanic_survive()
        train_names, test_names = titanic_names()

        model = RandomForestClassifier(n_estimators=5, max_depth=2)
        model.fit(X_train, y_train)

        self.explainer = ClassifierExplainer(
                            model, X_test, y_test, 
                            cats=[{'Gender': ['Sex_female', 'Sex_male', 'Sex_nan']}, 
                                                'Deck', 'Embarked'],
                            labels=['Not survived', 'Survived'],
                            idxs=test_names)

    def test_rf_pickle(self):
        pickle_location = Path.cwd() / "rf_pickle_test.pkl"
        pickle.dump(self.explainer, open(str(pickle_location), "wb"))
        assert pickle_location.exists
        pickle_location.unlink()


class TestXGBClassifierExplainerPicklable(unittest.TestCase):
    def setUp(self):
        X_train, y_train, X_test, y_test = titanic_survive()
        train_names, test_names = titanic_names()

        model = XGBClassifier(n_estimators=5, max_depth=2)
        model.fit(X_train, y_train)

        self.explainer = ClassifierExplainer(
                            model, X_test, y_test, 
                            cats=[{'Gender': ['Sex_female', 'Sex_male', 'Sex_nan']}, 
                                                'Deck', 'Embarked'],
                            labels=['Not survived', 'Survived'],
                            idxs=test_names)

    def test_xgb_pickle(self):
        pickle_location = Path.cwd() / "xgb_pickle_test.pkl"
        pickle.dump(self.explainer, open(str(pickle_location), "wb"))
        assert pickle_location.exists
        pickle_location.unlink()

class TestRFRegressionExplainerPicklable(unittest.TestCase):
    def setUp(self):
        X_train, y_train, X_test, y_test = titanic_fare()
        train_names, test_names = titanic_names()

        model = RandomForestRegressor(n_estimators=5, max_depth=2)
        model.fit(X_train, y_train)

        self.explainer = RegressionExplainer(
                            model, X_test, y_test, 
                            cats=[{'Gender': ['Sex_female', 'Sex_male', 'Sex_nan']}, 
                                                'Deck', 'Embarked'],
                            idxs=test_names)

    def test_rf_pickle(self):
        pickle_location = Path.cwd() / "rf_reg_pickle_test.pkl"
        pickle.dump(self.explainer, open(str(pickle_location), "wb"))
        assert pickle_location.exists
        pickle_location.unlink()


class TestXGBRegressionExplainerPicklable(unittest.TestCase):
    def setUp(self):
        X_train, y_train, X_test, y_test = titanic_fare()
        train_names, test_names = titanic_names()

        model = XGBRegressor(n_estimators=5, max_depth=2)
        model.fit(X_train, y_train)

        self.explainer = RegressionExplainer(
                            model, X_test, y_test, 
                            cats=[{'Gender': ['Sex_female', 'Sex_male', 'Sex_nan']}, 
                                                'Deck', 'Embarked'],
                            idxs=test_names)

    def test_xgb_pickle(self):
        pickle_location = Path.cwd() / "xgb_reg_pickle_test.pkl"
        pickle.dump(self.explainer, open(str(pickle_location), "wb"))
        assert pickle_location.exists
        pickle_location.unlink()