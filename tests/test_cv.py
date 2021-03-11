import unittest

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from explainerdashboard.explainers import ClassifierExplainer, RegressionExplainer
from explainerdashboard.datasets import titanic_survive, titanic_fare


class ClassifierCVTests(unittest.TestCase):
    def setUp(self):
        X_train, y_train, X_test, y_test = titanic_survive()

        model = RandomForestClassifier(n_estimators=5, max_depth=2)
        model.fit(X_train, y_train)

        self.explainer = ClassifierExplainer(
                            model, X_train.iloc[:50], y_train.iloc[:50], 
                            cats=[{'Gender': ['Sex_female', 'Sex_male', 'Sex_nan']}, 
                                                'Deck', 'Embarked'],
                            cv=3)

    def test_cv_permutation_importances(self):
        self.assertIsInstance(self.explainer.permutation_importances(), pd.DataFrame)
        self.assertIsInstance(self.explainer.permutation_importances(pos_label=0), pd.DataFrame)

    def test_cv_metrics(self):
        self.assertIsInstance(self.explainer.metrics(), dict)
        self.assertIsInstance(self.explainer.metrics(pos_label=0), dict)


class RegressionCVTests(unittest.TestCase):
    def setUp(self):
        X_train, y_train, X_test, y_test = titanic_fare()
        model = RandomForestRegressor(n_estimators=5, max_depth=2).fit(X_train, y_train)

        self.explainer = RegressionExplainer(
                            model, X_test, y_test, 
                            cats=[{'Gender': ['Sex_female', 'Sex_male', 'Sex_nan']}, 
                                                'Deck', 'Embarked'],
                            cv=3)

    def test_cv_permutation_importances(self):
        self.assertIsInstance(self.explainer.permutation_importances(), pd.DataFrame)

    def test_cv_metrics(self):
        self.assertIsInstance(self.explainer.metrics(), dict)



