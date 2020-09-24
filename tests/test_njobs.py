import unittest

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from explainerdashboard.explainers import ClassifierExplainer
from explainerdashboard.datasets import titanic_survive, titanic_names


class NJobs5ExplainerTests(unittest.TestCase):
    def setUp(self):
        X_train, y_train, X_test, y_test = titanic_survive()
        train_names, test_names = titanic_names()

        model = RandomForestClassifier(n_estimators=5, max_depth=2)
        model.fit(X_train, y_train)

        self.explainer = ClassifierExplainer(
                            model, X_test, y_test, roc_auc_score, n_jobs=5)

    def test_permutation_importances(self):
        self.assertIsInstance(self.explainer.permutation_importances, pd.DataFrame)
        self.assertIsInstance(self.explainer.permutation_importances_cats, pd.DataFrame)


class NJobsMinusOneExplainerTests(unittest.TestCase):
    def setUp(self):
        X_train, y_train, X_test, y_test = titanic_survive()
        train_names, test_names = titanic_names()

        model = RandomForestClassifier(n_estimators=5, max_depth=2)
        model.fit(X_train, y_train)

        self.explainer = ClassifierExplainer(
                            model, X_test, y_test, roc_auc_score, n_jobs=-1)

    def test_permutation_importances(self):
        self.assertIsInstance(self.explainer.permutation_importances, pd.DataFrame)
        self.assertIsInstance(self.explainer.permutation_importances_cats, pd.DataFrame)