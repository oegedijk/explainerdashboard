import unittest

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, roc_auc_score

import plotly.graph_objects as go
import dtreeviz 

from explainerdashboard.explainers import RandomForestRegressionExplainer
from explainerdashboard.explainers import RandomForestClassifierExplainer
from explainerdashboard.datasets import titanic_survive, titanic_fare, titanic_names


class ClassifierBunchTests(unittest.TestCase):
    def setUp(self):
        X_train, y_train, X_test, y_test = titanic_survive()
        train_names, test_names = titanic_names()
        _, self.names = titanic_names()

        model = RandomForestClassifier(n_estimators=5, max_depth=2)
        model.fit(X_train, y_train)

        self.explainer = RandomForestClassifierExplainer(
                            model, X_test, y_test, roc_auc_score, 
                            shap='tree',
                            cats=['Sex', 'Cabin', 'Embarked'],
                            idxs=test_names, 
                            labels=['Not survived', 'Survived'])

    def test_graphviz_available(self):
        self.assertIsInstance(self.explainer.graphviz_available, bool)

    def test_decision_trees(self):
        dt = self.explainer.decision_trees
        self.assertIsInstance(dt, list)
        self.assertIsInstance(dt[0], dtreeviz.shadow.ShadowDecTree)

    def test_decisiontree_df(self):
        df = self.explainer.decisiontree_df(tree_idx=0, index=0)
        self.assertIsInstance(df, pd.DataFrame)

        df = self.explainer.decisiontree_df(tree_idx=0, index=self.names[0])
        self.assertIsInstance(df, pd.DataFrame)

    def test_plot_trees(self):
        fig = self.explainer.plot_trees(index=0)
        self.assertIsInstance(fig, go.Figure)

        fig = self.explainer.plot_trees(index=self.names[0])
        self.assertIsInstance(fig, go.Figure)

        fig = self.explainer.plot_trees(index=self.names[0], highlight_tree=0)
        self.assertIsInstance(fig, go.Figure)

    def test_calculate_properties(self):
        self.explainer.calculate_properties()


class RegressionBunchTests(unittest.TestCase):
    def setUp(self):
        X_train, y_train, X_test, y_test = titanic_fare()
        self.test_len = len(X_test)

        train_names, test_names = titanic_names()
        _, self.names = titanic_names()

        model = RandomForestRegressor(n_estimators=5, max_depth=2)
        model.fit(X_train, y_train)

        self.explainer = RandomForestRegressionExplainer(
                            model, X_test, y_test, r2_score, 
                            shap='tree',
                            cats=['Sex', 'Cabin', 'Embarked'],
                            idxs=test_names)

    def test_graphviz_available(self):
        self.assertIsInstance(self.explainer.graphviz_available, bool)

    def test_decision_trees(self):
        dt = self.explainer.decision_trees
        self.assertIsInstance(dt, list)
        self.assertIsInstance(dt[0], dtreeviz.shadow.ShadowDecTree)

    def test_decisiontree_df(self):
        df = self.explainer.decisiontree_df(tree_idx=0, index=0)
        self.assertIsInstance(df, pd.DataFrame)

        df = self.explainer.decisiontree_df(tree_idx=0, index=self.names[0])
        self.assertIsInstance(df, pd.DataFrame)

    def test_plot_trees(self):
        fig = self.explainer.plot_trees(index=0)
        self.assertIsInstance(fig, go.Figure)

        fig = self.explainer.plot_trees(index=self.names[0])
        self.assertIsInstance(fig, go.Figure)

        fig = self.explainer.plot_trees(index=self.names[0], highlight_tree=0)
        self.assertIsInstance(fig, go.Figure)

    def test_calculate_properties(self):
        self.explainer.calculate_properties()