import unittest

import pandas as pd
import numpy as np

from xgboost import XGBClassifier, XGBRegressor

import plotly.graph_objects as go
import dtreeviz 

from explainerdashboard.explainers import RegressionExplainer
from explainerdashboard.explainers import ClassifierExplainer
from explainerdashboard.datasets import *


class XGBClassifierExplainerTests(unittest.TestCase):
    def setUp(self):
        X_train, y_train, X_test, y_test = titanic_survive()
        train_names, test_names = titanic_names()
        _, self.names = titanic_names()

        model = XGBClassifier(n_estimators=5)
        model.fit(X_train, y_train)

        self.explainer = ClassifierExplainer(
                            model, X_test, y_test, 
                            cats=['Sex', 'Cabin', 'Embarked'],
                            idxs=test_names, 
                            labels=['Not survived', 'Survived'])

    def test_graphviz_available(self):
        self.assertIsInstance(self.explainer.graphviz_available, bool)

    def test_decision_trees(self):
        dt = self.explainer.decision_trees
        self.assertIsInstance(dt, list)
        self.assertIsInstance(dt[0], dtreeviz.models.shadow_decision_tree.ShadowDecTree)

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

class XGBMultiClassifierExplainerTests(unittest.TestCase):
    def setUp(self):
        X_train, y_train, X_test, y_test = titanic_embarked()
        train_names, test_names = titanic_names()
        _, self.names = titanic_names()

        model = XGBClassifier(n_estimators=5)
        model.fit(X_train, y_train)

        self.explainer = ClassifierExplainer(
                            model, X_test, y_test, model_output='raw',
                            cats=[{'Gender': ['Sex_female', 'Sex_male', 'Sex_nan']}, 
                                                'Deck', 'Embarked'],
                            idxs=test_names, 
                            labels=['Queenstown', 'Southampton', 'Cherbourg'])

    def test_graphviz_available(self):
        self.assertIsInstance(self.explainer.graphviz_available, bool)

    def test_decision_trees(self):
        dt = self.explainer.decision_trees
        self.assertIsInstance(dt, list)
        self.assertIsInstance(dt[0], dtreeviz.models.shadow_decision_tree.ShadowDecTree)

    def test_decisiontree_df(self):
        df = self.explainer.decisiontree_df(tree_idx=0, index=0)
        self.assertIsInstance(df, pd.DataFrame)

        df = self.explainer.decisiontree_df(tree_idx=0, index=self.names[0])
        self.assertIsInstance(df, pd.DataFrame)

        df = self.explainer.decisiontree_df(tree_idx=0, index=self.names[0], pos_label=0)
        self.assertIsInstance(df, pd.DataFrame)


    def test_plot_trees(self):
        fig = self.explainer.plot_trees(index=0)
        self.assertIsInstance(fig, go.Figure)

        fig = self.explainer.plot_trees(index=self.names[0])
        self.assertIsInstance(fig, go.Figure)

        fig = self.explainer.plot_trees(index=self.names[0], highlight_tree=0)
        self.assertIsInstance(fig, go.Figure)

        fig = self.explainer.plot_trees(index=self.names[0], pos_label=0)
        self.assertIsInstance(fig, go.Figure)

    def test_calculate_properties(self):
        self.explainer.calculate_properties()


class XGBRegressionExplainerTests(unittest.TestCase):
    def setUp(self):
        X_train, y_train, X_test, y_test = titanic_fare()
        self.test_len = len(X_test)

        train_names, test_names = titanic_names()
        _, self.names = titanic_names()

        model = XGBRegressor(n_estimators=5, max_depth=2)
        model.fit(X_train, y_train)

        self.explainer = RegressionExplainer(
                            model, X_test, y_test, 
                            cats=[{'Gender': ['Sex_female', 'Sex_male', 'Sex_nan']}, 
                                                'Deck', 'Embarked'],
                            idxs=test_names)

    def test_graphviz_available(self):
        self.assertIsInstance(self.explainer.graphviz_available, bool)

    def test_decision_trees(self):
        dt = self.explainer.decision_trees
        self.assertIsInstance(dt, list)
        self.assertIsInstance(dt[0], dtreeviz.models.shadow_decision_tree.ShadowDecTree)

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