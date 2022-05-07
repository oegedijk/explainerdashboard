import unittest

import pandas as pd

import plotly.graph_objects as go
import dtreeviz 

from xgboost import XGBClassifier

from explainerdashboard.explainers import ClassifierExplainer
from explainerdashboard.datasets import *


def test_xgbclas_graphviz_available(precalculated_xgb_classifier_explainer):
    assert isinstance(precalculated_xgb_classifier_explainer.graphviz_available, bool)

def test_xgbclas_shadow_trees(precalculated_xgb_classifier_explainer):
    dt = precalculated_xgb_classifier_explainer.shadow_trees
    assert isinstance(dt, list)
    assert isinstance(dt[0], dtreeviz.models.shadow_decision_tree.ShadowDecTree)

def test_xgbclas_decisionpath_df(precalculated_xgb_classifier_explainer, test_names):
    df = precalculated_xgb_classifier_explainer.get_decisionpath_df(tree_idx=0, index=0)
    assert isinstance(df, pd.DataFrame)

    df = precalculated_xgb_classifier_explainer.get_decisionpath_df(tree_idx=0, index=test_names[0])
    assert isinstance(df, pd.DataFrame)

def test_xgbclas_plot_trees(precalculated_xgb_classifier_explainer, test_names):
    fig = precalculated_xgb_classifier_explainer.plot_trees(index=0)
    assert isinstance(fig, go.Figure)

    fig = precalculated_xgb_classifier_explainer.plot_trees(index=test_names[0])
    assert isinstance(fig, go.Figure)

    fig = precalculated_xgb_classifier_explainer.plot_trees(index=test_names[0], highlight_tree=0)
    assert isinstance(fig, go.Figure)



def test_xgbreg_graphviz_available(precalculated_xgb_regression_explainer):
    assert isinstance(precalculated_xgb_regression_explainer.graphviz_available, bool)

def test_xgbreg_shadow_trees(precalculated_xgb_regression_explainer):
    dt = precalculated_xgb_regression_explainer.shadow_trees
    assert isinstance(dt, list)
    assert isinstance(dt[0], dtreeviz.models.shadow_decision_tree.ShadowDecTree)

def test_xgbreg_decisionpath_df(precalculated_xgb_regression_explainer, test_names):
    df = precalculated_xgb_regression_explainer.get_decisionpath_df(tree_idx=0, index=0)
    assert isinstance(df, pd.DataFrame)

    df = precalculated_xgb_regression_explainer.get_decisionpath_df(tree_idx=0, index=test_names[0])
    assert isinstance(df, pd.DataFrame)

def test_xgbreg_plot_trees(precalculated_xgb_regression_explainer, test_names):
    fig = precalculated_xgb_regression_explainer.plot_trees(index=0)
    assert isinstance(fig, go.Figure)

    fig = precalculated_xgb_regression_explainer.plot_trees(index=test_names[0])
    assert isinstance(fig, go.Figure)

    fig = precalculated_xgb_regression_explainer.plot_trees(index=test_names[0], highlight_tree=0)
    assert isinstance(fig, go.Figure)


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
                                                'Deck'],
                            idxs=test_names, 
                            labels=['Queenstown', 'Southampton', 'Cherbourg'])

    def test_graphviz_available(self):
        self.assertIsInstance(self.explainer.graphviz_available, bool)

    def test_shadow_trees(self):
        dt = self.explainer.shadow_trees
        self.assertIsInstance(dt, list)
        self.assertIsInstance(dt[0], dtreeviz.models.shadow_decision_tree.ShadowDecTree)

    def test_decisionpath_df(self):
        df = self.explainer.get_decisionpath_df(tree_idx=0, index=0)
        self.assertIsInstance(df, pd.DataFrame)

        df = self.explainer.get_decisionpath_df(tree_idx=0, index=self.names[0])
        self.assertIsInstance(df, pd.DataFrame)

        df = self.explainer.get_decisionpath_df(tree_idx=0, index=self.names[0], pos_label=0)
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