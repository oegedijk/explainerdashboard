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




def test_graphviz_available(precalculated_xgb_multiclass_explainer):
    assert isinstance(precalculated_xgb_multiclass_explainer.graphviz_available, bool)

def test_shadow_trees(precalculated_xgb_multiclass_explainer):
    dt = precalculated_xgb_multiclass_explainer.shadow_trees
    assert isinstance(dt, list)
    assert isinstance(dt[0], dtreeviz.models.shadow_decision_tree.ShadowDecTree)

def test_decisionpath_df(precalculated_xgb_multiclass_explainer, test_names):
    df = precalculated_xgb_multiclass_explainer.get_decisionpath_df(tree_idx=0, index=0)
    assert isinstance(df, pd.DataFrame)

    df = precalculated_xgb_multiclass_explainer.get_decisionpath_df(tree_idx=0, index=test_names[0])
    assert isinstance(df, pd.DataFrame)

    df = precalculated_xgb_multiclass_explainer.get_decisionpath_df(tree_idx=0, index=test_names[0], pos_label=0)
    assert isinstance(df, pd.DataFrame)


def test_plot_trees(precalculated_xgb_multiclass_explainer, test_names):
    fig = precalculated_xgb_multiclass_explainer.plot_trees(index=0)
    assert isinstance(fig, go.Figure)

    fig = precalculated_xgb_multiclass_explainer.plot_trees(index=test_names[0])
    assert isinstance(fig, go.Figure)

    fig = precalculated_xgb_multiclass_explainer.plot_trees(index=test_names[0], highlight_tree=0)
    assert isinstance(fig, go.Figure)

    fig = precalculated_xgb_multiclass_explainer.plot_trees(index=test_names[0], pos_label=0)
    assert isinstance(fig, go.Figure)

def test_calculate_properties(precalculated_xgb_multiclass_explainer):
    precalculated_xgb_multiclass_explainer.calculate_properties()