

import pandas as pd

import plotly.graph_objects as go
import dtreeviz 


def test_rfclas_graphviz_available(precalculated_rf_classifier_explainer):
    assert isinstance(precalculated_rf_classifier_explainer.graphviz_available, bool)

def test_rfclas_shadow_trees(precalculated_rf_classifier_explainer):
    dt = precalculated_rf_classifier_explainer.shadow_trees
    assert isinstance(dt, list)
    assert isinstance(dt[0], dtreeviz.models.shadow_decision_tree.ShadowDecTree)

def test_rfclas_decisionpath_df(precalculated_rf_classifier_explainer, test_names):
    df = precalculated_rf_classifier_explainer.get_decisionpath_df(tree_idx=0, index=0)
    assert isinstance(df, pd.DataFrame)

    df = precalculated_rf_classifier_explainer.get_decisionpath_df(tree_idx=0, index=test_names[0])
    assert isinstance(df, pd.DataFrame)

def test_rfclas_plot_trees(precalculated_rf_classifier_explainer, test_names):
    fig = precalculated_rf_classifier_explainer.plot_trees(index=0)
    assert isinstance(fig, go.Figure)

    fig = precalculated_rf_classifier_explainer.plot_trees(index=test_names[0])
    assert isinstance(fig, go.Figure)

    fig = precalculated_rf_classifier_explainer.plot_trees(index=test_names[0], highlight_tree=0)
    assert isinstance(fig, go.Figure)

def test_rfclas_calculate_properties(precalculated_rf_classifier_explainer):
    precalculated_rf_classifier_explainer.calculate_properties()

def test_rfreg_shadow_trees(precalculated_rf_regression_explainer):
    dt = precalculated_rf_regression_explainer.shadow_trees
    assert isinstance(dt, list)
    assert isinstance(dt[0], dtreeviz.models.shadow_decision_tree.ShadowDecTree)

def test_rfreg_decisionpath_df(precalculated_rf_regression_explainer, test_names):
    df = precalculated_rf_regression_explainer.get_decisionpath_df(tree_idx=0, index=0)
    assert isinstance(df, pd.DataFrame)

    df = precalculated_rf_regression_explainer.get_decisionpath_df(tree_idx=0, index=test_names[0])
    assert isinstance(df, pd.DataFrame)

def test_rfreg_plot_trees(precalculated_rf_regression_explainer, test_names):
    fig = precalculated_rf_regression_explainer.plot_trees(index=0)
    assert isinstance(fig, go.Figure)

    fig = precalculated_rf_regression_explainer.plot_trees(index=test_names[0])
    assert isinstance(fig, go.Figure)

    fig = precalculated_rf_regression_explainer.plot_trees(index=test_names[0], highlight_tree=0)
    assert isinstance(fig, go.Figure)

def test_rfreg_calculate_properties(precalculated_rf_regression_explainer):
    precalculated_rf_regression_explainer.calculate_properties()