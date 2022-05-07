import pandas as pd


import plotly.graph_objects as go


def test_residuals(precalculated_rf_regression_explainer):
    assert isinstance(precalculated_rf_regression_explainer.residuals, pd.Series)

def test_metrics(precalculated_rf_regression_explainer):
    metrics_dict = precalculated_rf_regression_explainer.metrics()
    assert isinstance(metrics_dict, dict)
    assert ('root-mean-squared-error' in metrics_dict)
    assert ('mean-absolute-error' in metrics_dict)
    assert ('R-squared' in metrics_dict)
    assert isinstance(precalculated_rf_regression_explainer.metrics_descriptions(), dict) 

def test_plot_predicted_vs_actual(precalculated_rf_regression_explainer):
    fig = precalculated_rf_regression_explainer.plot_predicted_vs_actual(logs=False)
    assert isinstance(fig, go.Figure)

    fig = precalculated_rf_regression_explainer.plot_predicted_vs_actual(logs=True)
    assert isinstance(fig, go.Figure)

    fig = precalculated_rf_regression_explainer.plot_predicted_vs_actual(log_x=True, log_y=True)
    assert isinstance(fig, go.Figure)

def test_plot_residuals(precalculated_rf_regression_explainer):
    fig = precalculated_rf_regression_explainer.plot_residuals()
    assert isinstance(fig, go.Figure)

    fig = precalculated_rf_regression_explainer.plot_residuals(vs_actual=True)
    assert isinstance(fig, go.Figure)

    fig = precalculated_rf_regression_explainer.plot_residuals(residuals='ratio')
    assert isinstance(fig, go.Figure)

    fig = precalculated_rf_regression_explainer.plot_residuals(residuals='log-ratio')
    assert isinstance(fig, go.Figure)

    fig = precalculated_rf_regression_explainer.plot_residuals(residuals='log-ratio', vs_actual=True)
    assert isinstance(fig, go.Figure)

def test_plot_residuals_vs_feature(precalculated_rf_regression_explainer):
    fig = precalculated_rf_regression_explainer.plot_residuals_vs_feature("Age")
    assert isinstance(fig, go.Figure)

    fig = precalculated_rf_regression_explainer.plot_residuals_vs_feature("Age", residuals='log-ratio')
    assert isinstance(fig, go.Figure)

    fig = precalculated_rf_regression_explainer.plot_residuals_vs_feature("Age", dropna=True)
    assert isinstance(fig, go.Figure)

    fig = precalculated_rf_regression_explainer.plot_residuals_vs_feature("Gender", points=False)
    assert isinstance(fig, go.Figure)

    fig = precalculated_rf_regression_explainer.plot_residuals_vs_feature("Gender", winsor=10)
    assert isinstance(fig, go.Figure)

def test_plot_y_vs_feature(precalculated_rf_regression_explainer):
    fig = precalculated_rf_regression_explainer.plot_y_vs_feature("Age")
    assert isinstance(fig, go.Figure)

    fig = precalculated_rf_regression_explainer.plot_y_vs_feature("Age", dropna=True)
    assert isinstance(fig, go.Figure)

    fig = precalculated_rf_regression_explainer.plot_y_vs_feature("Gender", points=False)
    assert isinstance(fig, go.Figure)

    fig = precalculated_rf_regression_explainer.plot_y_vs_feature("Gender", winsor=10)
    assert isinstance(fig, go.Figure)

def test_plot_preds_vs_feature(precalculated_rf_regression_explainer):
    fig = precalculated_rf_regression_explainer.plot_preds_vs_feature("Age")
    assert isinstance(fig, go.Figure)

    fig = precalculated_rf_regression_explainer.plot_preds_vs_feature("Age", dropna=True)
    assert isinstance(fig, go.Figure)

    fig = precalculated_rf_regression_explainer.plot_preds_vs_feature("Gender", points=False)
    assert isinstance(fig, go.Figure)

    fig = precalculated_rf_regression_explainer.plot_preds_vs_feature("Gender", winsor=10)
    assert isinstance(fig, go.Figure)
