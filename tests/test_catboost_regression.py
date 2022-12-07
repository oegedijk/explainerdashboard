
import pandas as pd
import numpy as np
from pandas.api.types import is_categorical_dtype, is_numeric_dtype

import plotly.graph_objs as go


def test_explainer_len(precalculated_catboost_regression_explainer, testlen):
    assert (len(precalculated_catboost_regression_explainer) == testlen)

def test_int_idx(precalculated_catboost_regression_explainer, test_names):
    assert (precalculated_catboost_regression_explainer.get_idx(test_names[0]) == 0)

def test_random_index(precalculated_catboost_regression_explainer):
    assert isinstance(precalculated_catboost_regression_explainer.random_index(), int)
    assert isinstance(precalculated_catboost_regression_explainer.random_index(return_str=True), str)

def test_row_from_input(precalculated_catboost_regression_explainer):
    input_row = precalculated_catboost_regression_explainer.get_row_from_input(
        precalculated_catboost_regression_explainer.X.iloc[[0]].values.tolist())
    assert isinstance(input_row, pd.DataFrame)
    
    input_row = precalculated_catboost_regression_explainer.get_row_from_input(
        precalculated_catboost_regression_explainer.X_merged
        [precalculated_catboost_regression_explainer.columns_ranked_by_shap()]
        .iloc[[0]].values.tolist(), ranked_by_shap=True)
    assert isinstance(input_row, pd.DataFrame)

def test_prediction_result_df(precalculated_catboost_regression_explainer):
    df = precalculated_catboost_regression_explainer.prediction_result_df(0)
    assert isinstance(df, pd.DataFrame)

def test_preds(precalculated_catboost_regression_explainer):
    assert isinstance(precalculated_catboost_regression_explainer.preds, np.ndarray)

def test_pred_percentiles(precalculated_catboost_regression_explainer):
    assert isinstance(precalculated_catboost_regression_explainer.pred_percentiles(), np.ndarray)

def test_columns_ranked_by_shap(precalculated_catboost_regression_explainer):
    assert isinstance(precalculated_catboost_regression_explainer.columns_ranked_by_shap(), list)

def test_ordered_cats(precalculated_catboost_regression_explainer):
    assert (precalculated_catboost_regression_explainer.ordered_cats("Sex") == ['NOT_ENCODED', 'Sex_female', 'Sex_male'])
    assert (precalculated_catboost_regression_explainer.ordered_cats("Deck", topx=2, sort='alphabet') == ['Deck_A', 'Deck_B'])

    assert isinstance(precalculated_catboost_regression_explainer.ordered_cats("Deck", sort='freq'), list)
    assert isinstance(precalculated_catboost_regression_explainer.ordered_cats("Deck", topx=3, sort='freq'), list)
    assert isinstance(precalculated_catboost_regression_explainer.ordered_cats("Deck", sort='shap'), list)
    assert isinstance(precalculated_catboost_regression_explainer.ordered_cats("Deck", topx=3, sort='shap'), list)

def test_get_col(precalculated_catboost_regression_explainer):
    assert isinstance(precalculated_catboost_regression_explainer.get_col("Sex"), pd.Series)
    assert is_categorical_dtype(precalculated_catboost_regression_explainer.get_col("Sex"))

    assert isinstance(precalculated_catboost_regression_explainer.get_col("Age"), pd.Series)
    assert (is_numeric_dtype(precalculated_catboost_regression_explainer.get_col("Age")))

def test_permutation_importances(precalculated_catboost_regression_explainer):
    assert isinstance(precalculated_catboost_regression_explainer.get_permutation_importances_df(), pd.DataFrame)

def test_X_cats(precalculated_catboost_regression_explainer):
    assert isinstance(precalculated_catboost_regression_explainer.X_cats, pd.DataFrame)

def test_metrics(precalculated_catboost_regression_explainer):
    assert isinstance(precalculated_catboost_regression_explainer.metrics(), dict)
    assert isinstance(precalculated_catboost_regression_explainer.metrics_descriptions(), dict)

def test_mean_abs_shap_df(precalculated_catboost_regression_explainer):
    assert isinstance(precalculated_catboost_regression_explainer.get_mean_abs_shap_df(), pd.DataFrame)

def test_permutation_importances_df(precalculated_catboost_regression_explainer):
    assert isinstance(precalculated_catboost_regression_explainer.get_permutation_importances_df(), pd.DataFrame)
    assert isinstance(precalculated_catboost_regression_explainer.get_permutation_importances_df(topx=3), pd.DataFrame)
    assert isinstance(precalculated_catboost_regression_explainer.get_permutation_importances_df(cutoff=0.01), pd.DataFrame)

def test_contrib_df(precalculated_catboost_regression_explainer):
    assert isinstance(precalculated_catboost_regression_explainer.get_contrib_df(0), pd.DataFrame)
    assert isinstance(precalculated_catboost_regression_explainer.get_contrib_df(0, topx=3), pd.DataFrame)
    assert isinstance(precalculated_catboost_regression_explainer.get_contrib_df(0, sort='importance'), pd.DataFrame)
    assert isinstance(precalculated_catboost_regression_explainer.get_contrib_df(X_row=precalculated_catboost_regression_explainer.X.iloc[[0]]), pd.DataFrame)

def test_contrib_summary_df(precalculated_catboost_regression_explainer):
    assert isinstance(precalculated_catboost_regression_explainer.get_contrib_summary_df(0), pd.DataFrame)
    assert isinstance(precalculated_catboost_regression_explainer.get_contrib_summary_df(0, topx=3), pd.DataFrame)
    assert isinstance(precalculated_catboost_regression_explainer.get_contrib_summary_df(0, round=3), pd.DataFrame)
    assert isinstance(precalculated_catboost_regression_explainer.get_contrib_summary_df(0, sort='high-to-low'), pd.DataFrame)
    assert isinstance(precalculated_catboost_regression_explainer.get_contrib_summary_df(0, sort='low-to-high'), pd.DataFrame)
    assert isinstance(precalculated_catboost_regression_explainer.get_contrib_summary_df(0, sort='importance'), pd.DataFrame)
    assert isinstance(precalculated_catboost_regression_explainer.get_contrib_summary_df(X_row=precalculated_catboost_regression_explainer.X.iloc[[0]]), pd.DataFrame)

def test_shap_base_value(precalculated_catboost_regression_explainer):
    assert isinstance(precalculated_catboost_regression_explainer.shap_base_value(), (np.floating, float))

def test_shap_values_shape(precalculated_catboost_regression_explainer):
    assert (precalculated_catboost_regression_explainer.get_shap_values_df().shape == (len(precalculated_catboost_regression_explainer), len(precalculated_catboost_regression_explainer.merged_cols)))

def test_shap_values(precalculated_catboost_regression_explainer):
    assert isinstance(precalculated_catboost_regression_explainer.get_shap_values_df(), pd.DataFrame)

def test_mean_abs_shap(precalculated_catboost_regression_explainer):
    assert isinstance(precalculated_catboost_regression_explainer.get_mean_abs_shap_df(), pd.DataFrame)

def test_calculate_properties(precalculated_catboost_regression_explainer):
    precalculated_catboost_regression_explainer.calculate_properties()

def test_pdp_df(precalculated_catboost_regression_explainer):
    assert isinstance(precalculated_catboost_regression_explainer.pdp_df("Age"), pd.DataFrame)
    assert isinstance(precalculated_catboost_regression_explainer.pdp_df("Sex"), pd.DataFrame)
    assert isinstance(precalculated_catboost_regression_explainer.pdp_df("Deck"), pd.DataFrame)
    assert isinstance(precalculated_catboost_regression_explainer.pdp_df("Age", index=0), pd.DataFrame)
    assert isinstance(precalculated_catboost_regression_explainer.pdp_df("Sex", index=0), pd.DataFrame)

def test_plot_importances(precalculated_catboost_regression_explainer):
    fig = precalculated_catboost_regression_explainer.plot_importances()
    assert isinstance(fig, go.Figure)

    fig = precalculated_catboost_regression_explainer.plot_importances(kind='permutation')
    assert isinstance(fig, go.Figure)

    fig = precalculated_catboost_regression_explainer.plot_importances(topx=3)
    assert isinstance(fig, go.Figure)

def test_plot_shap_detailed(precalculated_catboost_regression_explainer):
    fig = precalculated_catboost_regression_explainer.plot_importances_detailed()
    assert isinstance(fig, go.Figure)

    fig = precalculated_catboost_regression_explainer.plot_importances_detailed(topx=3)
    assert isinstance(fig, go.Figure)

def test_plot_dependence(precalculated_catboost_regression_explainer):
    fig = precalculated_catboost_regression_explainer.plot_dependence("Age")
    assert isinstance(fig, go.Figure)

    fig = precalculated_catboost_regression_explainer.plot_dependence("Sex")
    assert isinstance(fig, go.Figure)

    fig = precalculated_catboost_regression_explainer.plot_dependence("Age", "Sex")
    assert isinstance(fig, go.Figure)

    fig = precalculated_catboost_regression_explainer.plot_dependence("Age", highlight_index=0)
    assert isinstance(fig, go.Figure)

    fig = precalculated_catboost_regression_explainer.plot_dependence("Sex", highlight_index=0)
    assert isinstance(fig, go.Figure)

    fig = precalculated_catboost_regression_explainer.plot_dependence("Deck", topx=3, sort="freq")
    assert isinstance(fig, go.Figure)

    fig = precalculated_catboost_regression_explainer.plot_dependence("Deck", topx=3, sort="shap")
    assert isinstance(fig, go.Figure)

    fig = precalculated_catboost_regression_explainer.plot_dependence("Deck", sort="freq")
    assert isinstance(fig, go.Figure)

    fig = precalculated_catboost_regression_explainer.plot_dependence("Deck", sort="shap")
    assert isinstance(fig, go.Figure)

def test_plot_contributions(precalculated_catboost_regression_explainer):
    fig = precalculated_catboost_regression_explainer.plot_contributions(0)
    assert isinstance(fig, go.Figure)

    fig = precalculated_catboost_regression_explainer.plot_contributions(0, topx=3)
    assert isinstance(fig, go.Figure)

    fig = precalculated_catboost_regression_explainer.plot_contributions(0, sort='high-to-low')
    assert isinstance(fig, go.Figure)

    fig = precalculated_catboost_regression_explainer.plot_contributions(0, sort='low-to-high')
    assert isinstance(fig, go.Figure)

    fig = precalculated_catboost_regression_explainer.plot_contributions(0, sort='importance')
    assert isinstance(fig, go.Figure)

    fig = precalculated_catboost_regression_explainer.plot_contributions(X_row=precalculated_catboost_regression_explainer.X.iloc[[0]])
    assert isinstance(fig, go.Figure)

def test_plot_pdp(precalculated_catboost_regression_explainer):
    fig = precalculated_catboost_regression_explainer.plot_pdp("Age")
    assert isinstance(fig, go.Figure)

    fig = precalculated_catboost_regression_explainer.plot_pdp("Sex")
    assert isinstance(fig, go.Figure)

    fig = precalculated_catboost_regression_explainer.plot_pdp("Sex", index=0)
    assert isinstance(fig, go.Figure)

    fig = precalculated_catboost_regression_explainer.plot_pdp("Age", index=0)
    assert isinstance(fig, go.Figure)

    fig = precalculated_catboost_regression_explainer.plot_pdp("Age", X_row=precalculated_catboost_regression_explainer.X.iloc[[0]])
    assert isinstance(fig, go.Figure)

def test_yaml(precalculated_catboost_regression_explainer):
    yaml = precalculated_catboost_regression_explainer.to_yaml()
    assert isinstance(yaml, str)

def test_residuals(precalculated_catboost_regression_explainer):
    assert isinstance(precalculated_catboost_regression_explainer.residuals, pd.Series)

def test_metrics(precalculated_catboost_regression_explainer):
    metrics_dict = precalculated_catboost_regression_explainer.metrics()
    assert isinstance(metrics_dict, dict)
    assert ('root-mean-squared-error' in metrics_dict)
    assert ('mean-absolute-error' in metrics_dict)
    assert ('R-squared' in metrics_dict)
    assert isinstance(precalculated_catboost_regression_explainer.metrics_descriptions(), dict) 

def test_plot_predicted_vs_actual(precalculated_catboost_regression_explainer):
    fig = precalculated_catboost_regression_explainer.plot_predicted_vs_actual(logs=False)
    assert isinstance(fig, go.Figure)

    fig = precalculated_catboost_regression_explainer.plot_predicted_vs_actual(logs=True)
    assert isinstance(fig, go.Figure)

    fig = precalculated_catboost_regression_explainer.plot_predicted_vs_actual(log_x=True, log_y=True)
    assert isinstance(fig, go.Figure)

def test_plot_residuals(precalculated_catboost_regression_explainer):
    fig = precalculated_catboost_regression_explainer.plot_residuals()
    assert isinstance(fig, go.Figure)

    fig = precalculated_catboost_regression_explainer.plot_residuals(vs_actual=True)
    assert isinstance(fig, go.Figure)

    fig = precalculated_catboost_regression_explainer.plot_residuals(residuals='ratio')
    assert isinstance(fig, go.Figure)

    fig = precalculated_catboost_regression_explainer.plot_residuals(residuals='log-ratio')
    assert isinstance(fig, go.Figure)

    fig = precalculated_catboost_regression_explainer.plot_residuals(residuals='log-ratio', vs_actual=True)
    assert isinstance(fig, go.Figure)

def test_plot_residuals_vs_feature(precalculated_catboost_regression_explainer):
    fig = precalculated_catboost_regression_explainer.plot_residuals_vs_feature("Age")
    assert isinstance(fig, go.Figure)

    fig = precalculated_catboost_regression_explainer.plot_residuals_vs_feature("Age", residuals='log-ratio')
    assert isinstance(fig, go.Figure)

    fig = precalculated_catboost_regression_explainer.plot_residuals_vs_feature("Age", dropna=True)
    assert isinstance(fig, go.Figure)

    fig = precalculated_catboost_regression_explainer.plot_residuals_vs_feature("Sex", points=False)
    assert isinstance(fig, go.Figure)

    fig = precalculated_catboost_regression_explainer.plot_residuals_vs_feature("Sex", winsor=10)
    assert isinstance(fig, go.Figure)

def test_plot_y_vs_feature(precalculated_catboost_regression_explainer):
    fig = precalculated_catboost_regression_explainer.plot_y_vs_feature("Age")
    assert isinstance(fig, go.Figure)

    fig = precalculated_catboost_regression_explainer.plot_y_vs_feature("Age", dropna=True)
    assert isinstance(fig, go.Figure)

    fig = precalculated_catboost_regression_explainer.plot_y_vs_feature("Sex", points=False)
    assert isinstance(fig, go.Figure)

    fig = precalculated_catboost_regression_explainer.plot_y_vs_feature("Sex", winsor=10)
    assert isinstance(fig, go.Figure)

def test_plot_preds_vs_feature(precalculated_catboost_regression_explainer):
    fig = precalculated_catboost_regression_explainer.plot_preds_vs_feature("Age")
    assert isinstance(fig, go.Figure)

    fig = precalculated_catboost_regression_explainer.plot_preds_vs_feature("Age", dropna=True)
    assert isinstance(fig, go.Figure)

    fig = precalculated_catboost_regression_explainer.plot_preds_vs_feature("Sex", points=False)
    assert isinstance(fig, go.Figure)

    fig = precalculated_catboost_regression_explainer.plot_preds_vs_feature("Sex", winsor=10)
    assert isinstance(fig, go.Figure)


