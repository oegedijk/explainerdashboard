import pandas as pd
import numpy as np
from pandas.api.types import is_categorical_dtype, is_numeric_dtype

import plotly.graph_objects as go


def test_explainer_len(precalculated_rf_regression_explainer, testlen):
    assert (len(precalculated_rf_regression_explainer) == testlen)

def test_int_idx(precalculated_rf_regression_explainer, test_names):
    assert (precalculated_rf_regression_explainer.get_idx(test_names[0]) == 0)

def test_random_index(precalculated_rf_regression_explainer):
    assert isinstance(precalculated_rf_regression_explainer.random_index(), int)
    assert isinstance(precalculated_rf_regression_explainer.random_index(return_str=True), str)

def test_index_exists(precalculated_rf_regression_explainer):
    assert (precalculated_rf_regression_explainer.index_exists(0))
    assert (precalculated_rf_regression_explainer.index_exists(precalculated_rf_regression_explainer.idxs[0]))
    assert not (precalculated_rf_regression_explainer.index_exists('bla'))

def test_row_from_input(precalculated_rf_regression_explainer):
    input_row = precalculated_rf_regression_explainer.get_row_from_input(
        precalculated_rf_regression_explainer.X.iloc[[0]].values.tolist())
    assert isinstance(input_row, pd.DataFrame)

    input_row = precalculated_rf_regression_explainer.get_row_from_input(
        precalculated_rf_regression_explainer.X_merged.iloc[[0]].values.tolist())
    assert isinstance(input_row, pd.DataFrame)

    input_row = precalculated_rf_regression_explainer.get_row_from_input(
        precalculated_rf_regression_explainer.X_merged
        [precalculated_rf_regression_explainer.columns_ranked_by_shap()]
        .iloc[[0]].values.tolist(), ranked_by_shap=True)
    assert isinstance(input_row, pd.DataFrame)
    
def test_prediction_result_df(precalculated_rf_regression_explainer):
    df = precalculated_rf_regression_explainer.prediction_result_df(0)
    assert isinstance(df, pd.DataFrame)

def test_preds(precalculated_rf_regression_explainer):
    assert isinstance(precalculated_rf_regression_explainer.preds, np.ndarray)

def test_pred_percentiles(precalculated_rf_regression_explainer):
    assert isinstance(precalculated_rf_regression_explainer.pred_percentiles(), np.ndarray)

def test_columns_ranked_by_shap(precalculated_rf_regression_explainer):
    assert isinstance(precalculated_rf_regression_explainer.columns_ranked_by_shap(), list)

def test_get_col(precalculated_rf_regression_explainer):
    assert isinstance(precalculated_rf_regression_explainer.get_col("Gender"), pd.Series)
    assert (is_categorical_dtype(precalculated_rf_regression_explainer.get_col("Gender")))

    assert isinstance(precalculated_rf_regression_explainer.get_col("Age"), pd.Series)
    assert (is_numeric_dtype(precalculated_rf_regression_explainer.get_col("Age")))

def test_permutation_importances(precalculated_rf_regression_explainer):
    assert isinstance(precalculated_rf_regression_explainer.permutation_importances(), pd.DataFrame)

def test_X_cats(precalculated_rf_regression_explainer):
    assert isinstance(precalculated_rf_regression_explainer.X_cats, pd.DataFrame)

def test_metrics(precalculated_rf_regression_explainer):
    assert isinstance(precalculated_rf_regression_explainer.metrics(), dict)
    assert isinstance(precalculated_rf_regression_explainer.metrics_descriptions(), dict)

def test_mean_abs_shap_df(precalculated_rf_regression_explainer):
    assert isinstance(precalculated_rf_regression_explainer.mean_abs_shap_df(), pd.DataFrame)

def test_top_interactions(precalculated_rf_regression_explainer):
    assert isinstance(precalculated_rf_regression_explainer.top_shap_interactions("Age"), list)
    assert isinstance(precalculated_rf_regression_explainer.top_shap_interactions("Age", topx=4), list)

def test_permutation_importances_df(precalculated_rf_regression_explainer):
    assert isinstance(precalculated_rf_regression_explainer.get_permutation_importances_df(), pd.DataFrame)
    assert isinstance(precalculated_rf_regression_explainer.get_permutation_importances_df(topx=3), pd.DataFrame)
    assert isinstance(precalculated_rf_regression_explainer.get_permutation_importances_df(cutoff=0.01), pd.DataFrame)

def test_contrib_df(precalculated_rf_regression_explainer):
    assert isinstance(precalculated_rf_regression_explainer.get_contrib_df(0), pd.DataFrame)
    assert isinstance(precalculated_rf_regression_explainer.get_contrib_df(0, topx=3), pd.DataFrame)
    assert isinstance(precalculated_rf_regression_explainer.get_contrib_df(0, sort='high-to-low'), pd.DataFrame)
    assert isinstance(precalculated_rf_regression_explainer.get_contrib_df(0, sort='low-to-high'), pd.DataFrame)
    assert isinstance(precalculated_rf_regression_explainer.get_contrib_df(0, sort='importance'), pd.DataFrame)
    assert isinstance(precalculated_rf_regression_explainer.get_contrib_df(X_row=precalculated_rf_regression_explainer.X.iloc[[0]]), pd.DataFrame)

def test_contrib_summary_df(precalculated_rf_regression_explainer):
    assert isinstance(precalculated_rf_regression_explainer.get_contrib_summary_df(0), pd.DataFrame)
    assert isinstance(precalculated_rf_regression_explainer.get_contrib_summary_df(0, topx=3), pd.DataFrame)
    assert isinstance(precalculated_rf_regression_explainer.get_contrib_summary_df(0, round=3), pd.DataFrame)
    assert isinstance(precalculated_rf_regression_explainer.get_contrib_summary_df(0, sort='high-to-low'), pd.DataFrame)
    assert isinstance(precalculated_rf_regression_explainer.get_contrib_summary_df(0, sort='low-to-high'), pd.DataFrame)
    assert isinstance(precalculated_rf_regression_explainer.get_contrib_summary_df(0, sort='importance'), pd.DataFrame)
    assert isinstance(precalculated_rf_regression_explainer.get_contrib_summary_df(X_row=precalculated_rf_regression_explainer.X.iloc[[0]]), pd.DataFrame)

def test_shap_base_value(precalculated_rf_regression_explainer):
    assert isinstance(precalculated_rf_regression_explainer.shap_base_value(), (np.floating, float))

def test_shap_values_shape(precalculated_rf_regression_explainer):
    assert (precalculated_rf_regression_explainer.get_shap_values_df().shape == (len(precalculated_rf_regression_explainer), len(precalculated_rf_regression_explainer.merged_cols)))

def test_shap_values(precalculated_rf_regression_explainer):
    assert isinstance(precalculated_rf_regression_explainer.get_shap_values_df(), pd.DataFrame)

def test_shap_interaction_values(precalculated_rf_regression_explainer):
    assert isinstance(precalculated_rf_regression_explainer.shap_interaction_values(), np.ndarray)

def test_mean_abs_shap(precalculated_rf_regression_explainer):
    assert isinstance(precalculated_rf_regression_explainer.mean_abs_shap_df(), pd.DataFrame)

def test_memory_usage(precalculated_rf_regression_explainer):
    assert isinstance(precalculated_rf_regression_explainer.memory_usage(), pd.DataFrame)
    assert isinstance(precalculated_rf_regression_explainer.memory_usage(cutoff=1000), pd.DataFrame)

def test_calculate_properties(precalculated_rf_regression_explainer):
    precalculated_rf_regression_explainer.calculate_properties()

def test_shap_interaction_values_for_col(precalculated_rf_regression_explainer):
    assert isinstance(precalculated_rf_regression_explainer.shap_interaction_values_for_col("Age"), np.ndarray)
    assert (precalculated_rf_regression_explainer.shap_interaction_values_for_col("Age").shape ==
                    precalculated_rf_regression_explainer.get_shap_values_df().shape)

def test_pdp_df(precalculated_rf_regression_explainer):
    assert isinstance(precalculated_rf_regression_explainer.pdp_df("Age"), pd.DataFrame)
    assert isinstance(precalculated_rf_regression_explainer.pdp_df("Gender"), pd.DataFrame)
    assert isinstance(precalculated_rf_regression_explainer.pdp_df("Deck"), pd.DataFrame)
    assert isinstance(precalculated_rf_regression_explainer.pdp_df("Age", index=0), pd.DataFrame)
    assert isinstance(precalculated_rf_regression_explainer.pdp_df("Gender", index=0), pd.DataFrame)

def test_plot_importances(precalculated_rf_regression_explainer):
    fig = precalculated_rf_regression_explainer.plot_importances()
    assert isinstance(fig, go.Figure)

    fig = precalculated_rf_regression_explainer.plot_importances(kind='permutation')
    assert isinstance(fig, go.Figure)

    fig = precalculated_rf_regression_explainer.plot_importances(topx=3)
    assert isinstance(fig, go.Figure)

def test_plot_interactions(precalculated_rf_regression_explainer):
    fig = precalculated_rf_regression_explainer.plot_interactions_importance("Age")
    assert isinstance(fig, go.Figure)

    fig = precalculated_rf_regression_explainer.plot_interactions_importance("Age")
    assert isinstance(fig, go.Figure)

    fig = precalculated_rf_regression_explainer.plot_interactions_importance("Gender")
    assert isinstance(fig, go.Figure)

def test_plot_shap_interactions(precalculated_rf_regression_explainer):
    fig = precalculated_rf_regression_explainer.plot_contributions(0)
    assert isinstance(fig, go.Figure)

    fig = precalculated_rf_regression_explainer.plot_contributions(0, topx=3)
    assert isinstance(fig, go.Figure)

    fig = precalculated_rf_regression_explainer.plot_contributions(0, cutoff=0.05)
    assert isinstance(fig, go.Figure)

    fig = precalculated_rf_regression_explainer.plot_contributions(0, sort='high-to-low')
    assert isinstance(fig, go.Figure)

    fig = precalculated_rf_regression_explainer.plot_contributions(0, sort='low-to-high')
    assert isinstance(fig, go.Figure)

    fig = precalculated_rf_regression_explainer.plot_contributions(0, sort='importance')
    assert isinstance(fig, go.Figure)

def test_plot_shap_detailed(precalculated_rf_regression_explainer):
    fig = precalculated_rf_regression_explainer.plot_importances_detailed()
    assert isinstance(fig, go.Figure)

    fig = precalculated_rf_regression_explainer.plot_importances_detailed(topx=3)
    assert isinstance(fig, go.Figure)


def test_plot_interactions_detailed(precalculated_rf_regression_explainer):
    fig = precalculated_rf_regression_explainer.plot_interactions_detailed("Age")
    assert isinstance(fig, go.Figure)

    fig = precalculated_rf_regression_explainer.plot_interactions_detailed("Age", topx=3)
    assert isinstance(fig, go.Figure)

def test_plot_dependence(precalculated_rf_regression_explainer):
    fig = precalculated_rf_regression_explainer.plot_dependence("Age")
    assert isinstance(fig, go.Figure)

    fig = precalculated_rf_regression_explainer.plot_dependence("Gender")
    assert isinstance(fig, go.Figure)

    fig = precalculated_rf_regression_explainer.plot_dependence("Age", "Gender")
    assert isinstance(fig, go.Figure)

    fig = precalculated_rf_regression_explainer.plot_dependence("Age", highlight_index=0)
    assert isinstance(fig, go.Figure)

    fig = precalculated_rf_regression_explainer.plot_dependence("Gender", highlight_index=0)
    assert isinstance(fig, go.Figure)

def test_plot_contributions(precalculated_rf_regression_explainer):
    fig = precalculated_rf_regression_explainer.plot_contributions(0)
    assert isinstance(fig, go.Figure)

    fig = precalculated_rf_regression_explainer.plot_contributions(0, topx=3)
    assert isinstance(fig, go.Figure)

    fig = precalculated_rf_regression_explainer.plot_contributions(0, sort='high-to-low')
    assert isinstance(fig, go.Figure)

    fig = precalculated_rf_regression_explainer.plot_contributions(0, sort='low-to-high')
    assert isinstance(fig, go.Figure)

    fig = precalculated_rf_regression_explainer.plot_contributions(0, sort='importance')
    assert isinstance(fig, go.Figure)

    fig = precalculated_rf_regression_explainer.plot_contributions(X_row=precalculated_rf_regression_explainer.X.iloc[[0]])
    assert isinstance(fig, go.Figure)


def test_plot_interaction(precalculated_rf_regression_explainer):
    fig = precalculated_rf_regression_explainer.plot_interaction("Age", "Gender")
    assert isinstance(fig, go.Figure)

    fig = precalculated_rf_regression_explainer.plot_interaction("Gender", "Age")
    assert isinstance(fig, go.Figure)

    fig = precalculated_rf_regression_explainer.plot_interaction("Gender", "Age")
    assert isinstance(fig, go.Figure)

    fig = precalculated_rf_regression_explainer.plot_interaction("Age", "Gender", highlight_index=0)
    assert isinstance(fig, go.Figure)

def test_plot_pdp(precalculated_rf_regression_explainer):
    fig = precalculated_rf_regression_explainer.plot_pdp("Age")
    assert isinstance(fig, go.Figure)

    fig = precalculated_rf_regression_explainer.plot_pdp("Gender")
    assert isinstance(fig, go.Figure)

    fig = precalculated_rf_regression_explainer.plot_pdp("Gender", index=0)
    assert isinstance(fig, go.Figure)

    fig = precalculated_rf_regression_explainer.plot_pdp("Age", index=0)
    assert isinstance(fig, go.Figure)

    fig = precalculated_rf_regression_explainer.plot_pdp("Age", X_row=precalculated_rf_regression_explainer.X.iloc[[0]])
    assert isinstance(fig, go.Figure)

def test_yaml(precalculated_rf_regression_explainer):
    yaml = precalculated_rf_regression_explainer.to_yaml()
    assert isinstance(yaml, str)








