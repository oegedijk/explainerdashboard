import pytest

import pandas as pd
import numpy as np

import plotly.graph_objects as go


def test_preds(precalculated_rf_multiclass_explainer):
    assert isinstance(precalculated_rf_multiclass_explainer.preds, np.ndarray)

def test_pred_percentiles(precalculated_rf_multiclass_explainer):
    assert isinstance(precalculated_rf_multiclass_explainer.pred_percentiles(), np.ndarray)

def test_columns_ranked_by_shap(precalculated_rf_multiclass_explainer):
    assert isinstance(precalculated_rf_multiclass_explainer.columns_ranked_by_shap(), list)

def test_permutation_importances(precalculated_rf_multiclass_explainer):
    assert isinstance(precalculated_rf_multiclass_explainer.get_permutation_importances_df(), pd.DataFrame)
    
def test_X_cats(precalculated_rf_multiclass_explainer):
    assert isinstance(precalculated_rf_multiclass_explainer.X_cats, pd.DataFrame)

def test_metrics(precalculated_rf_multiclass_explainer):
    assert isinstance(precalculated_rf_multiclass_explainer.metrics(), dict)
    assert isinstance(precalculated_rf_multiclass_explainer.metrics_descriptions(), dict)

def test_mean_abs_shap_df(precalculated_rf_multiclass_explainer):
    assert isinstance(precalculated_rf_multiclass_explainer.get_mean_abs_shap_df(), pd.DataFrame)

def test_top_interactions(precalculated_rf_multiclass_explainer):
    assert isinstance(precalculated_rf_multiclass_explainer.top_shap_interactions("Age"), list)
    assert isinstance(precalculated_rf_multiclass_explainer.top_shap_interactions("Age", topx=4), list)

def test_permutation_importances_df(precalculated_rf_multiclass_explainer):
    assert isinstance(precalculated_rf_multiclass_explainer.get_permutation_importances_df(), pd.DataFrame)
    assert isinstance(precalculated_rf_multiclass_explainer.get_permutation_importances_df(topx=3), pd.DataFrame)
    assert isinstance(precalculated_rf_multiclass_explainer.get_permutation_importances_df(cutoff=0.01), pd.DataFrame)

def test_contrib_df(precalculated_rf_multiclass_explainer):
    assert isinstance(precalculated_rf_multiclass_explainer.get_contrib_df(0), pd.DataFrame)
    assert isinstance(precalculated_rf_multiclass_explainer.get_contrib_df(0, topx=3), pd.DataFrame)

def test_contrib_summary_df(precalculated_rf_multiclass_explainer):
    assert isinstance(precalculated_rf_multiclass_explainer.get_contrib_summary_df(0), pd.DataFrame)
    assert isinstance(precalculated_rf_multiclass_explainer.get_contrib_summary_df(0, topx=3), pd.DataFrame)
    assert isinstance(precalculated_rf_multiclass_explainer.get_contrib_summary_df(0, round=3), pd.DataFrame)

def test_shap_base_value(precalculated_rf_multiclass_explainer):
    assert isinstance(precalculated_rf_multiclass_explainer.shap_base_value(), (np.floating, float))

def test_shap_values_shape(precalculated_rf_multiclass_explainer):
    assert (precalculated_rf_multiclass_explainer.get_shap_values_df().shape == (len(precalculated_rf_multiclass_explainer), len(precalculated_rf_multiclass_explainer.merged_cols)))

def test_shap_values(precalculated_rf_multiclass_explainer):
    assert isinstance(precalculated_rf_multiclass_explainer.get_shap_values_df(), pd.DataFrame)

def test_shap_interaction_values(precalculated_rf_multiclass_explainer):
    assert isinstance(precalculated_rf_multiclass_explainer.shap_interaction_values(), np.ndarray)

def test_mean_abs_shap(precalculated_rf_multiclass_explainer):
    assert isinstance(precalculated_rf_multiclass_explainer.get_mean_abs_shap_df(), pd.DataFrame)

def test_calculate_properties(precalculated_rf_multiclass_explainer):
    precalculated_rf_multiclass_explainer.calculate_properties()

def test_shap_interaction_values_by_col(precalculated_rf_multiclass_explainer):
    assert isinstance(precalculated_rf_multiclass_explainer.shap_interaction_values_for_col("Age"), np.ndarray)
    assert (precalculated_rf_multiclass_explainer.shap_interaction_values_for_col("Age").shape == 
                    precalculated_rf_multiclass_explainer.get_shap_values_df().shape)

def test_pdp_df(precalculated_rf_multiclass_explainer):
    assert isinstance(precalculated_rf_multiclass_explainer.pdp_df("Age"), pd.DataFrame)
    assert isinstance(precalculated_rf_multiclass_explainer.pdp_df("Gender"), pd.DataFrame)
    assert isinstance(precalculated_rf_multiclass_explainer.pdp_df("Deck"), pd.DataFrame)
    assert isinstance(precalculated_rf_multiclass_explainer.pdp_df("Age", index=0), pd.DataFrame)
    assert isinstance(precalculated_rf_multiclass_explainer.pdp_df("Gender", index=0), pd.DataFrame)

def test_plot_importances(precalculated_rf_multiclass_explainer):
    fig = precalculated_rf_multiclass_explainer.plot_importances()
    assert isinstance(fig, go.Figure)

    fig = precalculated_rf_multiclass_explainer.plot_importances(kind='permutation')
    assert isinstance(fig, go.Figure)

    fig = precalculated_rf_multiclass_explainer.plot_importances(topx=3)
    assert isinstance(fig, go.Figure)

def test_plot_interactions(precalculated_rf_multiclass_explainer):
    fig = precalculated_rf_multiclass_explainer.plot_interactions_importance("Age")
    assert isinstance(fig, go.Figure)

    fig = precalculated_rf_multiclass_explainer.plot_interactions_importance("Age")
    assert isinstance(fig, go.Figure)

    fig = precalculated_rf_multiclass_explainer.plot_interactions_importance("Gender")
    assert isinstance(fig, go.Figure)

def test_plot_contributions(precalculated_rf_multiclass_explainer):
    fig = precalculated_rf_multiclass_explainer.plot_contributions(0)
    assert isinstance(fig, go.Figure)

    fig = precalculated_rf_multiclass_explainer.plot_contributions(0, topx=3)
    assert isinstance(fig, go.Figure)

    fig = precalculated_rf_multiclass_explainer.plot_contributions(0, cutoff=0.05)
    assert isinstance(fig, go.Figure)

def test_plot_shap_detailed(precalculated_rf_multiclass_explainer):
    fig = precalculated_rf_multiclass_explainer.plot_importances_detailed()
    assert isinstance(fig, go.Figure)

    fig = precalculated_rf_multiclass_explainer.plot_importances_detailed(topx=3)
    assert isinstance(fig, go.Figure)

def test_plot_interactions_detailed(precalculated_rf_multiclass_explainer):
    fig = precalculated_rf_multiclass_explainer.plot_interactions_detailed("Age")
    assert isinstance(fig, go.Figure)

    fig = precalculated_rf_multiclass_explainer.plot_interactions_detailed("Age", topx=3)
    assert isinstance(fig, go.Figure)

    fig = precalculated_rf_multiclass_explainer.plot_interactions_detailed("Age")
    assert isinstance(fig, go.Figure)

    fig = precalculated_rf_multiclass_explainer.plot_interactions_detailed("Gender")
    assert isinstance(fig, go.Figure)

def test_plot_dependence(precalculated_rf_multiclass_explainer):
    fig = precalculated_rf_multiclass_explainer.plot_dependence("Age")
    assert isinstance(fig, go.Figure)

    fig = precalculated_rf_multiclass_explainer.plot_dependence("Age", "Gender")
    assert isinstance(fig, go.Figure)

    fig = precalculated_rf_multiclass_explainer.plot_dependence("Age", highlight_index=0)
    assert isinstance(fig, go.Figure)

    fig = precalculated_rf_multiclass_explainer.plot_dependence("Gender", highlight_index=0)
    assert isinstance(fig, go.Figure)

def test_plot_interaction(precalculated_rf_multiclass_explainer):
    fig = precalculated_rf_multiclass_explainer.plot_dependence("Gender", "Age")
    assert isinstance(fig, go.Figure)

    fig = precalculated_rf_multiclass_explainer.plot_dependence("Age", "Gender")
    assert isinstance(fig, go.Figure)

    fig = precalculated_rf_multiclass_explainer.plot_dependence("Age", "Gender", highlight_index=0)
    assert isinstance(fig, go.Figure)

def test_plot_pdp(precalculated_rf_multiclass_explainer):
    fig = precalculated_rf_multiclass_explainer.plot_pdp("Age")
    assert isinstance(fig, go.Figure)

    fig = precalculated_rf_multiclass_explainer.plot_pdp("Gender")
    assert isinstance(fig, go.Figure)

    fig = precalculated_rf_multiclass_explainer.plot_pdp("Gender", index=0)
    assert isinstance(fig, go.Figure)

    fig = precalculated_rf_multiclass_explainer.plot_pdp("Age", index=0)
    assert isinstance(fig, go.Figure)

def test_pos_label(precalculated_rf_multiclass_explainer):
    precalculated_rf_multiclass_explainer.pos_label = 1
    precalculated_rf_multiclass_explainer.pos_label = "Southampton"
    assert isinstance(precalculated_rf_multiclass_explainer.pos_label, int)
    assert isinstance(precalculated_rf_multiclass_explainer.pos_label_str, str)
    assert (precalculated_rf_multiclass_explainer.pos_label == 1)
    assert (precalculated_rf_multiclass_explainer.pos_label_str == "Southampton")

def test_pred_probas(precalculated_rf_multiclass_explainer):
    assert isinstance(precalculated_rf_multiclass_explainer.pred_probas(), np.ndarray)

def test_metrics(precalculated_rf_multiclass_explainer):
    assert isinstance(precalculated_rf_multiclass_explainer.metrics(), dict)
    assert isinstance(precalculated_rf_multiclass_explainer.metrics(cutoff=0.9), dict)

def test_precision_df(precalculated_rf_multiclass_explainer):
    assert isinstance(precalculated_rf_multiclass_explainer.get_precision_df(), pd.DataFrame)
    assert isinstance(precalculated_rf_multiclass_explainer.get_precision_df(multiclass=True), pd.DataFrame)
    assert isinstance(precalculated_rf_multiclass_explainer.get_precision_df(quantiles=4), pd.DataFrame)

def test_lift_curve_df(precalculated_rf_multiclass_explainer):
    assert isinstance(precalculated_rf_multiclass_explainer.get_liftcurve_df(), pd.DataFrame)

def test_keep_shap_pos_label_only(precalculated_rf_multiclass_explainer):
    precalculated_rf_multiclass_explainer.keep_shap_pos_label_only()
    assert isinstance(precalculated_rf_multiclass_explainer.get_shap_values_df(), pd.DataFrame)

def test_calculate_properties(precalculated_rf_multiclass_explainer):
    precalculated_rf_multiclass_explainer.calculate_properties()
    
def test_plot_precision(precalculated_rf_multiclass_explainer):
    fig = precalculated_rf_multiclass_explainer.plot_precision()
    assert isinstance(fig, go.Figure)

    fig = precalculated_rf_multiclass_explainer.plot_precision(multiclass=True)
    assert isinstance(fig, go.Figure)

    fig = precalculated_rf_multiclass_explainer.plot_precision(quantiles=10, cutoff=0.5)
    assert isinstance(fig, go.Figure)

def test_plot_cumulative_precision(precalculated_rf_multiclass_explainer):
    fig = precalculated_rf_multiclass_explainer.plot_cumulative_precision()
    assert isinstance(fig, go.Figure)

def test_plot_confusion_matrix(precalculated_rf_multiclass_explainer):
    fig = precalculated_rf_multiclass_explainer.plot_confusion_matrix(percentage=False, binary=False)
    assert isinstance(fig, go.Figure)

    fig = precalculated_rf_multiclass_explainer.plot_confusion_matrix(percentage=False, binary=True)
    assert isinstance(fig, go.Figure)

    fig = precalculated_rf_multiclass_explainer.plot_confusion_matrix(percentage=True, binary=False)
    assert isinstance(fig, go.Figure)

    fig = precalculated_rf_multiclass_explainer.plot_confusion_matrix(percentage=True, binary=True)
    assert isinstance(fig, go.Figure)

    fig = precalculated_rf_multiclass_explainer.plot_confusion_matrix(normalize='all')
    assert isinstance(fig, go.Figure)

    fig = precalculated_rf_multiclass_explainer.plot_confusion_matrix(normalize='observed')
    assert isinstance(fig, go.Figure)

    fig = precalculated_rf_multiclass_explainer.plot_confusion_matrix(normalize='pred')
    assert isinstance(fig, go.Figure)

def test_plot_lift_curve(precalculated_rf_multiclass_explainer):
    fig = precalculated_rf_multiclass_explainer.plot_lift_curve()
    assert isinstance(fig, go.Figure)

    fig = precalculated_rf_multiclass_explainer.plot_lift_curve(percentage=True)
    assert isinstance(fig, go.Figure)

    fig = precalculated_rf_multiclass_explainer.plot_lift_curve(cutoff=0.5)
    assert isinstance(fig, go.Figure)

def test_plot_lift_curve(precalculated_rf_multiclass_explainer):
    fig = precalculated_rf_multiclass_explainer.plot_lift_curve()
    assert isinstance(fig, go.Figure)

    fig = precalculated_rf_multiclass_explainer.plot_lift_curve(percentage=True)
    assert isinstance(fig, go.Figure)

    fig = precalculated_rf_multiclass_explainer.plot_lift_curve(cutoff=0.5)
    assert isinstance(fig, go.Figure)

def test_plot_classification(precalculated_rf_multiclass_explainer):
    fig = precalculated_rf_multiclass_explainer.plot_classification()
    assert isinstance(fig, go.Figure)

    fig = precalculated_rf_multiclass_explainer.plot_classification(percentage=True)
    assert isinstance(fig, go.Figure)

    fig = precalculated_rf_multiclass_explainer.plot_classification(cutoff=0)
    assert isinstance(fig, go.Figure)

    fig = precalculated_rf_multiclass_explainer.plot_classification(cutoff=1)
    assert isinstance(fig, go.Figure)

def test_plot_roc_auc(precalculated_rf_multiclass_explainer):
    fig = precalculated_rf_multiclass_explainer.plot_roc_auc(0.5)
    assert isinstance(fig, go.Figure)

    fig = precalculated_rf_multiclass_explainer.plot_roc_auc(0.0)
    assert isinstance(fig, go.Figure)

    fig = precalculated_rf_multiclass_explainer.plot_roc_auc(1.0)
    assert isinstance(fig, go.Figure)

def test_plot_pr_auc(precalculated_rf_multiclass_explainer):
    fig = precalculated_rf_multiclass_explainer.plot_pr_auc(0.5)
    assert isinstance(fig, go.Figure)

    fig = precalculated_rf_multiclass_explainer.plot_pr_auc(0.0)
    assert isinstance(fig, go.Figure)

    fig = precalculated_rf_multiclass_explainer.plot_pr_auc(1.0)
    assert isinstance(fig, go.Figure)
