
import pytest

import pandas as pd
import numpy as np

import shap
import plotly.graph_objects as go


def test_linreg_explainer_len(precalculated_linear_regression_explainer, testlen):
    assert (len(precalculated_linear_regression_explainer) == testlen)

def test_linreg_int_idx(precalculated_linear_regression_explainer, test_names):
    assert (precalculated_linear_regression_explainer.get_idx(test_names[0]) == 0)

def test_linreg_random_index(precalculated_linear_regression_explainer):
    assert isinstance(precalculated_linear_regression_explainer.random_index(), int)
    assert isinstance(precalculated_linear_regression_explainer.random_index(return_str=True), str)

def test_linreg_preds(precalculated_linear_regression_explainer):
    assert isinstance(precalculated_linear_regression_explainer.preds, np.ndarray)

def test_linreg_pred_percentiles(precalculated_linear_regression_explainer):
    assert isinstance(precalculated_linear_regression_explainer.pred_percentiles(), np.ndarray)

def test_linreg_permutation_importances(precalculated_linear_regression_explainer):
    assert isinstance(precalculated_linear_regression_explainer.get_permutation_importances_df(), pd.DataFrame)

def test_linreg_metrics(precalculated_linear_regression_explainer):
    assert isinstance(precalculated_linear_regression_explainer.metrics(), dict)
    assert isinstance(precalculated_linear_regression_explainer.metrics_descriptions(), dict)

def test_linreg_mean_abs_shap_df(precalculated_linear_regression_explainer):
    assert isinstance(precalculated_linear_regression_explainer.get_mean_abs_shap_df(), pd.DataFrame)

def test_linreg_top_interactions(precalculated_linear_regression_explainer):
    assert isinstance(precalculated_linear_regression_explainer.top_shap_interactions("Age"), list)
    assert isinstance(precalculated_linear_regression_explainer.top_shap_interactions("Age", topx=4), list)

def test_linreg_contrib_df(precalculated_linear_regression_explainer):
    assert isinstance(precalculated_linear_regression_explainer.get_contrib_df(0), pd.DataFrame)
    assert isinstance(precalculated_linear_regression_explainer.get_contrib_df(0, topx=3), pd.DataFrame)

def test_linreg_shap_base_value(precalculated_linear_regression_explainer):
    assert isinstance(precalculated_linear_regression_explainer.shap_base_value(), (np.floating, float))

def test_linreg_shap_values_shape(precalculated_linear_regression_explainer):
    assert (precalculated_linear_regression_explainer.get_shap_values_df().shape == (len(precalculated_linear_regression_explainer), len(precalculated_linear_regression_explainer.merged_cols)))

def test_linreg_shap_values(precalculated_linear_regression_explainer):
    assert isinstance(precalculated_linear_regression_explainer.get_shap_values_df(), pd.DataFrame)

def test_linreg_mean_abs_shap(precalculated_linear_regression_explainer):
    assert isinstance(precalculated_linear_regression_explainer.get_mean_abs_shap_df(), pd.DataFrame)

def test_linreg_calculate_properties(precalculated_linear_regression_explainer):
    precalculated_linear_regression_explainer.calculate_properties(include_interactions=False)

def test_linreg_pdp_df(precalculated_linear_regression_explainer):
    assert isinstance(precalculated_linear_regression_explainer.pdp_df("Age"), pd.DataFrame)
    assert isinstance(precalculated_linear_regression_explainer.pdp_df("Gender"), pd.DataFrame)
    assert isinstance(precalculated_linear_regression_explainer.pdp_df("Deck"), pd.DataFrame)
    assert isinstance(precalculated_linear_regression_explainer.pdp_df("Age", index=0), pd.DataFrame)
    assert isinstance(precalculated_linear_regression_explainer.pdp_df("Gender", index=0), pd.DataFrame)



def test_logreg_preds(precalculated_logistic_regression_explainer):
    assert isinstance(precalculated_logistic_regression_explainer.preds, np.ndarray)

def test_logreg_pred_percentiles(precalculated_logistic_regression_explainer):
    assert isinstance(precalculated_logistic_regression_explainer.pred_percentiles(), np.ndarray)

def test_logreg_columns_ranked_by_shap(precalculated_logistic_regression_explainer):
    assert isinstance(precalculated_logistic_regression_explainer.columns_ranked_by_shap(), list)

def test_logreg_permutation_importances(precalculated_logistic_regression_explainer):
    assert isinstance(precalculated_logistic_regression_explainer.get_permutation_importances_df(), pd.DataFrame)

def test_logreg_metrics(precalculated_logistic_regression_explainer):
    assert isinstance(precalculated_logistic_regression_explainer.metrics(), dict)
    assert isinstance(precalculated_logistic_regression_explainer.metrics_descriptions(), dict)

def test_logreg_mean_abs_shap_df(precalculated_logistic_regression_explainer):
    assert isinstance(precalculated_logistic_regression_explainer.get_mean_abs_shap_df(), pd.DataFrame)

def test_logreg_contrib_df(precalculated_logistic_regression_explainer):
    assert isinstance(precalculated_logistic_regression_explainer.get_contrib_df(0), pd.DataFrame)
    assert isinstance(precalculated_logistic_regression_explainer.get_contrib_df(0, topx=3), pd.DataFrame)

def test_logreg_shap_base_value(precalculated_logistic_regression_explainer):
    assert isinstance(precalculated_logistic_regression_explainer.shap_base_value(), (np.floating, float))

def test_logreg_shap_values_shape(precalculated_logistic_regression_explainer):
    assert (precalculated_logistic_regression_explainer.get_shap_values_df().shape == (len(precalculated_logistic_regression_explainer), len(precalculated_logistic_regression_explainer.merged_cols)))

def test_logreg_shap_values(precalculated_logistic_regression_explainer):
    assert isinstance(precalculated_logistic_regression_explainer.get_shap_values_df(), pd.DataFrame)

def test_logreg_mean_abs_shap(precalculated_logistic_regression_explainer):
    assert isinstance(precalculated_logistic_regression_explainer.get_mean_abs_shap_df(), pd.DataFrame)

def test_logreg_calculate_properties(precalculated_logistic_regression_explainer):
    precalculated_logistic_regression_explainer.calculate_properties(include_interactions=False)

def test_logreg_pdp_df(precalculated_logistic_regression_explainer):
    assert isinstance(precalculated_logistic_regression_explainer.pdp_df("Age"), pd.DataFrame)
    assert isinstance(precalculated_logistic_regression_explainer.pdp_df("Gender"), pd.DataFrame)
    assert isinstance(precalculated_logistic_regression_explainer.pdp_df("Deck"), pd.DataFrame)
    assert isinstance(precalculated_logistic_regression_explainer.pdp_df("Age", index=0), pd.DataFrame)
    assert isinstance(precalculated_logistic_regression_explainer.pdp_df("Gender", index=0), pd.DataFrame)

def test_logreg_pos_label(precalculated_logistic_regression_explainer):
    precalculated_logistic_regression_explainer.pos_label = 1
    precalculated_logistic_regression_explainer.pos_label = "Not survived"
    assert isinstance(precalculated_logistic_regression_explainer.pos_label, int)
    assert isinstance(precalculated_logistic_regression_explainer.pos_label_str, str)
    assert (precalculated_logistic_regression_explainer.pos_label == 0)
    assert (precalculated_logistic_regression_explainer.pos_label_str == "Not survived")

def test_logreg_pred_probas(precalculated_logistic_regression_explainer):
    assert isinstance(precalculated_logistic_regression_explainer.pred_probas(), np.ndarray)

def test_logreg_metrics(precalculated_logistic_regression_explainer):
    assert isinstance(precalculated_logistic_regression_explainer.metrics(), dict)
    assert isinstance(precalculated_logistic_regression_explainer.metrics(cutoff=0.9), dict)

def test_logreg_precision_df(precalculated_logistic_regression_explainer):
    assert isinstance(precalculated_logistic_regression_explainer.get_precision_df(), pd.DataFrame)
    assert isinstance(precalculated_logistic_regression_explainer.get_precision_df(multiclass=True), pd.DataFrame)
    assert isinstance(precalculated_logistic_regression_explainer.get_precision_df(quantiles=4), pd.DataFrame)

def test_logreg_lift_curve_df(precalculated_logistic_regression_explainer):
    assert isinstance(precalculated_logistic_regression_explainer.get_liftcurve_df(), pd.DataFrame)



##### KERNEL TESTS

def test_logistic_regression_kernel_shap_values(logistic_regression_kernel_explainer):
    assert isinstance(logistic_regression_kernel_explainer.shap_base_value(), (np.floating, float))
    assert (logistic_regression_kernel_explainer.get_shap_values_df().shape == (len(logistic_regression_kernel_explainer), len(logistic_regression_kernel_explainer.merged_cols)))
    assert isinstance(logistic_regression_kernel_explainer.get_shap_values_df(), pd.DataFrame)

def test_linear_regression_kernel_shap_values(linear_regression_kernel_explainer):
    assert isinstance(linear_regression_kernel_explainer.shap_base_value(), (np.floating, float))
    assert (linear_regression_kernel_explainer.get_shap_values_df().shape == (len(linear_regression_kernel_explainer), len(linear_regression_kernel_explainer.merged_cols)))
    assert isinstance(linear_regression_kernel_explainer.get_shap_values_df(), pd.DataFrame)
