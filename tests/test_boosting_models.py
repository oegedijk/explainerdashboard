import pandas as pd
import numpy as np

def test_xgbreg_preds(precalculated_xgb_regression_explainer):
    assert isinstance(precalculated_xgb_regression_explainer.preds, np.ndarray)

def test_xgbreg_permutation_importances(precalculated_xgb_regression_explainer):
    assert isinstance(precalculated_xgb_regression_explainer.get_permutation_importances_df(), pd.DataFrame)

def test_xgbreg_shap_base_value(precalculated_xgb_regression_explainer):
    assert isinstance(precalculated_xgb_regression_explainer.shap_base_value(), (np.floating, float))

def test_xgbreg_shap_values_shape(precalculated_xgb_regression_explainer):
    assert (precalculated_xgb_regression_explainer.get_shap_values_df().shape == (len(precalculated_xgb_regression_explainer), len(precalculated_xgb_regression_explainer.merged_cols)))

def test_xgbreg_shap_values(precalculated_xgb_regression_explainer):
    assert isinstance(precalculated_xgb_regression_explainer.get_shap_values_df(), pd.DataFrame)

def test_xgbreg_mean_abs_shap(precalculated_xgb_regression_explainer):
    assert isinstance(precalculated_xgb_regression_explainer.get_mean_abs_shap_df(), pd.DataFrame)

def test_xgbreg_calculate_properties(precalculated_xgb_regression_explainer):
    precalculated_xgb_regression_explainer.calculate_properties(include_interactions=False)

def test_xgbreg_pdp_df(precalculated_xgb_regression_explainer):
    assert isinstance(precalculated_xgb_regression_explainer.pdp_df("Age"), pd.DataFrame)
    assert isinstance(precalculated_xgb_regression_explainer.pdp_df("Gender"), pd.DataFrame)
    assert isinstance(precalculated_xgb_regression_explainer.pdp_df("Deck"), pd.DataFrame)
    assert isinstance(precalculated_xgb_regression_explainer.pdp_df("Age", index=0), pd.DataFrame)
    assert isinstance(precalculated_xgb_regression_explainer.pdp_df("Gender", index=0), pd.DataFrame)


def test_lgbmreg_preds(precalculated_lgbm_regression_explainer):
    assert isinstance(precalculated_lgbm_regression_explainer.preds, np.ndarray)

def test_lgbmreg_permutation_importances(precalculated_lgbm_regression_explainer):
    assert isinstance(precalculated_lgbm_regression_explainer.get_permutation_importances_df(), pd.DataFrame)

def test_lgbmreg_shap_base_value(precalculated_lgbm_regression_explainer):
    assert isinstance(precalculated_lgbm_regression_explainer.shap_base_value(), (np.floating, float))

def test_lgbmreg_shap_values_shape(precalculated_lgbm_regression_explainer):
    assert (precalculated_lgbm_regression_explainer.get_shap_values_df().shape == (len(precalculated_lgbm_regression_explainer), len(precalculated_lgbm_regression_explainer.merged_cols)))

def test_lgbmreg_shap_values(precalculated_lgbm_regression_explainer):
    assert isinstance(precalculated_lgbm_regression_explainer.get_shap_values_df(), pd.DataFrame)

def test_lgbmreg_mean_abs_shap(precalculated_lgbm_regression_explainer):
    assert isinstance(precalculated_lgbm_regression_explainer.get_mean_abs_shap_df(), pd.DataFrame)

def test_lgbmreg_calculate_properties(precalculated_lgbm_regression_explainer):
    precalculated_lgbm_regression_explainer.calculate_properties(include_interactions=False)

def test_lgbmreg_pdp_df(precalculated_lgbm_regression_explainer):
    assert isinstance(precalculated_lgbm_regression_explainer.pdp_df("Age"), pd.DataFrame)
    assert isinstance(precalculated_lgbm_regression_explainer.pdp_df("Gender"), pd.DataFrame)
    assert isinstance(precalculated_lgbm_regression_explainer.pdp_df("Deck"), pd.DataFrame)
    assert isinstance(precalculated_lgbm_regression_explainer.pdp_df("Age", index=0), pd.DataFrame)
    assert isinstance(precalculated_lgbm_regression_explainer.pdp_df("Gender", index=0), pd.DataFrame)


def test_lgbmclas_preds(precalculated_lgbm_classifier_explainer):
    assert isinstance(precalculated_lgbm_classifier_explainer.preds, np.ndarray)

def test_lgbmclas_pred_probas(precalculated_lgbm_classifier_explainer):
    assert isinstance(precalculated_lgbm_classifier_explainer.pred_probas(), np.ndarray)

def test_lgbmclas_permutation_importances(precalculated_lgbm_classifier_explainer):
    assert isinstance(precalculated_lgbm_classifier_explainer.get_permutation_importances_df(), pd.DataFrame)

def test_lgbmclas_shap_base_value(precalculated_lgbm_classifier_explainer):
    assert isinstance(precalculated_lgbm_classifier_explainer.shap_base_value(), (np.floating, float))

def test_lgbmclas_shap_values_shape(precalculated_lgbm_classifier_explainer):
    assert (precalculated_lgbm_classifier_explainer.get_shap_values_df().shape == (len(precalculated_lgbm_classifier_explainer), len(precalculated_lgbm_classifier_explainer.merged_cols)))

def test_lgbmclas_shap_values(precalculated_lgbm_classifier_explainer):
    assert isinstance(precalculated_lgbm_classifier_explainer.get_shap_values_df(), pd.DataFrame)

def test_lgbmclas_shap_values_all_probabilities(precalculated_lgbm_classifier_explainer):
    assert (precalculated_lgbm_classifier_explainer.shap_base_value() >= 0)
    assert (precalculated_lgbm_classifier_explainer.shap_base_value() <= 1)
    assert (np.all(precalculated_lgbm_classifier_explainer.get_shap_values_df().sum(axis=1) + precalculated_lgbm_classifier_explainer.shap_base_value() >= 0))
    assert (np.all(precalculated_lgbm_classifier_explainer.get_shap_values_df().sum(axis=1) + precalculated_lgbm_classifier_explainer.shap_base_value() <= 1))

def test_lgbmclas_mean_abs_shap(precalculated_lgbm_classifier_explainer):
    assert isinstance(precalculated_lgbm_classifier_explainer.get_mean_abs_shap_df(), pd.DataFrame)

def test_lgbmclas_calculate_properties(precalculated_lgbm_classifier_explainer):
    precalculated_lgbm_classifier_explainer.calculate_properties(include_interactions=False)

def test_lgbmclas_pdp_df(precalculated_lgbm_classifier_explainer):
    assert isinstance(precalculated_lgbm_classifier_explainer.pdp_df("Age"), pd.DataFrame)
    assert isinstance(precalculated_lgbm_classifier_explainer.pdp_df("Gender"), pd.DataFrame)
    assert isinstance(precalculated_lgbm_classifier_explainer.pdp_df("Deck"), pd.DataFrame)
    assert isinstance(precalculated_lgbm_classifier_explainer.pdp_df("Age", index=0), pd.DataFrame)
    assert isinstance(precalculated_lgbm_classifier_explainer.pdp_df("Gender", index=0), pd.DataFrame)

def test_lgbmclas_metrics(precalculated_lgbm_classifier_explainer):
    assert isinstance(precalculated_lgbm_classifier_explainer.metrics(), dict)
    assert isinstance(precalculated_lgbm_classifier_explainer.metrics(cutoff=0.9), dict)

def test_lgbmclas_precision_df(precalculated_lgbm_classifier_explainer):
    assert isinstance(precalculated_lgbm_classifier_explainer.get_precision_df(), pd.DataFrame)
    assert isinstance(precalculated_lgbm_classifier_explainer.get_precision_df(multiclass=True), pd.DataFrame)
    assert isinstance(precalculated_lgbm_classifier_explainer.get_precision_df(quantiles=4), pd.DataFrame)

def test_lgbmclas_lift_curve_df(precalculated_lgbm_classifier_explainer):
    assert isinstance(precalculated_lgbm_classifier_explainer.get_liftcurve_df(), pd.DataFrame)

def test_lgbmclas_prediction_result_df(precalculated_lgbm_classifier_explainer):
    assert isinstance(precalculated_lgbm_classifier_explainer.prediction_result_df(0), pd.DataFrame)



def test_xgbclas_preds(precalculated_xgb_classifier_explainer):
    assert isinstance(precalculated_xgb_classifier_explainer.preds, np.ndarray)

def test_xgbclas_pred_probas(precalculated_xgb_classifier_explainer):
    assert isinstance(precalculated_xgb_classifier_explainer.pred_probas(), np.ndarray)

def test_xgbclas_permutation_importances(precalculated_xgb_classifier_explainer):
    assert isinstance(precalculated_xgb_classifier_explainer.get_permutation_importances_df(), pd.DataFrame)

def test_xgbclas_shap_base_value(precalculated_xgb_classifier_explainer):
    assert isinstance(precalculated_xgb_classifier_explainer.shap_base_value(), (np.floating, float))

def test_xgbclas_shap_values_shape(precalculated_xgb_classifier_explainer):
    assert (precalculated_xgb_classifier_explainer.get_shap_values_df().shape == (len(precalculated_xgb_classifier_explainer), len(precalculated_xgb_classifier_explainer.merged_cols)))

def test_xgbclas_shap_values(precalculated_xgb_classifier_explainer):
    assert isinstance(precalculated_xgb_classifier_explainer.get_shap_values_df(), pd.DataFrame)

def test_xgbclas_shap_values_all_probabilities(precalculated_xgb_classifier_explainer):
    assert (precalculated_xgb_classifier_explainer.shap_base_value() >= 0)
    assert (precalculated_xgb_classifier_explainer.shap_base_value() <= 1)
    assert (np.all(precalculated_xgb_classifier_explainer.get_shap_values_df().sum(axis=1) + precalculated_xgb_classifier_explainer.shap_base_value() >= 0))
    assert (np.all(precalculated_xgb_classifier_explainer.get_shap_values_df().sum(axis=1) + precalculated_xgb_classifier_explainer.shap_base_value() <= 1))

def test_xgbclas_mean_abs_shap(precalculated_xgb_classifier_explainer):
    assert isinstance(precalculated_xgb_classifier_explainer.get_mean_abs_shap_df(), pd.DataFrame)

def test_xgbclas_calculate_properties(precalculated_xgb_classifier_explainer):
    precalculated_xgb_classifier_explainer.calculate_properties(include_interactions=False)

def test_xgbclas_pdp_df(precalculated_xgb_classifier_explainer):
    assert isinstance(precalculated_xgb_classifier_explainer.pdp_df("Age"), pd.DataFrame)
    assert isinstance(precalculated_xgb_classifier_explainer.pdp_df("Gender"), pd.DataFrame)
    assert isinstance(precalculated_xgb_classifier_explainer.pdp_df("Deck"), pd.DataFrame)
    assert isinstance(precalculated_xgb_classifier_explainer.pdp_df("Age", index=0), pd.DataFrame)
    assert isinstance(precalculated_xgb_classifier_explainer.pdp_df("Gender", index=0), pd.DataFrame)

def test_xgbclas_metrics(precalculated_xgb_classifier_explainer):
    assert isinstance(precalculated_xgb_classifier_explainer.metrics(), dict)
    assert isinstance(precalculated_xgb_classifier_explainer.metrics(cutoff=0.9), dict)

def test_xgbclas_precision_df(precalculated_xgb_classifier_explainer):
    assert isinstance(precalculated_xgb_classifier_explainer.get_precision_df(), pd.DataFrame)
    assert isinstance(precalculated_xgb_classifier_explainer.get_precision_df(multiclass=True), pd.DataFrame)
    assert isinstance(precalculated_xgb_classifier_explainer.get_precision_df(quantiles=4), pd.DataFrame)

def test_xgbclas_lift_curve_df(precalculated_xgb_classifier_explainer):
    assert isinstance(precalculated_xgb_classifier_explainer.get_liftcurve_df(), pd.DataFrame)

def test_xgbclas_prediction_result_df(precalculated_xgb_classifier_explainer):
    assert isinstance(precalculated_xgb_classifier_explainer.prediction_result_df(0), pd.DataFrame)