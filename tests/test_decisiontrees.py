import pandas as pd
import numpy as np

def test_dtreg_preds(precalculated_dt_regression_explainer):
    assert isinstance(precalculated_dt_regression_explainer.preds, np.ndarray)

def test_dtreg_permutation_importances(precalculated_dt_regression_explainer):
    assert isinstance(precalculated_dt_regression_explainer.get_permutation_importances_df(), pd.DataFrame)

def test_dtreg_shap_base_value(precalculated_dt_regression_explainer):
    assert isinstance(precalculated_dt_regression_explainer.shap_base_value(), (np.floating, float))

def test_dtreg_shap_values_shape(precalculated_dt_regression_explainer):
    assert (precalculated_dt_regression_explainer.get_shap_values_df().shape == (len(precalculated_dt_regression_explainer), len(precalculated_dt_regression_explainer.merged_cols)))

def test_dtreg_shap_values(precalculated_dt_regression_explainer):
    assert isinstance(precalculated_dt_regression_explainer.get_shap_values_df(), pd.DataFrame)

def test_dtreg_mean_abs_shap(precalculated_dt_regression_explainer):
    assert isinstance(precalculated_dt_regression_explainer.get_mean_abs_shap_df(), pd.DataFrame)

def test_dtreg_calculate_properties(precalculated_dt_regression_explainer):
    precalculated_dt_regression_explainer.calculate_properties(include_interactions=False)

def test_dtreg_pdp_df(precalculated_dt_regression_explainer):
    assert isinstance(precalculated_dt_regression_explainer.pdp_df("Age"), pd.DataFrame)
    assert isinstance(precalculated_dt_regression_explainer.pdp_df("Gender"), pd.DataFrame)
    assert isinstance(precalculated_dt_regression_explainer.pdp_df("Deck"), pd.DataFrame)
    assert isinstance(precalculated_dt_regression_explainer.pdp_df("Age", index=0), pd.DataFrame)
    assert isinstance(precalculated_dt_regression_explainer.pdp_df("Gender", index=0), pd.DataFrame)

def test_etreg_preds(precalculated_et_regression_explainer):
    assert isinstance(precalculated_et_regression_explainer.preds, np.ndarray)

def test_etreg_permutation_importances(precalculated_et_regression_explainer):
    assert isinstance(precalculated_et_regression_explainer.get_permutation_importances_df(), pd.DataFrame)

def test_etreg_shap_base_value(precalculated_et_regression_explainer):
    assert isinstance(precalculated_et_regression_explainer.shap_base_value(), (np.floating, float))

def test_etreg_shap_values_shape(precalculated_et_regression_explainer):
    assert (precalculated_et_regression_explainer.get_shap_values_df().shape == (len(precalculated_et_regression_explainer), len(precalculated_et_regression_explainer.merged_cols)))

def test_etreg_shap_values(precalculated_et_regression_explainer):
    assert isinstance(precalculated_et_regression_explainer.get_shap_values_df(), pd.DataFrame)

def test_etreg_mean_abs_shap(precalculated_et_regression_explainer):
    assert isinstance(precalculated_et_regression_explainer.get_mean_abs_shap_df(), pd.DataFrame)

def test_etreg_calculate_properties(precalculated_et_regression_explainer):
    precalculated_et_regression_explainer.calculate_properties(include_interactions=False)

def test_etreg_pdp_df(precalculated_et_regression_explainer):
    assert isinstance(precalculated_et_regression_explainer.pdp_df("Age"), pd.DataFrame)
    assert isinstance(precalculated_et_regression_explainer.pdp_df("Gender"), pd.DataFrame)
    assert isinstance(precalculated_et_regression_explainer.pdp_df("Deck"), pd.DataFrame)
    assert isinstance(precalculated_et_regression_explainer.pdp_df("Age", index=0), pd.DataFrame)
    assert isinstance(precalculated_et_regression_explainer.pdp_df("Gender", index=0), pd.DataFrame)



def test_dtclas_preds(precalculated_dt_classifier_explainer):
    assert isinstance(precalculated_dt_classifier_explainer.preds, np.ndarray)

def test_dtclas_pred_probas(precalculated_dt_classifier_explainer):
    assert isinstance(precalculated_dt_classifier_explainer.pred_probas(), np.ndarray)

def test_dtclas_permutation_importances(precalculated_dt_classifier_explainer):
    assert isinstance(precalculated_dt_classifier_explainer.get_permutation_importances_df(), pd.DataFrame)

def test_dtclas_shap_base_value(precalculated_dt_classifier_explainer):
    assert isinstance(precalculated_dt_classifier_explainer.shap_base_value(), (np.floating, float))

def test_dtclas_shap_values_shape(precalculated_dt_classifier_explainer):
    assert (precalculated_dt_classifier_explainer.get_shap_values_df().shape == (len(precalculated_dt_classifier_explainer), len(precalculated_dt_classifier_explainer.merged_cols)))

def test_dtclas_shap_values(precalculated_dt_classifier_explainer):
    assert isinstance(precalculated_dt_classifier_explainer.get_shap_values_df(), pd.DataFrame)

def test_dtclas_shap_values_all_probabilities(precalculated_dt_classifier_explainer):
    assert (precalculated_dt_classifier_explainer.shap_base_value() >= 0)
    assert (precalculated_dt_classifier_explainer.shap_base_value() <= 1)
    assert (np.all(precalculated_dt_classifier_explainer.get_shap_values_df().sum(axis=1) + precalculated_dt_classifier_explainer.shap_base_value() >= 0))
    assert (np.all(precalculated_dt_classifier_explainer.get_shap_values_df().sum(axis=1) + precalculated_dt_classifier_explainer.shap_base_value() <= 1))

def test_dtclas_mean_abs_shap(precalculated_dt_classifier_explainer):
    assert isinstance(precalculated_dt_classifier_explainer.get_mean_abs_shap_df(), pd.DataFrame)

def test_dtclas_calculate_properties(precalculated_dt_classifier_explainer):
    precalculated_dt_classifier_explainer.calculate_properties(include_interactions=False)

def test_dtclas_pdp_df(precalculated_dt_classifier_explainer):
    assert isinstance(precalculated_dt_classifier_explainer.pdp_df("Age"), pd.DataFrame)
    assert isinstance(precalculated_dt_classifier_explainer.pdp_df("Gender"), pd.DataFrame)
    assert isinstance(precalculated_dt_classifier_explainer.pdp_df("Deck"), pd.DataFrame)
    assert isinstance(precalculated_dt_classifier_explainer.pdp_df("Age", index=0), pd.DataFrame)
    assert isinstance(precalculated_dt_classifier_explainer.pdp_df("Gender", index=0), pd.DataFrame)

def test_dtclas_metrics(precalculated_dt_classifier_explainer):
    assert isinstance(precalculated_dt_classifier_explainer.metrics(), dict)
    assert isinstance(precalculated_dt_classifier_explainer.metrics(cutoff=0.9), dict)

def test_dtclas_precision_df(precalculated_dt_classifier_explainer):
    assert isinstance(precalculated_dt_classifier_explainer.get_precision_df(), pd.DataFrame)
    assert isinstance(precalculated_dt_classifier_explainer.get_precision_df(multiclass=True), pd.DataFrame)
    assert isinstance(precalculated_dt_classifier_explainer.get_precision_df(quantiles=4), pd.DataFrame)

def test_dtclas_lift_curve_df(precalculated_dt_classifier_explainer):
    assert isinstance(precalculated_dt_classifier_explainer.get_liftcurve_df(), pd.DataFrame)

def test_dtclas_prediction_result_df(precalculated_dt_classifier_explainer):
    assert isinstance(precalculated_dt_classifier_explainer.prediction_result_df(0), pd.DataFrame)

def test_etclas_preds(precalculated_et_classifier_explainer):
    assert isinstance(precalculated_et_classifier_explainer.preds, np.ndarray)

def test_etclas_pred_probas(precalculated_et_classifier_explainer):
    assert isinstance(precalculated_et_classifier_explainer.pred_probas(), np.ndarray)

def test_etclas_permutation_importances(precalculated_et_classifier_explainer):
    assert isinstance(precalculated_et_classifier_explainer.get_permutation_importances_df(), pd.DataFrame)

def test_etclas_shap_base_value(precalculated_et_classifier_explainer):
    assert isinstance(precalculated_et_classifier_explainer.shap_base_value(), (np.floating, float))

def test_etclas_shap_values_shape(precalculated_et_classifier_explainer):
    assert (precalculated_et_classifier_explainer.get_shap_values_df().shape == (len(precalculated_et_classifier_explainer), len(precalculated_et_classifier_explainer.merged_cols)))

def test_etclas_shap_values(precalculated_et_classifier_explainer):
    assert isinstance(precalculated_et_classifier_explainer.get_shap_values_df(), pd.DataFrame)

def test_etclas_shap_values_all_probabilities(precalculated_et_classifier_explainer):
    assert (precalculated_et_classifier_explainer.shap_base_value() >= 0)
    assert (precalculated_et_classifier_explainer.shap_base_value() <= 1)
    assert (np.all(precalculated_et_classifier_explainer.get_shap_values_df().sum(axis=1) + precalculated_et_classifier_explainer.shap_base_value() >= 0))
    assert (np.all(precalculated_et_classifier_explainer.get_shap_values_df().sum(axis=1) + precalculated_et_classifier_explainer.shap_base_value() <= 1))

def test_etclas_mean_abs_shap(precalculated_et_classifier_explainer):
    assert isinstance(precalculated_et_classifier_explainer.get_mean_abs_shap_df(), pd.DataFrame)

def test_etclas_calculate_properties(precalculated_et_classifier_explainer):
    precalculated_et_classifier_explainer.calculate_properties(include_interactions=False)

def test_etclas_pdp_df(precalculated_et_classifier_explainer):
    assert isinstance(precalculated_et_classifier_explainer.pdp_df("Age"), pd.DataFrame)
    assert isinstance(precalculated_et_classifier_explainer.pdp_df("Gender"), pd.DataFrame)
    assert isinstance(precalculated_et_classifier_explainer.pdp_df("Deck"), pd.DataFrame)
    assert isinstance(precalculated_et_classifier_explainer.pdp_df("Age", index=0), pd.DataFrame)
    assert isinstance(precalculated_et_classifier_explainer.pdp_df("Gender", index=0), pd.DataFrame)

def test_etclas_metrics(precalculated_et_classifier_explainer):
    assert isinstance(precalculated_et_classifier_explainer.metrics(), dict)
    assert isinstance(precalculated_et_classifier_explainer.metrics(cutoff=0.9), dict)

def test_etclas_precision_df(precalculated_et_classifier_explainer):
    assert isinstance(precalculated_et_classifier_explainer.get_precision_df(), pd.DataFrame)
    assert isinstance(precalculated_et_classifier_explainer.get_precision_df(multiclass=True), pd.DataFrame)
    assert isinstance(precalculated_et_classifier_explainer.get_precision_df(quantiles=4), pd.DataFrame)

def test_etclas_lift_curve_df(precalculated_et_classifier_explainer):
    assert isinstance(precalculated_et_classifier_explainer.get_liftcurve_df(), pd.DataFrame)

def test_etclas_prediction_result_df(precalculated_et_classifier_explainer):
    assert isinstance(precalculated_et_classifier_explainer.prediction_result_df(0), pd.DataFrame)
