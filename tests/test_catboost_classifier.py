import pandas as pd
import numpy as np
from pandas.api.types import is_categorical_dtype, is_numeric_dtype

import plotly.graph_objs as go


def test_explainer_len(precalculated_catboost_classifier_explainer, testlen):
    assert (len(precalculated_catboost_classifier_explainer) == testlen)

def test_int_idx(precalculated_catboost_classifier_explainer, test_names):
    assert (precalculated_catboost_classifier_explainer.get_idx(test_names[0]) == 0)

def test_random_index(precalculated_catboost_classifier_explainer):
    assert isinstance(precalculated_catboost_classifier_explainer.random_index(), int)
    assert isinstance(precalculated_catboost_classifier_explainer.random_index(return_str=True), str)

def test_ordered_cats(precalculated_catboost_classifier_explainer):
    assert (precalculated_catboost_classifier_explainer.ordered_cats("Sex") == ['NOT_ENCODED', 'Sex_female', 'Sex_male'])
    assert (precalculated_catboost_classifier_explainer.ordered_cats("Deck", topx=2, sort='alphabet') == ['Deck_A', 'Deck_B'])

    assert isinstance(precalculated_catboost_classifier_explainer.ordered_cats("Deck", sort='freq'), list)
    assert isinstance(precalculated_catboost_classifier_explainer.ordered_cats("Deck", topx=3, sort='freq'), list)
    assert isinstance(precalculated_catboost_classifier_explainer.ordered_cats("Deck", sort='shap'), list)
    assert isinstance(precalculated_catboost_classifier_explainer.ordered_cats("Deck", topx=3, sort='shap'), list)

def test_preds(precalculated_catboost_classifier_explainer):
    assert isinstance(precalculated_catboost_classifier_explainer.preds, np.ndarray)

def test_row_from_input(precalculated_catboost_classifier_explainer):
    input_row = precalculated_catboost_classifier_explainer.get_row_from_input(
        precalculated_catboost_classifier_explainer.X.iloc[[0]].values.tolist())
    assert isinstance(input_row, pd.DataFrame)

    input_row = precalculated_catboost_classifier_explainer.get_row_from_input(
        precalculated_catboost_classifier_explainer.X_merged
        [precalculated_catboost_classifier_explainer.columns_ranked_by_shap()]
        .iloc[[0]].values.tolist(), ranked_by_shap=True)
    assert isinstance(input_row, pd.DataFrame)
    

def test_pred_percentiles(precalculated_catboost_classifier_explainer):
    assert isinstance(precalculated_catboost_classifier_explainer.pred_percentiles(), np.ndarray)

def test_columns_ranked_by_shap(precalculated_catboost_classifier_explainer):
    assert isinstance(precalculated_catboost_classifier_explainer.columns_ranked_by_shap(), list)

def test_permutation_importances(precalculated_catboost_classifier_explainer):
    assert isinstance(precalculated_catboost_classifier_explainer.get_permutation_importances_df(), pd.DataFrame)
    
def test_X_cats(precalculated_catboost_classifier_explainer):
    assert isinstance(precalculated_catboost_classifier_explainer.X_cats, pd.DataFrame)

def test_metrics(precalculated_catboost_classifier_explainer):
    assert isinstance(precalculated_catboost_classifier_explainer.metrics(), dict)

def test_mean_abs_shap_df(precalculated_catboost_classifier_explainer):
    assert isinstance(precalculated_catboost_classifier_explainer.get_mean_abs_shap_df(), pd.DataFrame)

def test_top_interactions(precalculated_catboost_classifier_explainer):
    assert isinstance(precalculated_catboost_classifier_explainer.top_shap_interactions("Age"), list)
    assert isinstance(precalculated_catboost_classifier_explainer.top_shap_interactions("Age", topx=4), list)

def test_permutation_importances_df(precalculated_catboost_classifier_explainer):
    assert isinstance(precalculated_catboost_classifier_explainer.get_permutation_importances_df(), pd.DataFrame)
    assert isinstance(precalculated_catboost_classifier_explainer.get_permutation_importances_df(topx=3), pd.DataFrame)
    assert isinstance(precalculated_catboost_classifier_explainer.get_permutation_importances_df(cutoff=0.01), pd.DataFrame)

def test_contrib_df(precalculated_catboost_classifier_explainer):
    assert isinstance(precalculated_catboost_classifier_explainer.get_contrib_df(0), pd.DataFrame)
    assert isinstance(precalculated_catboost_classifier_explainer.get_contrib_df(X_row=precalculated_catboost_classifier_explainer.X.iloc[[0]]), pd.DataFrame)

def test_shap_base_value(precalculated_catboost_classifier_explainer):
    assert isinstance(precalculated_catboost_classifier_explainer.shap_base_value(), (np.floating, float))

def test_shap_values_shape(precalculated_catboost_classifier_explainer):
    assert (precalculated_catboost_classifier_explainer.get_shap_values_df().shape == (len(precalculated_catboost_classifier_explainer), len(precalculated_catboost_classifier_explainer.merged_cols)))

def test_shap_values(precalculated_catboost_classifier_explainer):
    assert isinstance(precalculated_catboost_classifier_explainer.get_shap_values_df(), pd.DataFrame)

def test_calculate_properties(precalculated_catboost_classifier_explainer):
    precalculated_catboost_classifier_explainer.calculate_properties()

def test_prediction_result_df(precalculated_catboost_classifier_explainer):
    df = precalculated_catboost_classifier_explainer.prediction_result_df(0)
    assert isinstance(df, pd.DataFrame)

def test_pdp_df(precalculated_catboost_classifier_explainer):
    assert isinstance(precalculated_catboost_classifier_explainer.pdp_df("Age"), pd.DataFrame)
    assert isinstance(precalculated_catboost_classifier_explainer.pdp_df("Deck"), pd.DataFrame)
    assert isinstance(precalculated_catboost_classifier_explainer.pdp_df("Age", index=0), pd.DataFrame)
    assert isinstance(precalculated_catboost_classifier_explainer.pdp_df("Age", X_row=precalculated_catboost_classifier_explainer.X.iloc[[0]]), pd.DataFrame)

def test_plot_importances(precalculated_catboost_classifier_explainer):
    fig = precalculated_catboost_classifier_explainer.plot_importances()
    assert isinstance(fig, go.Figure)

    fig = precalculated_catboost_classifier_explainer.plot_importances(kind='permutation')
    assert isinstance(fig, go.Figure)

    fig = precalculated_catboost_classifier_explainer.plot_importances(topx=3)
    assert isinstance(fig, go.Figure)

def test_plot_contributions(precalculated_catboost_classifier_explainer):
    fig = precalculated_catboost_classifier_explainer.plot_contributions(0)
    assert isinstance(fig, go.Figure)

    fig = precalculated_catboost_classifier_explainer.plot_contributions(X_row=precalculated_catboost_classifier_explainer.X.iloc[[0]], sort='importance')
    assert isinstance(fig, go.Figure)


def test_plot_shap_detailed(precalculated_catboost_classifier_explainer):
    fig = precalculated_catboost_classifier_explainer.plot_importances_detailed()
    assert isinstance(fig, go.Figure)


def test_plot_dependence(precalculated_catboost_classifier_explainer):
    fig = precalculated_catboost_classifier_explainer.plot_dependence("Age")
    assert isinstance(fig, go.Figure)

    fig = precalculated_catboost_classifier_explainer.plot_dependence("Sex")
    assert isinstance(fig, go.Figure)

    fig = precalculated_catboost_classifier_explainer.plot_dependence("Age", "Sex")
    assert isinstance(fig, go.Figure)

    fig = precalculated_catboost_classifier_explainer.plot_dependence("Sex", "Sex")
    assert isinstance(fig, go.Figure)



def test_plot_pdp(precalculated_catboost_classifier_explainer):
    fig = precalculated_catboost_classifier_explainer.plot_pdp("Age")
    assert isinstance(fig, go.Figure)

    fig = precalculated_catboost_classifier_explainer.plot_pdp("Sex")
    assert isinstance(fig, go.Figure)

    fig = precalculated_catboost_classifier_explainer.plot_pdp("Sex", index=0)
    assert isinstance(fig, go.Figure)

    fig = precalculated_catboost_classifier_explainer.plot_pdp("Age", index=0)
    assert isinstance(fig, go.Figure)

    fig = precalculated_catboost_classifier_explainer.plot_pdp("Age", X_row=precalculated_catboost_classifier_explainer.X.iloc[[0]])
    assert isinstance(fig, go.Figure)


def test_yaml(precalculated_catboost_classifier_explainer):
    yaml = precalculated_catboost_classifier_explainer.to_yaml()
    assert isinstance(yaml, str)

def test_pos_label(precalculated_catboost_classifier_explainer):
    precalculated_catboost_classifier_explainer.pos_label = 1
    precalculated_catboost_classifier_explainer.pos_label = "Not survived"
    assert isinstance(precalculated_catboost_classifier_explainer.pos_label, int)
    assert isinstance(precalculated_catboost_classifier_explainer.pos_label_str, str)
    assert (precalculated_catboost_classifier_explainer.pos_label == 0)
    assert (precalculated_catboost_classifier_explainer.pos_label_str == "Not survived")

def test_pred_probas(precalculated_catboost_classifier_explainer):
    assert isinstance(precalculated_catboost_classifier_explainer.pred_probas(), np.ndarray)


def test_metrics(precalculated_catboost_classifier_explainer):
    assert isinstance(precalculated_catboost_classifier_explainer.metrics(), dict)
    assert isinstance(precalculated_catboost_classifier_explainer.metrics(cutoff=0.9), dict)
    assert isinstance(precalculated_catboost_classifier_explainer.metrics_descriptions(cutoff=0.9), dict)


def test_precision_df(precalculated_catboost_classifier_explainer):
    assert isinstance(precalculated_catboost_classifier_explainer.get_precision_df(), pd.DataFrame)
    assert isinstance(precalculated_catboost_classifier_explainer.get_precision_df(multiclass=True), pd.DataFrame)
    assert isinstance(precalculated_catboost_classifier_explainer.get_precision_df(quantiles=4), pd.DataFrame)

def test_lift_curve_df(precalculated_catboost_classifier_explainer):
    assert isinstance(precalculated_catboost_classifier_explainer.get_liftcurve_df(), pd.DataFrame)

def test_calculate_properties(precalculated_catboost_classifier_explainer):
    precalculated_catboost_classifier_explainer.calculate_properties()
    
def test_plot_precision(precalculated_catboost_classifier_explainer):
    fig = precalculated_catboost_classifier_explainer.plot_precision()
    assert isinstance(fig, go.Figure)

    fig = precalculated_catboost_classifier_explainer.plot_precision(multiclass=True)
    assert isinstance(fig, go.Figure)

    fig = precalculated_catboost_classifier_explainer.plot_precision(quantiles=10, cutoff=0.5)
    assert isinstance(fig, go.Figure)

def test_plot_cumulative_precision(precalculated_catboost_classifier_explainer):
    fig = precalculated_catboost_classifier_explainer.plot_cumulative_precision()
    assert isinstance(fig, go.Figure)

    fig = precalculated_catboost_classifier_explainer.plot_cumulative_precision(percentile=0.5)
    assert isinstance(fig, go.Figure)

    fig = precalculated_catboost_classifier_explainer.plot_cumulative_precision(percentile=0.1, pos_label=0)
    assert isinstance(fig, go.Figure)

def test_plot_confusion_matrix(precalculated_catboost_classifier_explainer):
    fig = precalculated_catboost_classifier_explainer.plot_confusion_matrix(percentage=False, binary=False)
    assert isinstance(fig, go.Figure)

    fig = precalculated_catboost_classifier_explainer.plot_confusion_matrix(percentage=False, binary=True)
    assert isinstance(fig, go.Figure)

    fig = precalculated_catboost_classifier_explainer.plot_confusion_matrix(percentage=True, binary=False)
    assert isinstance(fig, go.Figure)

    fig = precalculated_catboost_classifier_explainer.plot_confusion_matrix(percentage=True, binary=True)
    assert isinstance(fig, go.Figure)

    fig = precalculated_catboost_classifier_explainer.plot_confusion_matrix(normalize='all')
    assert isinstance(fig, go.Figure)

    fig = precalculated_catboost_classifier_explainer.plot_confusion_matrix(normalize='observed')
    assert isinstance(fig, go.Figure)

    fig = precalculated_catboost_classifier_explainer.plot_confusion_matrix(normalize='pred')
    assert isinstance(fig, go.Figure)


def test_plot_lift_curve(precalculated_catboost_classifier_explainer):
    fig = precalculated_catboost_classifier_explainer.plot_lift_curve()
    assert isinstance(fig, go.Figure)

    fig = precalculated_catboost_classifier_explainer.plot_lift_curve(percentage=True)
    assert isinstance(fig, go.Figure)

    fig = precalculated_catboost_classifier_explainer.plot_lift_curve(cutoff=0.5)
    assert isinstance(fig, go.Figure)

def test_plot_lift_curve(precalculated_catboost_classifier_explainer):
    fig = precalculated_catboost_classifier_explainer.plot_lift_curve()
    assert isinstance(fig, go.Figure)

    fig = precalculated_catboost_classifier_explainer.plot_lift_curve(percentage=True)
    assert isinstance(fig, go.Figure)

    fig = precalculated_catboost_classifier_explainer.plot_lift_curve(cutoff=0.5)
    assert isinstance(fig, go.Figure)

def test_plot_classification(precalculated_catboost_classifier_explainer):
    fig = precalculated_catboost_classifier_explainer.plot_classification()
    assert isinstance(fig, go.Figure)

    fig = precalculated_catboost_classifier_explainer.plot_classification(percentage=True)
    assert isinstance(fig, go.Figure)

    fig = precalculated_catboost_classifier_explainer.plot_classification(cutoff=0)
    assert isinstance(fig, go.Figure)

    fig = precalculated_catboost_classifier_explainer.plot_classification(cutoff=1)
    assert isinstance(fig, go.Figure)

def test_plot_roc_auc(precalculated_catboost_classifier_explainer):
    fig = precalculated_catboost_classifier_explainer.plot_roc_auc(0.5)
    assert isinstance(fig, go.Figure)

    fig = precalculated_catboost_classifier_explainer.plot_roc_auc(0.0)
    assert isinstance(fig, go.Figure)

    fig = precalculated_catboost_classifier_explainer.plot_roc_auc(1.0)
    assert isinstance(fig, go.Figure)

def test_plot_pr_auc(precalculated_catboost_classifier_explainer):
    fig = precalculated_catboost_classifier_explainer.plot_pr_auc(0.5)
    assert isinstance(fig, go.Figure)

    fig = precalculated_catboost_classifier_explainer.plot_pr_auc(0.0)
    assert isinstance(fig, go.Figure)

    fig = precalculated_catboost_classifier_explainer.plot_pr_auc(1.0)
    assert isinstance(fig, go.Figure)

def test_plot_prediction_result(precalculated_catboost_classifier_explainer):
    fig = precalculated_catboost_classifier_explainer.plot_prediction_result(0)
    assert isinstance(fig, go.Figure)