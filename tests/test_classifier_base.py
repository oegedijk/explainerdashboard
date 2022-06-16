import pytest

import pandas as pd
import numpy as np
from pandas.api.types import is_categorical_dtype, is_numeric_dtype


import plotly.graph_objects as go

from explainerdashboard import ClassifierExplainer, ExplainerDashboard
from explainerdashboard.explainer_methods import IndexNotFoundError



def test_explainer_with_dataframe_y(fitted_rf_classifier_model, classifier_data):
    _, _, X_test, y_test = classifier_data
    explainer = ClassifierExplainer(
        fitted_rf_classifier_model, 
        X_test, 
        y_test.to_frame(), 
        cats=[{'Gender': ['Sex_female', 'Sex_male', 'Sex_nan']}, 'Deck', 'Embarked'],
        cats_notencoded={'Gender': 'No Gender'},
        labels=['Not survived', 'Survived']
    )
    _ = ExplainerDashboard(explainer)

def test_explainer_contains(precalculated_rf_classifier_explainer, test_names):
    assert 1 in precalculated_rf_classifier_explainer
    assert test_names[0] in precalculated_rf_classifier_explainer
    assert 1000 not in precalculated_rf_classifier_explainer
    assert "randomname" not in precalculated_rf_classifier_explainer

def test_explainer_len(precalculated_rf_classifier_explainer, testlen):
    assert len(precalculated_rf_classifier_explainer) == testlen

def test_int_idx(precalculated_rf_classifier_explainer, test_names):
    assert precalculated_rf_classifier_explainer.get_idx(test_names[0]) == 0

def test_getindex(precalculated_rf_classifier_explainer, test_names):
    assert precalculated_rf_classifier_explainer.get_index(0) == test_names[0]
    assert precalculated_rf_classifier_explainer.get_index(test_names[0]) == test_names[0]
    assert precalculated_rf_classifier_explainer.get_index(-1) is None
    assert precalculated_rf_classifier_explainer.get_index(10_000) is None
    assert precalculated_rf_classifier_explainer.get_index("Non existent index") is None

def test_get_idx(precalculated_rf_classifier_explainer, test_names):
    assert precalculated_rf_classifier_explainer.get_idx(test_names[0]) == 0
    assert precalculated_rf_classifier_explainer.get_idx(5) == 5
    with pytest.raises(IndexNotFoundError):
        precalculated_rf_classifier_explainer.get_idx(-1)
    with pytest.raises(IndexNotFoundError):
        precalculated_rf_classifier_explainer.get_idx(1000)
    with pytest.raises(IndexNotFoundError):
        precalculated_rf_classifier_explainer.get_idx("randomname")


def test_random_index(precalculated_rf_classifier_explainer):
    assert isinstance(precalculated_rf_classifier_explainer.random_index(), int)
    assert isinstance(precalculated_rf_classifier_explainer.random_index(return_str=True), str)

def test_index_exists(precalculated_rf_classifier_explainer):
    assert (precalculated_rf_classifier_explainer.index_exists(0))
    assert (precalculated_rf_classifier_explainer.index_exists(precalculated_rf_classifier_explainer.idxs[0]))
    assert (not precalculated_rf_classifier_explainer.index_exists('bla'))

def test_preds(precalculated_rf_classifier_explainer):
    assert isinstance(precalculated_rf_classifier_explainer.preds, np.ndarray)

def test_cats_notencoded(precalculated_rf_classifier_explainer):
    assert precalculated_rf_classifier_explainer.get_contrib_df(0).query("col=='Gender'")['value'].item() == 'No Gender'

def test_row_from_input(precalculated_rf_classifier_explainer):
    input_row = precalculated_rf_classifier_explainer.get_row_from_input(
        precalculated_rf_classifier_explainer.X.iloc[[0]].values.tolist())
    assert isinstance(input_row, pd.DataFrame)

    input_row = precalculated_rf_classifier_explainer.get_row_from_input(
        precalculated_rf_classifier_explainer.X_merged.iloc[[0]].values.tolist())
    assert isinstance(input_row, pd.DataFrame)

    input_row = precalculated_rf_classifier_explainer.get_row_from_input(
        precalculated_rf_classifier_explainer.X_merged
        [precalculated_rf_classifier_explainer.columns_ranked_by_shap()]
        .iloc[[0]].values.tolist(), ranked_by_shap=True)
    assert isinstance(input_row, pd.DataFrame)
    
def test_pred_percentiles(precalculated_rf_classifier_explainer):
    assert isinstance(precalculated_rf_classifier_explainer.pred_percentiles(), np.ndarray)

def test_columns_ranked_by_shap(precalculated_rf_classifier_explainer):
    assert isinstance(precalculated_rf_classifier_explainer.columns_ranked_by_shap(), list)

def test_get_col(precalculated_rf_classifier_explainer):
    assert isinstance(precalculated_rf_classifier_explainer.get_col("Gender"), pd.Series)
    assert (is_categorical_dtype(precalculated_rf_classifier_explainer.get_col("Gender")))

    assert isinstance(precalculated_rf_classifier_explainer.get_col("Deck"), pd.Series)
    assert (is_categorical_dtype(precalculated_rf_classifier_explainer.get_col("Deck")))

    assert isinstance(precalculated_rf_classifier_explainer.get_col("Age"), pd.Series)
    assert (is_numeric_dtype(precalculated_rf_classifier_explainer.get_col("Age")))

def test_permutation_importances(precalculated_rf_classifier_explainer):
    assert isinstance(precalculated_rf_classifier_explainer.permutation_importances(), pd.DataFrame)
    
def test_X_cats(precalculated_rf_classifier_explainer):
    assert isinstance(precalculated_rf_classifier_explainer.X_cats, pd.DataFrame)

def test_metrics(precalculated_rf_classifier_explainer):
    assert isinstance(precalculated_rf_classifier_explainer.metrics(), dict)

def test_mean_abs_shap_df(precalculated_rf_classifier_explainer):
    assert isinstance(precalculated_rf_classifier_explainer.mean_abs_shap_df(), pd.DataFrame)

def test_top_interactions(precalculated_rf_classifier_explainer):
    assert isinstance(precalculated_rf_classifier_explainer.top_shap_interactions("Age"), list)
    assert isinstance(precalculated_rf_classifier_explainer.top_shap_interactions("Age", topx=4), list)


def test_permutation_importances_df(precalculated_rf_classifier_explainer):
    assert isinstance(precalculated_rf_classifier_explainer.get_permutation_importances_df(), pd.DataFrame)
    assert isinstance(precalculated_rf_classifier_explainer.get_permutation_importances_df(topx=3), pd.DataFrame)
    assert isinstance(precalculated_rf_classifier_explainer.get_permutation_importances_df(cutoff=0.01), pd.DataFrame)

def test_contrib_df(precalculated_rf_classifier_explainer):
    assert isinstance(precalculated_rf_classifier_explainer.get_contrib_df(0), pd.DataFrame)
    assert isinstance(precalculated_rf_classifier_explainer.get_contrib_df(0, topx=3), pd.DataFrame)
    assert isinstance(precalculated_rf_classifier_explainer.get_contrib_df(0, sort='high-to-low'), pd.DataFrame)
    assert isinstance(precalculated_rf_classifier_explainer.get_contrib_df(0, sort='low-to-high'), pd.DataFrame)
    assert isinstance(precalculated_rf_classifier_explainer.get_contrib_df(0, sort='importance'), pd.DataFrame)
    assert isinstance(precalculated_rf_classifier_explainer.get_contrib_df(X_row=precalculated_rf_classifier_explainer.X.iloc[[0]]), pd.DataFrame)

def test_contrib_summary_df(precalculated_rf_classifier_explainer):
    assert isinstance(precalculated_rf_classifier_explainer.get_contrib_summary_df(0), pd.DataFrame)
    assert isinstance(precalculated_rf_classifier_explainer.get_contrib_summary_df(0, topx=3), pd.DataFrame)
    assert isinstance(precalculated_rf_classifier_explainer.get_contrib_summary_df(0, round=3), pd.DataFrame)
    assert isinstance(precalculated_rf_classifier_explainer.get_contrib_summary_df(0, sort='low-to-high'), pd.DataFrame)
    assert isinstance(precalculated_rf_classifier_explainer.get_contrib_summary_df(0, sort='high-to-low'), pd.DataFrame)
    assert isinstance(precalculated_rf_classifier_explainer.get_contrib_summary_df(0, sort='importance'), pd.DataFrame)
    assert isinstance(precalculated_rf_classifier_explainer.get_contrib_summary_df(X_row=precalculated_rf_classifier_explainer.X.iloc[[0]]), pd.DataFrame)

def test_shap_base_value(precalculated_rf_classifier_explainer):
    assert isinstance(precalculated_rf_classifier_explainer.shap_base_value(), (np.floating, float))

def test_shap_values_shape(precalculated_rf_classifier_explainer):
    assert (precalculated_rf_classifier_explainer.get_shap_values_df().shape == (len(precalculated_rf_classifier_explainer), len(precalculated_rf_classifier_explainer.merged_cols)))

def test_shap_values(precalculated_rf_classifier_explainer):
    assert isinstance(precalculated_rf_classifier_explainer.get_shap_values_df(), pd.DataFrame)

def test_shap_interaction_values(precalculated_rf_classifier_explainer):
    assert isinstance(precalculated_rf_classifier_explainer.shap_interaction_values(), np.ndarray)

def test_mean_abs_shap_df(precalculated_rf_classifier_explainer):
    assert isinstance(precalculated_rf_classifier_explainer.mean_abs_shap_df(), pd.DataFrame)

def test_calculate_properties(precalculated_rf_classifier_explainer):
    precalculated_rf_classifier_explainer.calculate_properties()

def test_shap_interaction_values_by_col(precalculated_rf_classifier_explainer):
    assert isinstance(precalculated_rf_classifier_explainer.shap_interaction_values_for_col("Age"), np.ndarray)
    assert (precalculated_rf_classifier_explainer.shap_interaction_values_for_col("Age").shape == 
                    precalculated_rf_classifier_explainer.get_shap_values_df().shape)

def test_prediction_result_df(precalculated_rf_classifier_explainer):
    df = precalculated_rf_classifier_explainer.prediction_result_df(0)
    assert isinstance(df, pd.DataFrame)

def test_pdp_df(precalculated_rf_classifier_explainer):
    assert isinstance(precalculated_rf_classifier_explainer.pdp_df("Age"), pd.DataFrame)
    assert isinstance(precalculated_rf_classifier_explainer.pdp_df("Gender"), pd.DataFrame)
    assert isinstance(precalculated_rf_classifier_explainer.pdp_df("Deck"), pd.DataFrame)
    assert isinstance(precalculated_rf_classifier_explainer.pdp_df("Age", index=0), pd.DataFrame)
    assert isinstance(precalculated_rf_classifier_explainer.pdp_df("Gender", index=0), pd.DataFrame)
    assert isinstance(precalculated_rf_classifier_explainer.pdp_df("Age", X_row=precalculated_rf_classifier_explainer.X.iloc[[0]]), pd.DataFrame)

def test_memory_usage(precalculated_rf_classifier_explainer):
    assert isinstance(precalculated_rf_classifier_explainer.memory_usage(), pd.DataFrame)
    assert isinstance(precalculated_rf_classifier_explainer.memory_usage(cutoff=1000), pd.DataFrame)

def test_plot_importances(precalculated_rf_classifier_explainer):
    fig = precalculated_rf_classifier_explainer.plot_importances()
    assert isinstance(fig, go.Figure)

    fig = precalculated_rf_classifier_explainer.plot_importances(kind='permutation')
    assert isinstance(fig, go.Figure)

    fig = precalculated_rf_classifier_explainer.plot_importances(topx=3)
    assert isinstance(fig, go.Figure)

def test_plot_interactions(precalculated_rf_classifier_explainer):
    fig = precalculated_rf_classifier_explainer.plot_interactions_importance("Age")
    assert isinstance(fig, go.Figure)

    fig = precalculated_rf_classifier_explainer.plot_interactions_importance("Gender")
    assert isinstance(fig, go.Figure)

def test_plot_contributions(precalculated_rf_classifier_explainer):
    fig = precalculated_rf_classifier_explainer.plot_contributions(0)
    assert isinstance(fig, go.Figure)

    fig = precalculated_rf_classifier_explainer.plot_contributions(0, topx=3)
    assert isinstance(fig, go.Figure)

    fig = precalculated_rf_classifier_explainer.plot_contributions(0, cutoff=0.05)
    assert isinstance(fig, go.Figure)

    fig = precalculated_rf_classifier_explainer.plot_contributions(0, sort='high-to-low')
    assert isinstance(fig, go.Figure)

    fig = precalculated_rf_classifier_explainer.plot_contributions(0, sort='low-to-high')
    assert isinstance(fig, go.Figure)

    fig = precalculated_rf_classifier_explainer.plot_contributions(0, sort='importance')
    assert isinstance(fig, go.Figure)

    fig = precalculated_rf_classifier_explainer.plot_contributions(X_row=precalculated_rf_classifier_explainer.X.iloc[[0]], sort='importance')
    assert isinstance(fig, go.Figure)

def test_plot_shap_detailed(precalculated_rf_classifier_explainer):
    fig = precalculated_rf_classifier_explainer.plot_importances_detailed()
    assert isinstance(fig, go.Figure)

    fig = precalculated_rf_classifier_explainer.plot_importances_detailed(topx=3)
    assert isinstance(fig, go.Figure)

def test_plot_interactions_detailed(precalculated_rf_classifier_explainer):
    fig = precalculated_rf_classifier_explainer.plot_interactions_detailed("Age")
    assert isinstance(fig, go.Figure)

    fig = precalculated_rf_classifier_explainer.plot_interactions_detailed("Age", topx=3)
    assert isinstance(fig, go.Figure)

    fig = precalculated_rf_classifier_explainer.plot_interactions_detailed("Age")
    assert isinstance(fig, go.Figure)

    fig = precalculated_rf_classifier_explainer.plot_interactions_detailed("Gender")
    assert isinstance(fig, go.Figure)

def test_plot_dependence(precalculated_rf_classifier_explainer):
    fig = precalculated_rf_classifier_explainer.plot_dependence("Age")
    assert isinstance(fig, go.Figure)

    fig = precalculated_rf_classifier_explainer.plot_dependence("Age", "Gender")
    assert isinstance(fig, go.Figure)

    fig = precalculated_rf_classifier_explainer.plot_dependence("Age", highlight_index=0)
    assert isinstance(fig, go.Figure)

    fig = precalculated_rf_classifier_explainer.plot_dependence("Gender", highlight_index=0)
    assert isinstance(fig, go.Figure)

def test_plot_interaction(precalculated_rf_classifier_explainer):

    fig = precalculated_rf_classifier_explainer.plot_interaction("Gender", "Age")
    assert isinstance(fig, go.Figure)

    fig = precalculated_rf_classifier_explainer.plot_interaction("Age", "Gender", highlight_index=0)
    assert isinstance(fig, go.Figure)

def test_plot_pdp(precalculated_rf_classifier_explainer):
    fig = precalculated_rf_classifier_explainer.plot_pdp("Age")
    assert isinstance(fig, go.Figure)

    fig = precalculated_rf_classifier_explainer.plot_pdp("Gender")
    assert isinstance(fig, go.Figure)

    fig = precalculated_rf_classifier_explainer.plot_pdp("Gender", index=0)
    assert isinstance(fig, go.Figure)

    fig = precalculated_rf_classifier_explainer.plot_pdp("Age", index=0)
    assert isinstance(fig, go.Figure)

    fig = precalculated_rf_classifier_explainer.plot_pdp("Age", X_row=precalculated_rf_classifier_explainer.X.iloc[[0]])
    assert isinstance(fig, go.Figure)

def test_yaml(precalculated_rf_classifier_explainer):
    yaml = precalculated_rf_classifier_explainer.to_yaml()
    assert isinstance(yaml, str)

def test_yaml_return_dict(precalculated_rf_classifier_explainer):
    return_dict = precalculated_rf_classifier_explainer.to_yaml(return_dict=True)
    assert isinstance(return_dict, dict)




