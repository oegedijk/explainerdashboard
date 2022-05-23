import pytest

import pandas as pd
import numpy as np


def test_pipeline_columns_ranked_by_shap(classifier_pipeline_explainer):
    assert isinstance(classifier_pipeline_explainer.columns_ranked_by_shap(), list)

def test_pipeline_permutation_importances(classifier_pipeline_explainer):
    assert isinstance(classifier_pipeline_explainer.get_permutation_importances_df(), pd.DataFrame)

def test_pipeline_metrics(classifier_pipeline_explainer):
    assert isinstance(classifier_pipeline_explainer.metrics(), dict)
    assert isinstance(classifier_pipeline_explainer.metrics_descriptions(), dict)

def test_pipeline_mean_abs_shap_df(classifier_pipeline_explainer):
    assert isinstance(classifier_pipeline_explainer.get_mean_abs_shap_df(), pd.DataFrame)

def test_pipeline_contrib_df(classifier_pipeline_explainer):
    assert isinstance(classifier_pipeline_explainer.get_contrib_df(0), pd.DataFrame)
    assert isinstance(classifier_pipeline_explainer.get_contrib_df(X_row=classifier_pipeline_explainer.X.iloc[[0]]), pd.DataFrame)

def test_pipeline_shap_base_value(classifier_pipeline_explainer):
    assert isinstance(classifier_pipeline_explainer.shap_base_value(), (np.floating, float))

def test_pipeline_shap_values_shape(classifier_pipeline_explainer):
    assert (classifier_pipeline_explainer.get_shap_values_df().shape == (len(classifier_pipeline_explainer), len(classifier_pipeline_explainer.merged_cols)))

def test_pipeline_shap_values(classifier_pipeline_explainer):
    assert isinstance(classifier_pipeline_explainer.get_shap_values_df(), pd.DataFrame)

def test_pipeline_pdp_df(classifier_pipeline_explainer):
    assert isinstance(classifier_pipeline_explainer.pdp_df("num__age"), pd.DataFrame)
    assert isinstance(classifier_pipeline_explainer.pdp_df("cat__sex"), pd.DataFrame)
    assert isinstance(classifier_pipeline_explainer.pdp_df("num__age", index=0), pd.DataFrame)
    assert isinstance(classifier_pipeline_explainer.pdp_df("cat__sex", index=0), pd.DataFrame)

def test_pipeline_kernel_columns_ranked_by_shap(classifier_pipeline_kernel_explainer):
    assert isinstance(classifier_pipeline_kernel_explainer.columns_ranked_by_shap(), list)

def test_pipeline_kernel_permutation_importances(classifier_pipeline_kernel_explainer):
    assert isinstance(classifier_pipeline_kernel_explainer.get_permutation_importances_df(), pd.DataFrame)

def test_pipeline_kernel_metrics(classifier_pipeline_kernel_explainer):
    assert isinstance(classifier_pipeline_kernel_explainer.metrics(), dict)
    assert isinstance(classifier_pipeline_kernel_explainer.metrics_descriptions(), dict)

def test_pipeline_kernel_mean_abs_shap_df(classifier_pipeline_kernel_explainer):
    assert isinstance(classifier_pipeline_kernel_explainer.get_mean_abs_shap_df(), pd.DataFrame)

def test_pipeline_kernel_contrib_df(classifier_pipeline_kernel_explainer):
    assert isinstance(classifier_pipeline_kernel_explainer.get_contrib_df(0), pd.DataFrame)
    assert isinstance(classifier_pipeline_kernel_explainer.get_contrib_df(X_row=classifier_pipeline_kernel_explainer.X.iloc[[0]]), pd.DataFrame)

def test_pipeline_kernel_shap_base_value(classifier_pipeline_kernel_explainer):
    assert isinstance(classifier_pipeline_kernel_explainer.shap_base_value(), (np.floating, float))

def test_pipeline_kernel_shap_values_shape(classifier_pipeline_kernel_explainer):
    assert (classifier_pipeline_kernel_explainer.get_shap_values_df().shape == (len(classifier_pipeline_kernel_explainer), len(classifier_pipeline_kernel_explainer.merged_cols)))

def test_pipeline_kernel_shap_values(classifier_pipeline_kernel_explainer):
    assert isinstance(classifier_pipeline_kernel_explainer.get_shap_values_df(), pd.DataFrame)

def test_pipeline_kernel_pdp_df(classifier_pipeline_kernel_explainer):
    assert isinstance(classifier_pipeline_kernel_explainer.pdp_df("age"), pd.DataFrame)
    assert isinstance(classifier_pipeline_kernel_explainer.pdp_df("sex"), pd.DataFrame)
    assert isinstance(classifier_pipeline_kernel_explainer.pdp_df("age", index=0), pd.DataFrame)
    assert isinstance(classifier_pipeline_kernel_explainer.pdp_df("sex", index=0), pd.DataFrame)

