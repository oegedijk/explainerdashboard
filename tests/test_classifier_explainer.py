import pandas as pd
import numpy as np

import plotly.graph_objects as go

def test_pos_label(precalculated_rf_classifier_explainer):
    precalculated_rf_classifier_explainer.pos_label = 1
    precalculated_rf_classifier_explainer.pos_label = "Not survived"
    assert isinstance(precalculated_rf_classifier_explainer.pos_label, int)
    assert isinstance(precalculated_rf_classifier_explainer.pos_label_str, str)
    assert (precalculated_rf_classifier_explainer.pos_label == 0)
    assert (precalculated_rf_classifier_explainer.pos_label_str == "Not survived")

def test_custom_metrics(precalculated_rf_classifier_explainer):
    def meandiff_metric1(y_true, y_pred):
        return np.mean(y_true)-np.mean(y_pred)

    def meandiff_metric2(y_true, y_pred, cutoff):
        return np.mean(y_true)-np.mean(np.where(y_pred>cutoff, 1, 0))

    def meandiff_metric3(y_true, y_pred, pos_label):
        return np.mean(np.where(y_true==pos_label, 1, 0))-np.mean(y_pred[:, pos_label])

    def meandiff_metric4(y_true, y_pred, cutoff, pos_label):
        return np.mean(np.where(y_true==pos_label, 1, 0))-np.mean(np.where(y_pred[:, pos_label] > cutoff, 1, 0))

    metrics = np.array(list(precalculated_rf_classifier_explainer.metrics(
        show_metrics=[meandiff_metric1, meandiff_metric2, meandiff_metric3, meandiff_metric4]
        ).values()))
    assert (np.all(metrics==metrics[0]))


def test_pred_probas(precalculated_rf_classifier_explainer):
    assert isinstance(precalculated_rf_classifier_explainer.pred_probas(), np.ndarray)
    assert isinstance(precalculated_rf_classifier_explainer.pred_probas(1), np.ndarray)
    assert isinstance(precalculated_rf_classifier_explainer.pred_probas("Survived"), np.ndarray)

def test_metrics(precalculated_rf_classifier_explainer):
    assert isinstance(precalculated_rf_classifier_explainer.metrics(), dict)
    assert isinstance(precalculated_rf_classifier_explainer.metrics(cutoff=0.9), dict)
    assert isinstance(precalculated_rf_classifier_explainer.metrics_descriptions(cutoff=0.9), dict)

def test_precision_df(precalculated_rf_classifier_explainer):
    assert isinstance(precalculated_rf_classifier_explainer.get_precision_df(), pd.DataFrame)
    assert isinstance(precalculated_rf_classifier_explainer.get_precision_df(multiclass=True), pd.DataFrame)
    assert isinstance(precalculated_rf_classifier_explainer.get_precision_df(quantiles=4), pd.DataFrame)

def test_lift_curve_df(precalculated_rf_classifier_explainer):
    assert isinstance(precalculated_rf_classifier_explainer.get_liftcurve_df(), pd.DataFrame)

def test_calculate_properties(precalculated_rf_classifier_explainer):
    precalculated_rf_classifier_explainer.calculate_properties()
    
def test_plot_precision(precalculated_rf_classifier_explainer):
    fig = precalculated_rf_classifier_explainer.plot_precision()
    assert isinstance(fig, go.Figure)

    fig = precalculated_rf_classifier_explainer.plot_precision(multiclass=True)
    assert isinstance(fig, go.Figure)

    fig = precalculated_rf_classifier_explainer.plot_precision(quantiles=10, cutoff=0.5)
    assert isinstance(fig, go.Figure)

def test_plot_cumulutive_precision(precalculated_rf_classifier_explainer):
    fig = precalculated_rf_classifier_explainer.plot_cumulative_precision()
    assert isinstance(fig, go.Figure)

    fig = precalculated_rf_classifier_explainer.plot_cumulative_precision(percentile=0.5)
    assert isinstance(fig, go.Figure)

    fig = precalculated_rf_classifier_explainer.plot_cumulative_precision(percentile=0.1, pos_label=0)
    assert isinstance(fig, go.Figure)

def test_plot_confusion_matrix(precalculated_rf_classifier_explainer):
    fig = precalculated_rf_classifier_explainer.plot_confusion_matrix(percentage=False, binary=False)
    assert isinstance(fig, go.Figure)

    fig = precalculated_rf_classifier_explainer.plot_confusion_matrix(percentage=False, binary=True)
    assert isinstance(fig, go.Figure)

    fig = precalculated_rf_classifier_explainer.plot_confusion_matrix(percentage=True, binary=False)
    assert isinstance(fig, go.Figure)

    fig = precalculated_rf_classifier_explainer.plot_confusion_matrix(percentage=True, binary=True)
    assert isinstance(fig, go.Figure)

    fig = precalculated_rf_classifier_explainer.plot_confusion_matrix(normalize='all')
    assert isinstance(fig, go.Figure)

    fig = precalculated_rf_classifier_explainer.plot_confusion_matrix(normalize='observed')
    assert isinstance(fig, go.Figure)

    fig = precalculated_rf_classifier_explainer.plot_confusion_matrix(normalize='pred')
    assert isinstance(fig, go.Figure)

def test_plot_lift_curve(precalculated_rf_classifier_explainer):
    fig = precalculated_rf_classifier_explainer.plot_lift_curve()
    assert isinstance(fig, go.Figure)

    fig = precalculated_rf_classifier_explainer.plot_lift_curve(percentage=True)
    assert isinstance(fig, go.Figure)

    fig = precalculated_rf_classifier_explainer.plot_lift_curve(cutoff=0.5)
    assert isinstance(fig, go.Figure)

    fig = precalculated_rf_classifier_explainer.plot_lift_curve(add_wizard=False, round=3)
    assert isinstance(fig, go.Figure)

def test_plot_lift_curve(precalculated_rf_classifier_explainer):
    fig = precalculated_rf_classifier_explainer.plot_lift_curve()
    assert isinstance(fig, go.Figure)

    fig = precalculated_rf_classifier_explainer.plot_lift_curve(percentage=True)
    assert isinstance(fig, go.Figure)

    fig = precalculated_rf_classifier_explainer.plot_lift_curve(cutoff=0.5)
    assert isinstance(fig, go.Figure)

def test_plot_classification(precalculated_rf_classifier_explainer):
    fig = precalculated_rf_classifier_explainer.plot_classification()
    assert isinstance(fig, go.Figure)

    fig = precalculated_rf_classifier_explainer.plot_classification(percentage=True)
    assert isinstance(fig, go.Figure)

    fig = precalculated_rf_classifier_explainer.plot_classification(cutoff=0)
    assert isinstance(fig, go.Figure)

    fig = precalculated_rf_classifier_explainer.plot_classification(cutoff=1)
    assert isinstance(fig, go.Figure)

def test_plot_roc_auc(precalculated_rf_classifier_explainer):
    fig = precalculated_rf_classifier_explainer.plot_roc_auc(0.5)
    assert isinstance(fig, go.Figure)

    fig = precalculated_rf_classifier_explainer.plot_roc_auc(0.0)
    assert isinstance(fig, go.Figure)

    fig = precalculated_rf_classifier_explainer.plot_roc_auc(1.0)
    assert isinstance(fig, go.Figure)

def test_plot_pr_auc(precalculated_rf_classifier_explainer):
    fig = precalculated_rf_classifier_explainer.plot_pr_auc(0.5)
    assert isinstance(fig, go.Figure)

    fig = precalculated_rf_classifier_explainer.plot_pr_auc(0.0)
    assert isinstance(fig, go.Figure)

    fig = precalculated_rf_classifier_explainer.plot_pr_auc(1.0)
    assert isinstance(fig, go.Figure)

def test_plot_prediction_result(precalculated_rf_classifier_explainer):
    fig = precalculated_rf_classifier_explainer.plot_prediction_result(0)
    assert isinstance(fig, go.Figure)

