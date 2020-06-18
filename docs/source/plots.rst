Plots
*****

The BaseExplainer class provides a number of useful analytical plots::

    explainer.plot_importances(...)
    explainer.plot_shap_contributions(...)
    explainer.plot_shap_summary(...)
    explainer.plot_shap_interaction_summary(...)
    explainer.plot_shap_dependence(...)
    explainer.plot_shap_interaction_dependence(...)
    explainer.plot_pdp(...)

For the derived ClassifierExplainer class some additional plots are available::

    explainer.plot_precision(...)
    explainer.plot_cumulative_precision(...)
    explainer.plot_classification(...)
    explainer.plot_confusion_matrix(...)
    explainer.plot_lift_curve(...)
    explainer.plot_roc_auc(...)
    explainer.plot_pr_auc(...)

For the derived RegressionExplainer class again some additional plots::

    explainer.plot_predicted_vs_actual(...)
    explainer.plot_residuals(...)
    explainer.plot_residuals_vs_feature(...)

Finally RandomForestExplainer provides::

    explainer.plot_trees(...)
    explainer.decision_path(...)


BaseExplainer: Plots
====================
.. autoclass:: explainerdashboard.explainers.BaseExplainer
   :members: plot_importances, plot_shap_contributions, plot_shap_summary, plot_shap_interaction_summary, plot_shap_dependence, plot_shap_interaction, plot_pdp
   :member-order: bysource
   :exclude-members: __init__
   :noindex: 


ClassifierExplainer: Plots
==========================

.. autoclass:: explainerdashboard.explainers.ClassifierExplainer
   :members: plot_precision, plot_cumulative_precision, plot_classification, plot_lift_curve, plot_confusion_matrix, plot_roc_auc, plot_pr_auc
   :member-order: bysource
   :exclude-members: __init__
   :noindex: 


RegressionExplainer: Plots
==========================

.. autoclass:: explainerdashboard.explainers.RegressionExplainer
   :members: plot_predicted_vs_actual, plot_residuals,  plot_residuals_vs_feature
   :member-order: bysource
   :exclude-members: __init__
   :noindex:
   

RandomForestExplainer: Plots
=====================================

.. autoclass:: explainerdashboard.explainers.RandomForestExplainer
   :members: plot_trees, decision_path
   :member-order: bysource
   :exclude-members: __init__
   :noindex: 