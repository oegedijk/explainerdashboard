Plots
*****

The BaseExplainer class provides a number of useful analytical plots:

- plot_importances()
- plot_shap_contributions()
- plot_shap_summary()
- plot_shap_interaction_summary()
- plot_shap_dependence()
- plot_shap_interaction_dependence()
- plot_pdp()

For the derived ClassifierExplainer class some additional plots are available:

- plot_precision()
- plot_cumulative_precision()
- plot_classification()
- plot_confusion_matrix()
- plot_lift_curve()
- plot_roc_auc()
- plot_pr_auc()

For the derived RegressionExplainer class again some additional plots:

-  plot_predicted_vs_actual()
-  plot_residuals()
-  plot_residuals_vs_feature()

Finally RandomForestExplainer provides:

- plot_trees()
- decision_path(), decision_path_file(), decision_path_encoded()

.. _base_plots:
BaseExplainer: Plots
====================
.. autoclass:: explainerdashboard.explainers.BaseExplainer
   :members: plot_importances, plot_shap_contributions, plot_shap_summary, plot_shap_interaction_summary, plot_shap_dependence, plot_shap_interaction_dependence, plot_pdp
   :member-order: bysource
   :exclude-members: __init__
   :noindex: 

.. _classifier_plots:
ClassifierExplainer: Plots
==========================

.. autoclass:: explainerdashboard.explainers.ClassifierExplainer
   :members: plot_precision, plot_cumulative_precision, plot_classification, plot_lift_curve, plot_confusion_matrix, plot_roc_auc, plot_pr_auc
   :member-order: bysource
   :exclude-members: __init__
   :noindex: 

.. _regression_plots:
RegressionExplainer: Plots
==========================

.. autoclass:: explainerdashboard.explainers.RegressionExplainer
   :members: plot_predicted_vs_actual, plot_residuals,  plot_residuals_vs_feature
   :member-order: bysource
   :exclude-members: __init__
   :noindex:
   
.. _randomforest_plots:
RandomForestExplainerExplainer: Plots
=====================================

.. autoclass:: explainerdashboard.explainers.RandomForestExplainer
   :members: plot_trees, decision_path
   :member-order: bysource
   :exclude-members: __init__
   :noindex: 