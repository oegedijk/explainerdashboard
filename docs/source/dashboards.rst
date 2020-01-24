ExplainerDashboard
******************

In order to start an **ExplainerDashboard** you first need to contruct an *ExplainerBunch*.
On the basis of this explainer you can then quickly start an interactive dashboard.

You indicate which tabs to display using booleans in the constructor.

The individual tabs also take arguments. If you pass these as kwargs in the
constructor these will be passed down to the individual tabs.


ExplainerDashboard
==================

.. autoclass:: explainerdashboard.dashboards.ExplainerDashboard
   :members: __init__, run



Dashboard tabs
==============

Individual Contributions Tab (contributions=True)
-------------------------------------------------

Explains individual predictions, showing the shap values for each feature that
impacted the prediction. Also shows a pdp plot for each feature.

.. autoclass:: explainerdashboard.dashboard_tabs.contributions_tab.ContributionsTab

Model summary tab (model_summary=True)
--------------------------------------

Shows a summary of the model performance.

For classifiers, shows: precision plot, confusion matrix, ROC AUC en PR AUC curves 
and permutation importances and mean absolute SHAP values per feature. 

For regression models for now only shows permutation importances and mean absolute SHAP values per feature.

.. autoclass:: explainerdashboard.dashboard_tabs.model_summary_tab.ModelSummaryTab

Dependence tab (shap_dependence=True)
-------------------------------------

Shows a summary of the distributions of shap values for each feature. When clicked
shows the shap value plotted versus the feature value. 

.. autoclass:: explainerdashboard.dashboard_tabs.shap_dependence_tab.ShapDependenceTab

Interactions tab (shap_interaction=True)
----------------------------------------

Shows a summary of the distributions of shap interaction values for each a given feature. 
When clicked shows the shap interactions value plotted versus the feature value. 

.. autoclass:: explainerdashboard.dashboard_tabs.shap_interactions_tab.ShapInteractionsTab

Shadow Trees tab (shadow_trees=True)
------------------------------------

Shows the distributions of predictions of individual decision trees inside your
random forest.

.. autoclass:: explainerdashboard.dashboard_tabs.shadow_trees_tab.ShadowTreesTab