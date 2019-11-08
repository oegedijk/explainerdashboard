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

RandomForestDashboard
=====================

.. autoclass:: explainerdashboard.dashboards.RandomForestDashboard
   :members: __init__

Dashboard tabs
==============

Individual Contributions Tab (contributions=True)
-------------------------------------------------

Explains individual predictions, showing the shap values for each feature that
impacted the prediction. Also shows a pdp plot for each feature.

.. autofunction:: explainerdashboard.dashboard_tabs.contributions_tab.contributions_tab

Model summary tab (model_summary=True)
--------------------------------------

Shows a summary of the model performance.

For classifiers, shows: precision plot, confusion matrix, ROC AUC en PR AUC curves 
and permutation importances and mean absolute SHAP values per feature. 

For regression models for now only shows permutation importances and mean absolute SHAP values per feature.

.. autofunction:: explainerdashboard.dashboard_tabs.contributions_tab.contributions_tab

Dependence tab (shap_dependence=True)
------------------------------------

Shows a summary of the distributions of shap values for each feature. When clicked
shows the shap value plotted versus the feature value. 

.. autofunction:: explainerdashboard.dashboard_tabs.shap_dependence_tab.shap_dependence_tab

Interactions tab (shap_interaction=True)
------------------------------------

Shows a summary of the distributions of shap interaction values for each a given feature. 
When clicked shows the shap interactions value plotted versus the feature value. 

.. autofunction:: explainerdashboard.dashboard_tabs.shap_interactions_tab.shap_interactions_tab
