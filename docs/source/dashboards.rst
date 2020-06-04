ExplainerDashboard
******************

In order to start an ``ExplainerDashboard`` you first need to contruct an ``Explainer`` instance.
On the basis of this explainer you can then quickly start an interactive dashboard.

You can switch off individual tabs to display using booleans in the constructor. 
By default all tabs that are feasible will be displayed. Warning: the interactions tab can
take quite some time to compute, so you may want to switch it off if you're not particularly 
interested in interaction effects between features.

The individual tabs also take arguments. If you pass these as kwargs in the
constructor these will be passed down to the individual tabs.

Some example code::

    from explainerdashboard.dashboards import ExplainerDashboard

    ExplainerDashboard(explainer, title="Titanic Explainer",
                        model_summary=True,  
                        contributions=True,
                        shap_dependence=False,
                        shap_interaction=False,
                        decision_trees=False).run(port=8051)


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

Decision Trees tab (decision_trees=True)
----------------------------------------

Shows the distributions of predictions of individual decision trees inside your
random forest.

.. autoclass:: explainerdashboard.dashboard_tabs.decision_trees_tab.DecisionTreesTab