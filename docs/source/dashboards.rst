ExplainerDashboard
******************

In order to start an ``ExplainerDashboard`` you first need to contruct an ``Explainer`` instance.
On the basis of this explainer you can then quickly start an interactive dashboard. (note: if you 
are  working inside a jupyter notebook, then it's probably better to use 
``JupyterExplainerDashboard`` instead. See below)

You can pass a list of ``ExplainerComponents`` to the ``tabs`` parameter, or
alternatively switch off individual tabs to display using booleans. 
By default all tabs that are feasible will be displayed. Warning: the interactions tab can
take quite some time to compute, so you may want to switch it off if you're not particularly 
interested in interaction effects between features.

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

JupyterExplainerDashboard
=========================
If you are working within a notebook there is also an alternative ``JupyterExplainerDashboard`` that uses the ``JupyterDash`` 
backend instead of ``dash.Dash()``. You can either pass to ``run()`` ``mode='inline'`` or ``mode='jupyterlab'`` or 
``mode='external'`` to start the dashboard inline in a notebook, in seperate pane 
in jupyterlab, or in seperate browser tab respectively. This has the benefit
that you can keep working inside the notebook while the dashboard is running.

.. autoclass:: explainerdashboard.dashboards.JupyterExplainerDashboard
   :members: __init__, run

ExplainerTab
============

To run a single page of a particular ``ExplainerComponent`` or tab, there is ``ExplainerTab``. 
You can either pass the appropriate tab class definition, or a string identifier::

    ExplainerTab(explainer, "model_summary").run()
    ExplainerTab(explainer, "shap_dependence").run(port=8051)
    ExplainerTab(explainer, ShapInteractionsTab).run()


.. autoclass:: explainerdashboard.dashboards.ExplainerTab
   :members: __init__, run

JupyterExplainerTab
===================

Equivalent to JupyterExplainerDashboard, runs a single tab using the
JupyterDash() instead of dash.Dash(). You can either pass to ``run()`` 
``mode='inline'`` or ``mode='jupyterlab'`` or ``mode='external'`` to start 
the dashboard inline in a notebook, in seperate pane in jupyterlab, or in 
seperate browser tab respectively. 

.. autoclass:: explainerdashboard.dashboards.JupyterExplainerTab
   :members: __init__, run

InlineExplainerTab
==================

An alternative API to run a particular tab or component inline in a notebook. each
individual component can be accessed through InlineExplainer directly. Full tabs
can be found under the subclass ``tab``, shap related components under ``shap``, etc.

Examples::

    InlineExplainer(explainer).model_stats()
    InlineExplainer(explainer).shap.dependence()
    InlineExplainer(explainer, mode='external').tab.contributions()
    InlineExplainer(explainer).classifier.confusion_matrix()
    InlineExplainer(explainer).regression.residuals()
    InlineExplainer(explainer, width=1200, height=1000).shap_interaction()


.. autoclass:: explainerdashboard.dashboards.InlineExplainerTab
   :members: __init__, model_summary, importances, model_stats, contributions, shap_dependence, shap_interaction, decision_trees



Dashboard tabs
==============

Importances tab (importances=True)
----------------------------------

Shows an overview of the most important features according to either permutation importance
or mean absolute shap value.


Model summary tab (model_summary=True)
--------------------------------------

Shows a summary of the model performance.

For classifiers, shows: precision plot, confusion matrix, ROC AUC en PR AUC curves 
and permutation importances and mean absolute SHAP values per feature. 

For regression models for now only shows permutation importances and mean absolute SHAP values per feature.

.. autoclass:: explainerdashboard.dashboard_tabs.ModelSummaryTab

Individual Predictions Tab (contributions=True)
-----------------------------------------------

Explains individual predictions, showing the shap values for each feature that
impacted the prediction. Also shows a pdp plot for each feature.

.. autoclass:: explainerdashboard.dashboard_tabs.ContributionsTab

Feature Dependence tab (shap_dependence=True)
---------------------------------------------

Shows a summary of the distributions of shap values for each feature. When clicked
shows the shap value plotted versus the feature value. 

.. autoclass:: explainerdashboard.dashboard_tabs.ShapDependenceTab

Feature Interactions tab (shap_interaction=True)
------------------------------------------------

Shows a summary of the distributions of shap interaction values for each a given feature. 
When clicked shows the shap interactions value plotted versus the feature value. 

.. autoclass:: explainerdashboard.dashboard_tabs.ShapInteractionsTab

Decision Trees tab (decision_trees=True)
----------------------------------------

Shows the distributions of predictions of individual decision trees inside your
random forest.

.. autoclass:: explainerdashboard.dashboard_tabs.DecisionTreesTab