InlineExplainer
***************

As datascientists you often work inside a notebook environment where you 
quickly interactively like to explore your data. The ``InlineExplainer`` allows
you to do this by running components (or whole tabs) inline inside your Jupyter
notebook.

This allows you to quickly check model performance, look for shap importances,
etc. The components are sorted into subcategories and work with tab-completion.


Example use::

    InlineExplainer(explainer).importances()
    InlineExplainer(explainer).model_stats()
    InlineExplainer(explainer).prediction()
    InlineExplainer(explainer).random_index()
    InlineExplainer(explainer).tab.importances()
    InlineExplainer(explainer).tab.modelsummary()
    InlineExplainer(explainer).tab.contributions()
    InlineExplainer(explainer).tab.dependence()
    InlineExplainer(explainer).tab.interactions()
    InlineExplainer(explainer).tab.decisiontrees()
    InlineExplainer(explainer).shap.overview()
    InlineExplainer(explainer).shap.summary()
    InlineExplainer(explainer).shap.dependence()
    InlineExplainer(explainer).shap.interaction_overview()
    InlineExplainer(explainer).shap.interaction_summary()
    InlineExplainer(explainer).shap.interaction_dependence()
    InlineExplainer(explainer).shap.contributions_graph()
    InlineExplainer(explainer).shap.contributions_table()
    InlineExplainer(explainer).classifier.model_stats()
    InlineExplainer(explainer).classifier.precision()
    InlineExplainer(explainer).classifier.confusion_matrix()
    InlineExplainer(explainer).classifier.lift_curve()
    InlineExplainer(explainer).classifier.classification()
    InlineExplainer(explainer).classifier.roc_auc()
    InlineExplainer(explainer).classifier.pr_auc()
    InlineExplainer(explainer).regression.model_stats()
    InlineExplainer(explainer).regression.pred_vs_actual()
    InlineExplainer(explainer).regression.residuals()
    InlineExplainer(explainer).regression.residuals_vs_col()
    InlineExplainer(explainer).decisiontrees.overview()
    InlineExplainer(explainer).decisiontrees.decision_trees()
    InlineExplainer(explainer).decisiontrees.decisionpath_table()
    InlineExplainer(explainer).decisiontrees.decisionpath_graph()

You can also add options for the size of the output version, or to display 
the component in a separate tab::

    InlineExplainer(explainer, mode='external', width=1000, height=800).importances()



.. autoclass:: explainerdashboard.dashboards.InlineExplainer
   :members:  tab, shap, classifier, regression, decisiontrees, importances, model_stats, prediction, random_index, pdp
   :member-order: bysource

.. autoclass:: explainerdashboard.dashboards.InlineExplainerTabs
   :members: 

.. autoclass:: explainerdashboard.dashboards.InlineShapExplainer
   :members:  

.. autoclass:: explainerdashboard.dashboards.InlineClassifierExplainer
   :members: 

.. autoclass:: explainerdashboard.dashboards.InlineRegressionExplainer
   :members: 

.. autoclass:: explainerdashboard.dashboards.InlineDecisionTreesExplainer
   :members: 

