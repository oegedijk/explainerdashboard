InlineExplainer
***************

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


.. autoclass:: explainerdashboard.dashboards.InlineExplainer
   :members:  tab, shap, classifier, regression, decisiontrees, importances, model_stats, prediction, random_index, pdp
   :member-order: bysource