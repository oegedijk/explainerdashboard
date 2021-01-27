InlineExplainer
***************

As a data scientist you often work inside a notebook environment where you 
quickly interactively like to explore your data. The ``InlineExplainer`` allows
you to do this by running ``ExplainerComponents`` (or whole tabs) inline 
inside your Jupyter notebook (also works in Google Colab!).

.. image:: screenshots/inline_screenshot.*

This allows you to quickly check model performance, look for shap importances,
etc. The components are sorted into subcategories and work with tab-completion.


Example use::

    from explainerdashboard import InlineExplainer
    ie = InlineExplainer(explainer)
    ie.importances()
    ie.model_stats()
    ie.prediction()
    ie.random_index()
    ie.tab.importances()
    ie.tab.modelsummary()
    ie.tab.contributions()
    ie.tab.dependence()
    ie.tab.interactions()
    ie.tab.decisiontrees()
    ie.shap.overview()
    ie.shap.summary()
    ie.shap.dependence()
    ie.shap.interaction_overview()
    ie.shap.interaction_summary()
    ie.shap.interaction_dependence()
    ie.shap.contributions_graph()
    ie.shap.contributions_table()
    ie.classifier.model_stats()
    ie.classifier.precision()
    ie.classifier.confusion_matrix()
    ie.classifier.lift_curve()
    ie.classifier.classification()
    ie.classifier.roc_auc()
    ie.classifier.pr_auc()
    ie.regression.model_stats()
    ie.regression.pred_vs_actual()
    ie.regression.residuals()
    ie.regression.plots_vs_col()
    ie.decisiontrees.overview()
    ie.decisiontrees.decision_trees()
    ie.decisiontrees.decisionpath_table()
    ie.decisiontrees.decisionpath_graph()

You can also add options for the size of the output width, or to display 
the component in a separate tab ('external'), or running on a different port::

    InlineExplainer(explainer, mode='external', port=8051, width=1000, height=800).importances()

.. note::
   You can run a component without instantiating the InlineExplainer first,
   like for example ``InlineExplainer(explainer).importances()``, but then you
   cannot inspect the kwargs and docstring of that particular component. 
   So to inspect kwargs and docstring you would run::

      ie = InlineExplainer(explainer)
      ?ie.importances


(or alternatively hit shift-tab in jupyter of course)

You can kill an ``InlineExplainer`` running on a particular port with 
``ExplainerDashboard.terminate(port=8050)``.

InlineExplainer documentation
=============================


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

