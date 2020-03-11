ExplainerBunch
**************

In order to start an **explainerdashboard** you first need to construct an **ExplainerBunch**.


ExplainersBunches
=================
The abstract base class **BaseExplainerBunch** defines most of the functionality 
such as feature importances (both SHAP and permutation based), SHAP values, SHAP interaction values
partial dependences, individual contributions, etc. Along with a number of convenient
plotting methods 

The BaseExplainerBunch already provides a number of convenient plotting methods:

- plot_importances()
- plot_shap_contributions()
- plot_shap_summary()
- plot_shap_interaction_summary()
- plot_shap_dependence()
- plot_shap_interaction_dependence()
- plot_pdp()

example code::

    explainer = BaseExplainerBunch(model, X, y)
    explainer.plot_importances()
    explainer.plot_shap_contributions(index=0)


ClassifierBunch
===============

For classification models where we try to predict the probability of each class
we have a derived class called ClassifierBunch.

You can pass an additional parameter to __init__ with a list label names. For
multilabel classifier you can set the positive class with e.g. *explainer.pos_label=1*.
This will make sure that for example *explainer.pred_probas* will return the probability
of that label.You an also pass a string label if you pased *labels* to the constructor.

ClassifierBunch defines a number of additional plotting methods:

- plot_precision()
- plot_cumulative_precision()
- plot_classification()
- plot_confusion_matrix()
- plot_lift_curve()
- plot_roc_auc()
- plot_pr_auc()



example code::
    explainer = ClassifierBunch(model, X, y, labels=['Not Survived', 'Survived'])
    explainer.plot_confusion_matrix(cutoff=0.6)
    explainer.plot_roc_auc()

RegressionBunch
===============

For regression models where we try to predict a certain quantity.

You can pass an additional parameter to __init__ with the units of the predicted quantity.

RegressionBunch defines a number of additional plotting methods:

-  plot_predicted_vs_actual(self, round=2, logs=False):
-  plot_residuals(self, vs_actual=False, round=2, ratio=False):
-  plot_residuals_vs_feature(self, col)

example code::

    explainer = RegressionBunch(model, X, y, units='dollars')
    explainer.plot_predicted_vs_actual()
    explainer.plot_residuals()


RandomForestExplainerBunch
==========================

There is an additional mixin class specifically for scikitlearn type RandomForests
that defines additional methods and plots to investigate and visualize 
individual decision trees within the random forest.

- **RandomForestExplainerBunch**: uses shap.TreeExplainer plus generates so-called shadow trees

You can get a pd.DataFrame summary of the path that a specific index took through a specific tree.
You can also plot the individual prediction of each individual tree. 

- shadowtree_df(tree_idx, index)
- plot_trees(index)

And for dtreeviz visualization of individual decision trees (svg format):
- decision_path()
- decision_path_file()
- decision_path_encoded()

This also works with classifiers and regression models:

**RandomForestClassifierBunch(model, X, y)**
**RandomForestRegressionBunch(model, X, y)**

BaseExplainerBunch
==================
.. autoclass:: explainerdashboard.explainers.BaseExplainerBunch
   :members: mean_abs_shap_df, permutation_importances_df, importances_df, contrib_df, to_sql 
   :member-order: bysource

For the plotting methods see base_plots_

ClassifierBunch
===============

.. autoclass:: explainerdashboard.explainers.ClassifierBunch
   :members: random_index, precision_df
   :member-order: bysource

For the plotting methods see classifier_plots_

RegressionBunch
===============

.. autoclass:: explainerdashboard.explainers.RegressionBunch
   :members: random_index, residuals, metrics, prediction_result_markdown
   :member-order: bysource

For the plotting methods see regression_plots_


RandomForestExplainerBunch
==========================

.. autoclass:: explainerdashboard.explainers.RandomForestExplainerBunch
   :members: decisiontree_df, decisiontree_df_summary, plot_trees
   :member-order: bysource
   :exclude-members: __init__


RandomForestClassifierBunch
===========================

.. autoclass:: explainerdashboard.explainers.RandomForestClassifierBunch
   :member-order: bysource
   :exclude-members: __init__


RandomForestClassifierBunch
===========================

.. autoclass:: explainerdashboard.explainers.RandomForestRegressionBunch
   :member-order: bysource
   :exclude-members: __init__


