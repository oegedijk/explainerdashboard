Explainers
**********

In order to start an ``explainerdashboard`` you first need to construct an ``Explainer`` instance.


Explainers
==========
The abstract base class ``BaseExplainer`` defines most of the functionality 
such as feature importances (both SHAP and permutation based), SHAP values, SHAP interaction values
partial dependences, individual contributions, etc. Along with a number of convenient
plotting methods. In practice you will usually use ``ClassifierExplainer`` or ``RegressionExplainer``.

The BaseExplainer already provides a number of convenient plotting methods:

- ``plot_importances(...)``
- ``plot_shap_contributions(...)``
- ``plot_shap_summary(...)``
- ``plot_shap_interaction_summary(...)``
- ``plot_shap_dependence(...)``
- ``plot_shap_interaction_dependence(...)``
- ``plot_pdp(...)``

example code::

    explainer = BaseExplainer(model, X, y)
    explainer.plot_importances()
    explainer.plot_shap_contributions(index=0)


ClassifierExplainer
===================

For classification models where we try to predict the probability of each class
we have a derived class called ClassifierExplainer.

You can pass an additional parameter to ``__init__()`` with a list of label names. For
multilabel classifier you can set the positive class with e.g. ``explainer.pos_label=1``.
This will make sure that for example ``explainer.pred_probas`` will return the probability
of that label.You an also pass a string label if you passed ``labels`` to the constructor.

ClassifierExplainer defines a number of additional plotting methods:

- ``plot_precision(...)``
- ``plot_cumulative_precision(...)``
- ``plot_classification(...)``
- ``plot_confusion_matrix(...)``
- ``plot_lift_curve(...)``
- ``plot_roc_auc(...)``
- ``plot_pr_auc(...)``



example code::
    explainer = ClassifierExplainer(model, X, y, labels=['Not Survived', 'Survived'])
    explainer.plot_confusion_matrix(cutoff=0.6)
    explainer.plot_roc_auc()

RegressionExplainer
===================

For regression models where we try to predict a certain quantity.

You can pass an additional parameter to ``__init__()`` with the units of the predicted quantity.

RegressionExplainer defines a number of additional plotting methods:

-  ``plot_predicted_vs_actual(self, round=2, logs=False)``
-  ``plot_residuals(self, vs_actual=False, round=2, ratio=False)``
-  ``plot_residuals_vs_feature(self, col)``

example code::

    explainer = RegressionExplainer(model, X, y, units='dollars')
    explainer.plot_predicted_vs_actual()
    explainer.plot_residuals()


RandomForestExplainerExplainer
==============================

There is an additional mixin class specifically for sklearn RandomForests
that defines additional methods and plots to investigate and visualize 
individual decision trees within the random forest.

- ``RandomForestExplainer``: uses ``shap.TreeExplainer`` plus ``dtreeviz`` library to visualize individual decision trees.

You can get a pd.DataFrame summary of the path that a specific index took through a specific tree.
You can also plot the individual prediction of each individual tree. 

- ``decisiontree_df(tree_idx, index)``
- ``decisiontree_df_summary(tree_idx, index)``
- ``plot_trees(index)``

And for dtreeviz visualization of individual decision trees (svg format):
- ``decision_path()``
- ``decision_path_file()``
- ``decision_path_encoded()``

This also works with classifiers and regression models:

``explainer = RandomForestClassifierExplainer(model, X, y)``

``explainer = RandomForestRegressionExplainer(model, X, y)``

BaseExplainer
=============
.. autoclass:: explainerdashboard.explainers.BaseExplainer
   :members: mean_abs_shap_df, permutation_importances_df, importances_df, contrib_df, to_sql 
   :member-order: bysource

For the plotting methods see base_plots_

ClassifierExplainer
===================

.. autoclass:: explainerdashboard.explainers.ClassifierExplainer
   :members: random_index, precision_df
   :member-order: bysource

For the plotting methods see classifier_plots_

RegressionExplainer
===================

.. autoclass:: explainerdashboard.explainers.RegressionExplainer
   :members: random_index, residuals, metrics, prediction_result_markdown
   :member-order: bysource

For the plotting methods see regression_plots_


RandomForestExplainer
=====================

.. autoclass:: explainerdashboard.explainers.RandomForestExplainer
   :members: decisiontree_df, decisiontree_df_summary, plot_trees
   :member-order: bysource
   :exclude-members: __init__


RandomForestClassifierExplainer
===============================

.. autoclass:: explainerdashboard.explainers.RandomForestClassifierExplainer
   :member-order: bysource
   :exclude-members: __init__


RandomForestRegressionExplainer
===============================

.. autoclass:: explainerdashboard.explainers.RandomForestRegressionExplainer
   :member-order: bysource
   :exclude-members: __init__


