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

- ``plot_importances(kind='shap', topx=None, cats=False, round=3, pos_label=None)``
- ``plot_shap_contributions(index, cats=True, topx=None, cutoff=None, round=2, pos_label=None)``
- ``plot_shap_summary(topx=None, cats=False, pos_label=None)``
- ``plot_shap_interaction_summary(col, topx=None, cats=False, pos_label=None)``
- ``plot_shap_dependence(col, color_col=None, highlight_idx=None,pos_label=None)``
- ``plot_shap_interaction(col, interact_col, highlight_idx=None, pos_label=None)``
- ``plot_pdp(col, index=None, drop_na=True, sample=100, num_grid_lines=100, num_grid_points=10, pos_label=None)``

example code::

    explainer = BaseExplainer(model, X, y)
    explainer.plot_importances(cats=True)
    explainer.plot_shap_contributions(index=0)
    explainer.plot_shap_dependence("Fare")
    explainer.plot_shap_interaction("Fare", "PassengerClass")
    explainer.plot_pdp("Sex", index=0)

`More examples in the notebook on the github repo. <https://github.com/oegedijk/explainerdashboard/blob/master/explainer_examples.ipynb>`_

ClassifierExplainer
===================

For classification models where we try to predict the probability of each class
we have a derived class called ClassifierExplainer.

You can pass an additional parameter to ``__init__()`` with a list of label names. For
multilabel classifier you can set the positive class with e.g. ``explainer.pos_label=1``.
This will make sure that for example ``explainer.pred_probas`` will return the probability
of that label.You an also pass a string label if you passed ``labels`` to the constructor.

ClassifierExplainer defines a number of additional plotting methods:

- ``plot_precision(bin_size=None, quantiles=None, cutoff=None, multiclass=False, pos_label=None)``
- ``plot_cumulative_precision(pos_label=None)``
- ``plot_classification(cutoff=0.5, percentage=True, pos_label=None)``
- ``plot_confusion_matrix(cutoff=0.5, normalized=False, binary=False, pos_label=None)``
- ``plot_lift_curve(cutoff=None, percentage=False, round=2, pos_label=None)``
- ``plot_roc_auc(cutoff=0.5, pos_label=None)``
- ``plot_pr_auc(cutoff=0.5, pos_label=None)``


example code::

    explainer = ClassifierExplainer(model, X, y, labels=['Not Survived', 'Survived'])
    explainer.plot_confusion_matrix(cutoff=0.6)
    explainer.plot_precision(quantiles=10, cutoff=0.6, multiclass=True)
    explainer.plot_lift_curve(percentage=True)
    explainer.plot_roc_auc(cutoff=0.7)
    explainer.plot_pr_auc(cutoff=0.3)

`More examples in the notebook on the github repo. <https://github.com/oegedijk/explainerdashboard/blob/master/explainer_examples.ipynb>`_


RegressionExplainer
===================

For regression models where we try to predict a certain quantity.

You can pass an additional parameter to ``__init__()`` with the units of the predicted quantity.

RegressionExplainer defines a number of additional plotting methods:

-  ``plot_predicted_vs_actual(round=2, logs=False)``
-  ``plot_residuals(vs_actual=False, round=2, ratio=False)``
-  ``plot_residuals_vs_feature(col)``

example code::

    explainer = RegressionExplainer(model, X, y, units='dollars')
    explainer.plot_predicted_vs_actual()
    explainer.plot_residuals()

`More examples in the notebook on the github repo. <https://github.com/oegedijk/explainerdashboard/blob/master/explainer_examples.ipynb>`_


RandomForestExplainerExplainer
==============================

There is an additional mixin class specifically for sklearn RandomForests
that defines additional methods and plots to investigate and visualize 
individual decision trees within the random forest.

- ``RandomForestExplainer``: uses ``dtreeviz`` library to visualize individual decision trees.

You can get a pd.DataFrame summary of the path that a specific index took through a specific tree.
You can also plot the individual prediction of each individual tree. 

- ``decisiontree_df(tree_idx, index)``
- ``decisiontree_df_summary(tree_idx, index)``
- ``plot_trees(index)``

And for dtreeviz visualization of individual decision trees (svg format):
- ``decision_path(tree_idx, index)``
- ``decision_path_file(tree_idx, index)``
- ``decision_path_encoded(tree_idx, index)``

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


