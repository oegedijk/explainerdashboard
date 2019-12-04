ExplainerBunch
**************

In order to start an **explainerdashboard** you first need to construct an **ExplainerBunch**.


ExplainersBunches
=================
The abstract base class **BaseExplainerBunch** defines most of the functionality 
such as feature importances (both SHAP and permutation based), SHAP values, SHAP interaction values
partial dependences, individual contributions, etc. Along with a number of convenient
plotting methods The derived classes that you would actually instantiate specify what kind of 
ShapExplainer to use to calculate the SHAP values.

e.g.:

- **TreeExplainerBunch**(model, X, y): uses shap.TreeExplainer
- **LinearExplainerBunch**(model, X, y): uses shap.LinearExplainer
- **DeepExplainerBunch**(model, X, y): uses shap.DeepExplainer
- **KernelExplainerBunch**(model, X, y): uses shap.KernelExplainer

These Explainer objects can be used for regression models where the output is a single
scalar prediction. They provide  number of convenient plotting methods:

- plot_importances()
- plot_shap_contributions()
- plot_shap_summary()
- plot_shap_interaction_summary()
- plot_shap_dependence()
- plot_shap_interaction_dependence()
- plot_pdp()

example code::

    explainer = TreeExplainerBunch(model, X, y)
    explainer.plot_importances()
    explainer.plot_shap_contributions(index=0)


ClassifierBunch
===============

For classification models where we try to predict the probability of each class
we have a MixIn class called ClassifierBunch.

You can pass an additional parameter to __init__ with a list label names. For
multilabel classifier you can set the positive class with e.g. *explainer.pos_label=1*.
This will make sure that for example *explainer.pred_probas* will return the probability
of that label.You an also pass a string label if you pased *labels* to the constructor.

ClassifierBunch defines a number of additional plotting methods:

- plot_precision()
- plot_confusion_matrix()
- plot_roc_auc()
- plot_pr_auc()

You would instantiate an Explainer class for a specific type of model, e.g.:

- **TreeClassifierBunch**(model, X, y)
- **LinearClassifierBunch**(model, X, y)
- **DeepClassifierBunch**(model, X, y)
- **KernelClassifierBunch**(model, X, y)


example code::

    explainer = TreeExplainerBunch(model, X, y, labels=['Not Survived', 'Survived'])
    explainer.plot_confusion_matrix()
    explainer.plot_roc_auc()

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

This also works with classifiers:

- **RandomForestClassifierBunch(model, X, y)**

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

TreeExplainerBunch
==================

.. autoclass:: explainerdashboard.explainers.TreeExplainerBunch
   :member-order: bysource
   :exclude-members: __init__


LinearExplainerBunch
====================

.. autoclass:: explainerdashboard.explainers.LinearExplainerBunch
   :member-order: bysource
   :exclude-members: __init__


DeepExplainerBunch
==================

.. autoclass:: explainerdashboard.explainers.DeepExplainerBunch
   :member-order: bysource
   :exclude-members: __init__


KernelExplainerBunch
====================

.. autoclass:: explainerdashboard.explainers.KernelExplainerBunch
   :member-order: bysource
   :exclude-members: __init__


RandomForestExplainerBunch
==========================

.. autoclass:: explainerdashboard.explainers.RandomForestExplainerBunch
   :members: shadowtree_df, shadowtree_df_summary, plot_trees
   :member-order: bysource
   :exclude-members: __init__


TreeClassifierBunch
===================

.. autoclass:: explainerdashboard.explainers.TreeClassifierBunch
   :member-order: bysource
   :exclude-members: __init__

RandomForestClassifierBunch
===========================

.. autoclass:: explainerdashboard.explainers.RandomForestClassifierBunch
   :member-order: bysource
   :exclude-members: __init__


