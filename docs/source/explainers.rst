ExplainerBunch
**************

In order to start an **explainerdashboard** you first need to start an *ExplainerBunch*.

BaseExplainerBunch
==================
.. autoclass:: explainerdashboard.explainers.BaseExplainerBunch
   :members: mean_abs_shap_df, permutation_importances_df, importances_df, contrib_df, to_sql 
   :member-order: bysource

BaseExplainerBunch: Plots
=========================
.. autoclass:: explainerdashboard.explainers.BaseExplainerBunch
   :members: plot_importances, plot_shap_contributions, plot_shap_summary, plot_shap_interaction_summary, plot_shap_dependence, plot_shap_interaction_dependence, plot_pdp
   :member-order: bysource
   :exclude-members: __init__

ClassifierBunch
===============

.. autoclass:: explainerdashboard.explainers.ClassifierBunch
   :members: random_index, precision_df, plot_precision, plot_confusion_matrix, plot_roc_auc, plot_pr_auc
   :member-order: bysource

RandomForestBunch
=================

.. autoclass:: explainerdashboard.explainers.RandomForestBunch
   :members: shadowtree_df, shadowtree_df_summary, plot_trees
   :member-order: bysource
   :exclude-members: __init__

TreeModelClassifierBunch
========================

.. autoclass:: explainerdashboard.explainers.TreeModelClassifierBunch
   :member-order: bysource
   :exclude-members: __init__

RandomForestClassifierBunch
===========================

.. autoclass:: explainerdashboard.explainers.RandomForestClassifierBunch
   :member-order: bysource
   :exclude-members: __init__


