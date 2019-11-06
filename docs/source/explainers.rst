ExplainerBunch
**************

In order to start an **explainerdashboard** you first need to start an *ExplainerBunch*.

BaseExplainerBunch
==================
.. autoclass:: explainerdashboard.explainers.BaseExplainerBunch
   :members: __init__, mean_abs_shap_df, permutation_importances_df, importances_df, contrib_df, to_sql 

BaseExplainerBunch: Plots
=========================
.. autoclass:: explainerdashboard.explainers.BaseExplainerBunch
   :members: plot_importances, plot_shap_contributions, plot_shap_summary, plot_shap_interaction_summary, plot_shap_dependence, plot_shap_interaction_dependence, plot_pdp

ClassifierBunch
===============

.. autoclass:: explainerdashboard.explainers.ClassifierBunch
   :members: random_index, precision_df, plot_precision, plot_confusion_matrix, plot_roc_auc, plot_pr_auc

RandomForestBunch
=================

.. autoclass:: explainerdashboard.explainers.RandomForestBunch
   :members: shadowtree_df, shadowtree_df_summary, plot_trees

TreeModelClassifierBunch
========================

.. autoclass:: explainerdashboard.explainers.TreeModelClassifierBunch

RandomForestClassifierBunch
===========================

.. autoclass:: explainerdashboard.explainers.RandomForestClassifierBunch



