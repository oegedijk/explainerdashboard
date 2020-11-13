Explainers
**********

Simple example
==============

In order to start an ``ExplainerDashboard`` you first need to construct an 
``Explainer`` instance. They come in two flavours and at its most basic they 
only need a model, and a test set X and y::

    from explainerdashboard import ClassifierExplainer, RegressionExplainer

    explainer = ClassifierExplainer(model, X_test, y_test)
    explainer = RegressionExplainer(model, X_test, y_test)


This is enough to launch an ExplainerDashboard::

    from explainerdashboard import ExplainerDashboard
    ExplainerDashboard(explainer).run()

.. image:: screenshots/screenshot.*


Or you can use it interactively in a notebook to inspect your model 
using the built-in plotting methods, e.g.::

    explainer.plot_confusion_matrix()
    explainer.plot_shap_contributions(index=0)
    explainer.plot_shap_dependence("Fare", color_col="Sex")

.. image:: screenshots/notebook_screenshot.png

For the full lists of plots available see :ref:`Plots<Plots>`.


Or you can start an interactive :ref:`ExplainerComponent<ExplainerComponents>` 
in your notebook using :ref:`InlineExplainer<InlineExplainer>`, e.g.::

    from explainerdashboard import InlineExplainer
    
    InlineExplainer(explainer).tab.importances()
    InlineExplainer(explainer).classifier.roc_auc()
    InlineExplainer(explainer).regression.residuals_vs_col()
    InlineExplainer(explainer).shap.overview()

.. image:: screenshots/inline_screenshot.*


Parameters
==========

There are a number of optional parameters that can either make sure that
SHAP values get calculated in the appropriate way, or that make the explainer 
give a bit nicer and more convenient output::

    ClassifierExplainer(model, X_test, y_test, 
            shap='linear', # manually set shap type, overrides default 'guess'
            X_background=X_train, # set background dataset for shap calculations
            model_output='logodds', # set model_output to logodds (vs probability)
            cats=['Sex', 'Deck', 'Embarked'], # makes it easy to group onehotencoded vars
            idxs=test_names, # index with str identifier
            descriptions=feature_descriptions, # show long feature descriptions in hovers
            target='Survival', # the name of the target variable (y)
            labels=['Not survived', 'Survived']) # show target labels instead of ['0', '1']

cats
----

If you have onehot-encoded your categorical variables, they will show up as a 
lot of independent features. This clutters your feature space, and often makes 
it hard to interpret the effect of the underlying categorical feature. 

However as long as you encoded the onehot categories with an underscore like 
``CategoricalFeature_Category``, then ``explainerdashboard`` can group these 
categories together again in the various plots and tables. For this you can 
pass a list of onehotencoded categorical features to the parameter ``cats``.
For the titanic example this would be:
    - ``Sex``: ``Sex_female``, ``Sex_male``
    - ``Deck``: ``Deck_A``, ``Deck_B``, etc
    - ``Embarked``: ``Embarked_Southampton``, ``Embarked_Cherbourg``, etc

So you would pass ``cats=['Sex', 'Deck', 'Embarked']``. You can now use these 
categorical features directly as input for plotting methods, e.g. 
``explainer.plot_shap_dependence("Deck")``. For other methods you can pass
a parameter ``cats=True``, to indicate that you'd like to group the categorical
features in your output. 

idxs
----

You may have specific identifiers (names, customer id's, etc) for each row in your dataset.
If you pass these the the Explainer object, you can index using both the 
numerical index, e.g. ``explainer.contrib_df(0)`` for the first row, or using the 
identifier, e.g. ``explainer.contrib_df("Braund, Mr. Owen Harris")``.

The proper name or idxs will be use used in all ``ExplainerComponents`` that
allow index selection. 

By default the index of pandas dataframe ``X`` will be used as idxs.

descriptions
------------

``descriptions`` can be passed as a dictionary of descriptions for each variable.
In order to be explanatory, you often have to explain the meaning of the features 
themselves (especially if the naming is not obvious).
Passing the dict along to descriptions will show hover-over tooltips for the 
various features in the dashboard.

target
------

Name of the target variable. By default the name of the pd.Series ``y`` is used

labels
------
labels: The outcome variables for a classification  ``y`` are assumed to 
be encoded 0, 1 (, 2, 3, ...) This is not very human readable, so you can pass a 
list of human readable labels such as ``labels=['Not survived', 'Survived']``.

units
-----

For regression models the units of the ``y`` variable. E.g. if the model is predicting
house prices in dollar you can set ``units='$'``. This will be displayed along
the axis of various plots.


X_background
------------

Some models like sklearn ``LogisticRegression`` (as well as certain gradienst boosting 
algorithms in probability space) need a background dataset to calculate shap values. 
These can be passed as ``X_background``. If you don't pass an X_background, Explainer 
uses X instead but gives off a warning.

Usually a representative background dataset of a couple of hunderd rows should be
enough to get decent shap values.

model_output
------------

By default ``model_output`` for classifiers is set to ``"probability"``, as this 
is more intuitively explainable to non data scientist stakeholders.
However for certain models (e.g. ``XGBClassifier``, ``LGBMCLassifier``, ``CatBoostClassifier``), 
need a background dataset X_background to calculate shap values in probability 
space, and are not able to calculate shap interaction values in probability space.
Therefore you can also pass model_output='logodds', in which case shap values 
get calculated faster and interaction effects can be studied. Now you just need
to explain to your stakeholders what logodds are :)

shap
----

By default ``shap='guess'``, which means that the Explainer will try to guess 
based on the model what kind of shap explainer it needs: e.g. 
``shap.TreeExplainer(...)``, ``shap.LinearExplainer(...)``, etc.

In case the guess fails or you'd like to override it, you can set it manually:
e.g. ``shap='tree'``, ``shap='linear'``, ``shap='kernel'``, ``shap='deep'``, etc.

model_output, X_background example
----------------------------------

An example of using setting ``X_background`` and ``model_output`` with a 
LogisticRegression::

    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression()
    model.fit(X_train, y_train)

    explainer = ClassifierExplainer(model, X_test, y_test, 
                                        shap='linear', 
                                        X_background=X_train, 
                                        model_output='logodds')
    ExplainerDashboard(explainer).run()


permutation_cv
--------------

Normally permutation importances get calculated over a single fold (assuming the
data is the test set). However if you pass the training set to the explainer,
you may wish to cross-validate calculate the permutation importances. In that
case pass the number of folds to ``permutation_cv``.

na_fill
-------

If you fill missing values with some extreme value such as ``-999`` (typical for
tree based methods), these can mess with the horizontal axis of your plots. 
In order to filter these out, you need to tell the explainer what the extreme value 
is that you used to fill. Defaults to ``-999``.

Plots
=====

Shared Plots
------------

The abstract base class ``BaseExplainer`` defines most of the functionality 
such as feature importances (both SHAP and permutation based), SHAP values, SHAP interaction values
partial dependences, individual contributions, etc. Along with a number of convenient
plotting methods. In practice you will use ``ClassifierExplainer`` 
or ``RegressionExplainer``, however they both inherit all of these basic methods.


The BaseExplainer already provides a number of convenient plotting methods::

    plot_importances(kind='shap', topx=None, cats=False, round=3, pos_label=None)
    plot_shap_contributions(index, cats=True, topx=None, cutoff=None, round=2, pos_label=None)
    plot_shap_summary(topx=None, cats=False, pos_label=None)
    plot_shap_interaction_summary(col, topx=None, cats=False, pos_label=None)
    plot_shap_dependence(col, color_col=None, highlight_idx=None,pos_label=None)
    plot_shap_interaction(col, interact_col, highlight_idx=None, pos_label=None)
    plot_pdp(col, index=None, drop_na=True, sample=100, num_grid_lines=100, num_grid_points=10, pos_label=None)

example code::

    explainer = ClassifierExplainer(model, X, y, cats=['Sex', 'Deck', 'Embarked']) 
    explainer.plot_importances(cats=True)
    explainer.plot_shap_contributions(index=0, topx=5)
    explainer.plot_shap_dependence("Fare")
    explainer.plot_shap_interaction("Fare", "PassengerClass")
    explainer.plot_pdp("Sex", index=0)

plot_importances
^^^^^^^^^^^^^^^^

.. automethod:: explainerdashboard.explainers.BaseExplainer.plot_importances

plot_shap_contributions
^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: explainerdashboard.explainers.BaseExplainer.plot_shap_contributions

plot_shap_dependence
^^^^^^^^^^^^^^^^^^^^

.. automethod:: explainerdashboard.explainers.BaseExplainer.plot_shap_dependence

plot_shap_interaction
^^^^^^^^^^^^^^^^^^^^^

.. automethod:: explainerdashboard.explainers.BaseExplainer.plot_shap_interaction

plot_pdp
^^^^^^^^

.. automethod:: explainerdashboard.explainers.BaseExplainer.plot_pdp

plot_shap_summary
^^^^^^^^^^^^^^^^^

.. automethod:: explainerdashboard.explainers.BaseExplainer.plot_shap_summary

plot_shap_interaction_summary
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: explainerdashboard.explainers.BaseExplainer.plot_shap_interaction_summary



Classifier Plots
----------------

``ClassifierExplainer`` defines a number of additional plotting methods::

    plot_precision(bin_size=None, quantiles=None, cutoff=None, multiclass=False, pos_label=None)
    plot_cumulative_precision(pos_label=None)
    plot_classification(cutoff=0.5, percentage=True, pos_label=None)
    plot_confusion_matrix(cutoff=0.5, normalized=False, binary=False, pos_label=None)
    plot_lift_curve(cutoff=None, percentage=False, round=2, pos_label=None)
    plot_roc_auc(cutoff=0.5, pos_label=None)
    plot_pr_auc(cutoff=0.5, pos_label=None)


example code::

    explainer = ClassifierExplainer(model, X, y, labels=['Not Survived', 'Survived'])
    explainer.plot_confusion_matrix(cutoff=0.6)
    explainer.plot_precision(quantiles=10, cutoff=0.6, multiclass=True)
    explainer.plot_lift_curve(percentage=True)
    explainer.plot_roc_auc(cutoff=0.7)
    explainer.plot_pr_auc(cutoff=0.3)

More examples in the `notebook on the github repo. <https://github.com/oegedijk/explainerdashboard/blob/master/explainer_examples.ipynb>`_

plot_precision
^^^^^^^^^^^^^^

.. automethod:: explainerdashboard.explainers.ClassifierExplainer.plot_precision

plot_cumulative_precision
^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: explainerdashboard.explainers.ClassifierExplainer.plot_cumulative_precision

plot_classification
^^^^^^^^^^^^^^^^^^^

.. automethod:: explainerdashboard.explainers.ClassifierExplainer.plot_classification

plot_confusion_matrix
^^^^^^^^^^^^^^^^^^^^^

.. automethod:: explainerdashboard.explainers.ClassifierExplainer.plot_confusion_matrix

plot_lift_curve
^^^^^^^^^^^^^^^

.. automethod:: explainerdashboard.explainers.ClassifierExplainer.plot_lift_curve

plot_roc_auc
^^^^^^^^^^^^

.. automethod:: explainerdashboard.explainers.ClassifierExplainer.plot_roc_auc

plot_pr_auc
^^^^^^^^^^^

.. automethod:: explainerdashboard.explainers.ClassifierExplainer.plot_pr_auc

Regression Plots 
----------------

For the derived RegressionExplainer class again some additional plots::

    explainer.plot_predicted_vs_actual(...)
    explainer.plot_residuals(...)
    explainer.plot_residuals_vs_feature(...)

plot_predicted_vs_actual
^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: explainerdashboard.explainers.RegressionExplainer.plot_predicted_vs_actual

plot_residuals
^^^^^^^^^^^^^^

.. automethod:: explainerdashboard.explainers.RegressionExplainer.plot_residuals

plot_residuals_vs_feature
^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: explainerdashboard.explainers.RegressionExplainer.plot_residuals_vs_feature


DecisionTree Plots
------------------

There are additional mixin classes specifically for ``sklearn`` ``RandomForests``
and for xgboost models that define additional methods and plots to investigate and visualize 
individual decision trees within the ensemblke. These
uses the ``dtreeviz`` library to visualize individual decision trees.

You can get a pd.DataFrame summary of the path that a specific index row took 
through a specific decision tree.
You can also plot the individual predictions of each individual tree for 
specific row in your data indentified by ``index``::

    explainer.decisiontree_df(tree_idx, index)
    explainer.decisiontree_summary_df(tree_idx, index)
    explainer.plot_trees(index)

And for dtreeviz visualization of individual decision trees (svg format)::

    explainer.decision_path(tree_idx, index)
    explainer.decision_path_file(tree_idx, index)
    explainer.decision_path_encoded(tree_idx, index)

These methods are part of the ``RandomForestExplainer`` and XGBExplainer`` mixin
classes that get automatically loaded when you pass either a RandomForest
or XGBoost model.


plot_trees
^^^^^^^^^^

.. automethod:: explainerdashboard.explainers.RandomForestExplainer.plot_trees

decision_path
^^^^^^^^^^^^^

.. automethod:: explainerdashboard.explainers.RandomForestExplainer.decision_path

decision_path_file
^^^^^^^^^^^^^^^^^^

.. automethod:: explainerdashboard.explainers.RandomForestExplainer.decision_path_file

decision_path_encoded
^^^^^^^^^^^^^^^^^^^^^

.. automethod:: explainerdashboard.explainers.RandomForestExplainer.decision_path_encoded



Other explainer outputs
=======================

Base outputs
------------

Some other useful tables and outputs you can get out of the explainer::

    metrics()
    metrics_markdown(round=2)
    mean_abs_shap_df(topx=None, cutoff=None, cats=False, pos_label=None)
    permutation_importances_df(topx=None, cutoff=None, cats=False, pos_label=None)
    importances_df(kind="shap", topx=None, cutoff=None, cats=False, pos_label=None)
    contrib_df(index, cats=True, topx=None, cutoff=None, pos_label=None)
    contrib_summary_df(index, cats=True, topx=None, cutoff=None, round=2, pos_label=None)
    interactions_df(col, cats=False, topx=None, cutoff=None, pos_label=None)

metrics
^^^^^^^

.. automethod:: explainerdashboard.explainers.BaseExplainer.metrics

metrics_markdown
^^^^^^^^^^^^^^^^

.. automethod:: explainerdashboard.explainers.BaseExplainer.metrics_markdown

mean_abs_shap_df
^^^^^^^^^^^^^^^^

.. automethod:: explainerdashboard.explainers.BaseExplainer.mean_abs_shap_df

permutation_importances_df
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: explainerdashboard.explainers.BaseExplainer.permutation_importances_df

importances_df
^^^^^^^^^^^^^^

.. automethod:: explainerdashboard.explainers.BaseExplainer.importances_df

contrib_df
^^^^^^^^^^

.. automethod:: explainerdashboard.explainers.BaseExplainer.contrib_df

contrib_summary_df
^^^^^^^^^^^^^^^^^^

.. automethod:: explainerdashboard.explainers.BaseExplainer.contrib_summary_df

interactions_df
^^^^^^^^^^^^^^^

.. automethod:: explainerdashboard.explainers.BaseExplainer.interactions_df



Classifier outputs
------------------

For ``ClassifierExplainer`` in addition::

    random_index(y_values=None, return_str=False,pred_proba_min=None, pred_proba_max=None,
                    pred_percentile_min=None, pred_percentile_max=None, pos_label=None)
    prediction_result_markdown(index, include_percentile=True, round=2, pos_label=None)
    cutoff_from_percentile(percentile, pos_label=None)
    precision_df(bin_size=None, quantiles=None, multiclass=False, round=3, pos_label=None)
    lift_curve_df(pos_label=None)


random_index
^^^^^^^^^^^^

.. automethod:: explainerdashboard.explainers.ClassifierExplainer.random_index

prediction_result_markdown
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: explainerdashboard.explainers.ClassifierExplainer.prediction_result_markdown

cutoff_from_percentile
^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: explainerdashboard.explainers.ClassifierExplainer.cutoff_from_percentile

precision_df
^^^^^^^^^^^^

.. automethod:: explainerdashboard.explainers.ClassifierExplainer.precision_df

lift_curve_df
^^^^^^^^^^^^^

.. automethod:: explainerdashboard.explainers.ClassifierExplainer.lift_curve_df


Regression outputs
------------------


For ``RegressionExplainer``::

    prediction_result_markdown(index, round=2)
    random_index(y_min=None, y_max=None, pred_min=None, pred_max=None, 
                    residuals_min=None, residuals_max=None,
                    abs_residuals_min=None, abs_residuals_max=None,
                    return_str=False)

prediction_result_markdown
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: explainerdashboard.explainers.RegressionExplainer.prediction_result_markdown

random_index
^^^^^^^^^^^^

.. automethod:: explainerdashboard.explainers.RegressionExplainer.random_index


RandomForest and XGBoost outputs
--------------------------------

For RandomForest and XGBoost models mixin classes that visualize individual 
decision trees will be loaded: ``RandomForestExplainer`` and ``XGBExplainer``
with the following additional methods::

    decisiontree_df(tree_idx, index, pos_label=None)
    decisiontree_summary_df(tree_idx, index, round=2, pos_label=None)
    decision_path_file(tree_idx, index)
    decision_path_encoded(tree_idx, index)
    decision_path(tree_idx, index)


decisiontree_df
^^^^^^^^^^^^^^^

.. automethod:: explainerdashboard.explainers.RandomForestExplainer.decisiontree_df

decisiontree_summary_df
^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: explainerdashboard.explainers.RandomForestExplainer.decisiontree_summary_df

decision_path_file
^^^^^^^^^^^^^^^^^^

.. automethod:: explainerdashboard.explainers.RandomForestExplainer.decision_path_file

decision_path_encoded
^^^^^^^^^^^^^^^^^^^^^

.. automethod:: explainerdashboard.explainers.RandomForestExplainer.decision_path_encoded

decision_path
^^^^^^^^^^^^^

.. automethod:: explainerdashboard.explainers.RandomForestExplainer.decision_path


Calculated Properties
=====================

In general ``Explainers`` don't calculate any properties of the model or the 
data until they are needed for an output, so-called lazy calculation. When the
property is calculated once, it is stored for next time. So the first time 
you invoke a plot involving shap values may take a while to calculate. The next
time will be basically instant. 

You can access these properties directly from the explainer, e.g. ``explainer.shap_values``. 
For classifier models if you want values for a particular ``pos_label`` you can
pass this (int) label ``explainer.shap_values(0)`` would get the shap values for the 0'th class.

In order to calculate all properties of the explainer at once, you can call
``explainer.calculate_properties()``. (``ExplainerComponents`` have a similar method
``component.calculate_dependencies()`` to calculate all properties that that specific
component will need). 

The various properties are::

    explainer.preds
    explainer.pred_percentiles
    explainer.permutation_importances
    explainer.permutation_importances_cats
    explainer.shap_base_value
    explainer.shap_values
    explainer.shap_values_cats
    explainer.shap_interaction_values
    explainer.shap_interaction_values_cats
    explainer.mean_abs_shap
    explainer.mean_abs_shap_cats

For ``ClassifierExplainer``::

    explainer.y_binary
    explainer.pred_probas_raw
    explainer.pred_percentiles_raw
    explainer.pred_probas

For ``RegressionExplainer``::

    explainer.residuals
    explainer.abs_residuals


Setting pos_label
=================

For ``ClassifierExplainer`` you can calculate most properties for multiple labels as
the positive label. With a binary classification usually label '1' is the positive class,
but in some cases you might also be interested in the '0' label.

For multiclass classification you may want to investigate shap dependences for
the various classes.

You can pass a parameter ``pos_label`` to almost every property or method, to get
the output for that specific positive label. If you don't pass a ``pos_label`` 
manually to a specific method, the global ``pos_label`` will be used. You can set
this directly on the explainer (even us str labels if you've set these)::

    explainer.pos_label = 0
    explainer.plot_shap_dependence("Fare") # will show plot for pos_label=0
    explainer.pos_label = 'Survived' 
    explainer.plot_shap_dependence("Fare") # will now show plot for pos_label=1
    explainer.plot_shap_dependence("Fare", pos_label=0) # show plot for label 0, without changing explainer.pos_label

The ``ExplainerDashboard`` will show a dropdown menu in the header to choose
a particular ``pos_label``. Changing this will basically update every single
plot in the dashboard. 


BaseExplainer
=============

.. autoclass:: explainerdashboard.explainers.BaseExplainer
   :members: mean_abs_shap_df, permutation_importances_df, importances_df, contrib_df, to_sql,
            plot_importances, plot_shap_contributions, plot_shap_summary, 
            plot_shap_interaction_summary, plot_shap_dependence, plot_shap_interaction, plot_pdp
   :member-order: bysource

ClassifierExplainer
===================

For classification models where we try to predict the probability of each class
we have a derived class called ClassifierExplainer.

You can pass an additional parameter to ``__init__()`` with a list of label names. For
multilabel classifier you can set the positive class with e.g. ``explainer.pos_label=1``.
This will make sure that for example ``explainer.pred_probas`` will return the probability
of that label.You an also pass a string label if you passed ``labels`` to the constructor.

More examples in the `notebook on the github repo. <https://github.com/oegedijk/explainerdashboard/blob/master/explainer_examples.ipynb>`_


.. autoclass:: explainerdashboard.explainers.ClassifierExplainer
   :members: random_index, precision_df, 
        plot_precision, plot_cumulative_precision, plot_classification, 
        plot_lift_curve, plot_confusion_matrix, plot_roc_auc, plot_pr_auc
   :member-order: bysource
   :noindex:


RegressionExplainer
===================

For regression models where we try to predict a certain quantity.

You can pass an additional parameter to ``__init__()`` with the units of the predicted quantity.

More examples in the `notebook on the github repo. <https://github.com/oegedijk/explainerdashboard/blob/master/explainer_examples.ipynb>`_

.. autoclass:: explainerdashboard.explainers.RegressionExplainer
   :members: random_index, residuals, metrics, prediction_result_markdown,
            plot_predicted_vs_actual, plot_residuals,  plot_residuals_vs_feature
   :member-order: bysource
   :noindex:


RandomForestExplainer
=====================

The ``RandomForestExplainer`` mixin class provides additional functionality
in order to explore individual decision trees within the RandomForest.
This can be very useful for showing stakeholders that a RandomForest is
indeed just a collection of simple decision trees that you then calculate
the average off. This Mixin class will be automatically included
whenever you pass a ``RandomForestClassifier`` or ``RandomForestRegressor`` model.

.. autoclass:: explainerdashboard.explainers.RandomForestExplainer
   :members: decisiontree_df, decisiontree_summary_df, plot_trees, decision_path
   :member-order: bysource
   :exclude-members: 
   :noindex:




XGBExplainer
============

The ``XGBExplainer`` mixin class provides additional functionality
in order to explore individual decision trees within an xgboost ensemble model.
This can be very useful for showing stakeholders that a xgboost is
indeed just a collection of simple decision trees that get summed together. 
This Mixin class will be automatically included
whenever you pass a ``XGBClassifier`` or ``XGBRegressor`` model.


.. autoclass:: explainerdashboard.explainers.XGBExplainer
   :members: decisiontree_df, decisiontree_summary_df, plot_trees, decision_path
   :member-order: bysource
   :exclude-members: 
   :noindex:





