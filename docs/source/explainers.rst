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
    explainer.plot_contributions(index=0)
    explainer.plot_dependence("Fare", color_col="Sex")

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
            index_name="Passenger", # description of index
            descriptions=feature_descriptions, # show long feature descriptions in hovers
            target='Survival', # the name of the target variable (y)
            precision='float32', # save memory by setting lower precision. Default is 'float64'
            labels=['Not survived', 'Survived']) # show target labels instead of ['0', '1']

cats
----

If you have onehot-encoded your categorical variables, they will show up as a 
lot of independent features. This clutters your feature space, and often makes 
it hard to interpret the effect of the underlying categorical feature. 

You can pass a ``dict`` to the parameter ``cats`` specifying which are the
onehotencoded columns, and what the grouped feature name should be::

    ClassifierExplainer(model, X, y, cats={'Gender': ['Sex_male', 'Sex_female']})

However if you encoded your feature with ``pd.get_dummies(df, prefix=['Name'])``,
then the resulting onehot encoded columns should be named 
'Name_John', 'Name_Mary', Name_Bob', etc. (or in general
CategoricalFeature_Category), then you can simply pass a list of the prefixes
to cats::

    ClassifierExplainer(model, X, y, cats=['Sex', 'Deck', 'Embarked'])

And you can also combine the two methods::

    ClassifierExplainer(model, X, y, 
        cats=[{'Gender': ['Sex_male', 'Sex_female']}, 'Deck', 'Embarked'])



You can now use these categorical features directly as input for plotting methods, e.g. 
``explainer.plot_dependence("Deck")``, which will now generate violin plots
instead of the default scatter plots. 

cats_notencoded
---------------

When you have onehotencoded a categorical feature, you may have dropped some columns
during feature selection. Or there are new categories in the test set that were not encoded
as columns in the training set. In that cases all columns in your onehot encoding may be equal 
to ``0`` for some rows. By default the value assigned to the aggregated feature for such cases is ``'NOT_ENCODED'``,
but this can be overriden with the ``cats_notencoded`` parameter::

    ClassifierExplainer(model, X, y, 
        cats=[{'Gender': ['Sex_male', 'Sex_female']}, 'Deck', 'Embarked'],
        cats_notencoded={'Gender': 'Gender Other', 'Deck': 'Unknown Deck', 'Embarked':'Stowaway'})



idxs
----

You may have specific identifiers (names, customer id's, etc) for each row in 
your dataset. By default ``X.index`` will get used
to identify individual rows/records in the dashboard. And you can index using both the 
numerical index, e.g. ``explainer.get_contrib_df(0)`` for the first row, or using the 
identifier, e.g. ``explainer.get_contrib_df("Braund, Mr. Owen Harris")``.

You can override using ``X.index`` by passing a list/array/Series ``idxs``
to the explainer::

    from explainerdashboard.datasets import titanic_names

    test_names = titanic_names(test_only=True)
    ClassifierExplainer(model, X_test, y_test, idxs=test_names)

index_name
----------

By default ``X.index.name`` or ``idxs.name`` is used as the description of the index,
but you can also pass it explicitly, e.g.: ``index_name="Passenger"``.

descriptions
------------

``descriptions`` can be passed as a dictionary of descriptions for each feature.
In order to be explanatory, you often have to explain the meaning of the features 
themselves (especially if the naming is not obvious).
Passing the dict along to descriptions will show hover-over tooltips for the 
various features in the dashboard. If you grouped onehotencoded features with
the ``cats`` parameter, you can also give descriptions of these groups, e.g::

    ClassifierExplainer(model, X, y, 
        cats=[{'Gender': ['Sex_male', 'Sex_female']}, 'Deck', 'Embarked'],
        descriptions={
            'Gender': 'Gender of the passenger',
            'Fare': 'The price of the ticket paid for by the passenger',
            'Deck': 'The deck of the cabin of the passenger',
            'Age': 'Age of the passenger in year'
        })


target
------

Name of the target variable. By default the name of the ``y`` (``y.name``) is used 
if ``y`` is a ``pd.Series``, else it defaults to ``'target'``, bu this can be overriden::

    ClassifierExplainer(model, X, y, target="Survival")

labels
------
For ``ClassifierExplainer`` only: The outcome variables for a classification  ``y`` are assumed to 
be encoded ``0, 1 (, 2, 3, ...)`` You can assign string labels by passing e.g.
``labels=['Not survived', 'Survived']``::

    ClassifierExplainer(model, X, y, labels=['Not survived', 'Survived'])

units
-----

For ``RegressionExplainer`` only: the units of the ``y`` variable. E.g. if the model is predicting
house prices in dollars you can set ``units='$'``. If it is predicting maintenance
time you can set ``units='hours'``, etc. This will then be displayed along
the axis of various plots::

    RegressionExplainer(model, X, y, units="$")


X_background
------------

Some models like sklearn ``LogisticRegression`` (as well as certain gradient boosting 
algorithms such as `xgboost` in probability space) need a background dataset to calculate shap values. 
These can be passed as ``X_background``. If you don't pass an X_background, Explainer 
uses X instead but gives off a warning. (You want to limit the size of X_background
in order to keep the SHAP calculations from getting too slow. Usually a representative 
background dataset of a couple of hunderd rows should be enough to get decent shap values.)

model_output
------------

By default ``model_output`` for classifiers is set to ``"probability"``, as this 
is more intuitively explainable to non data scientist stakeholders.
However certain models (e.g. ``XGBClassifier``, ``LGBMCLassifier``, ``CatBoostClassifier``), 
need a background dataset ``X_background`` to calculate SHAP values in probability 
space, and are not able to calculate shap interaction values in probability space at all.
Therefore you can also pass model_output='logodds', in which case shap values 
get calculated faster and interaction effects can be studied. Now you just need
to explain to your stakeholders what logodds are :)

shap
----

By default ``shap='guess'``, which means that the Explainer will try to guess 
based on the model what kind of shap explainer it needs: e.g. 
``shap.TreeExplainer(...)``, ``shap.LinearExplainer(...)``, etc.

In case the guess fails or you'd like to override it, you can set it manually:
e.g. ``shap='tree'`` for ``shap.TreeExplainer``, ``shap='linear'`` for ``shap.LinearExplainer``, 
``shap='kernel'`` for ``shap.KernelExplainer``, ``shap='deep'`` for ``shap.DeepExplainer``, etc.

model_output, X_background example
----------------------------------

An example of using setting ``X_background`` and ``model_output`` with a 
``LogisticRegression``::

    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression()
    model.fit(X_train, y_train)

    explainer = ClassifierExplainer(model, X_test, y_test, 
                                        shap='linear', 
                                        X_background=X_train, 
                                        model_output='logodds')
    ExplainerDashboard(explainer).run()


cv
--

Normally metrics and permutation importances get calculated over a single fold 
(assuming the data ``X`` is the test set). However if you pass the training set 
to the explainer, you may wish to cross-validate calculate the permutation 
importances and metrics. In that case pass the number of folds to ``cv``. 
Note that custom metrics do not work with cross validation for now.


na_fill
-------

If you fill missing values with some extreme value such as ``-999`` (typical for
tree based methods), these can mess with the horizontal axis of your plots. 
In order to filter these out, you need to tell the explainer what the extreme value 
is that you used to fill. Defaults to ``-999``.

precision
---------

You can set the precision of the calculated shap values, predictions, etc, in
order to save on memory usage. Default is ``'float64'``, but ``'float32'`` is probably
fine, maybe even ``'float16'`` for your application.

Pre-calculated shap values
==========================

Perhaps you already have calculated the shap values somewhere, or you can calculate 
them off on a giant cluster somewhere, or your model supports `GPU generated shap values <https://github.com/rapidsai/gputreeshap>`_. 
    
You can simply add these pre-calculated shap values to the explainer with 
``explainer.set_shap_values()`` and ``explainer.set_shap_interaction_values()`` methods.


Plots
=====

Shared Plots
------------

The abstract base class ``BaseExplainer`` defines most of the functionality 
such as feature importances (both SHAP and permutation based), SHAP values, SHAP interaction values
partial dependences, individual contributions, etc. Along with a number of convenient
plotting methods. In practice you will use ``ClassifierExplainer`` 
or ``RegressionExplainer``, however they both inherit all of these basic methods::


    plot_importances(kind='shap', topx=None, round=3, pos_label=None)
    plot_contributions(index, topx=None, cutoff=None, round=2, pos_label=None)
    plot_importances_detailed(topx=None, pos_label=None)
    plot_interactions_detailed(col, topx=None, pos_label=None)
    plot_dependence(col, color_col=None, highlight_idx=None, pos_label=None)
    plot_interaction(interact_col, highlight_idx=None, pos_label=None)
    plot_pdp(col, index=None, drop_na=True, sample=100, num_grid_lines=100, num_grid_points=10, pos_label=None)

example code::

    explainer = ClassifierExplainer(model, X, y, cats=['Sex', 'Deck', 'Embarked']) 
    explainer.plot_importances()
    explainer.plot_contributions(index=0, topx=5)
    explainer.plot_dependence("Fare")
    explainer.plot_interaction("Embarked", "PassengerClass")
    explainer.plot_pdp("Sex", index=0)

plot_importances
^^^^^^^^^^^^^^^^

.. automethod:: explainerdashboard.explainers.BaseExplainer.plot_importances

plot_importances_detailed
^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: explainerdashboard.explainers.BaseExplainer.plot_importances_detailed

plot_contributions
^^^^^^^^^^^^^^^^^^

.. automethod:: explainerdashboard.explainers.BaseExplainer.plot_contributions

plot_dependence
^^^^^^^^^^^^^^^

.. automethod:: explainerdashboard.explainers.BaseExplainer.plot_dependence

plot_interaction
^^^^^^^^^^^^^^^^

.. automethod:: explainerdashboard.explainers.BaseExplainer.plot_interaction

plot_pdp
^^^^^^^^

.. automethod:: explainerdashboard.explainers.BaseExplainer.plot_pdp

plot_interactions_importance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: explainerdashboard.explainers.BaseExplainer.plot_interactions_importance

plot_interactions_detailed
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: explainerdashboard.explainers.BaseExplainer.plot_interactions_detailed


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

More examples in the `notebook on the github repo. <https://github.com/oegedijk/explainerdashboard/blob/master/notebooks/explainer_examples.ipynb>`_

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

    explainer.get_decisionpath_df(tree_idx, index)
    explainer.get_decisionpath_summary_df(tree_idx, index)
    explainer.plot_trees(index)

And for dtreeviz visualization of individual decision trees (svg format)::

    explainer.decisiontree(tree_idx, index)
    explainer.decisiontree_file(tree_idx, index)
    explainer.decisiontree_encoded(tree_idx, index)

These methods are part of the ``RandomForestExplainer`` and XGBExplainer`` mixin
classes that get automatically loaded when you pass either a RandomForest
or XGBoost model.


plot_trees
^^^^^^^^^^

.. automethod:: explainerdashboard.explainers.RandomForestExplainer.plot_trees

decisiontree
^^^^^^^^^^^^

.. automethod:: explainerdashboard.explainers.RandomForestExplainer.decisiontree

decisiontree_file
^^^^^^^^^^^^^^^^^

.. automethod:: explainerdashboard.explainers.RandomForestExplainer.decisiontree_file

decisiontree_encoded
^^^^^^^^^^^^^^^^^^^^

.. automethod:: explainerdashboard.explainers.RandomForestExplainer.decisiontree_encoded


Other explainer outputs
=======================

Base outputs
------------

Some other useful tables and outputs you can get out of the explainer::

    metrics()
    get_mean_abs_shap_df(topx=None, cutoff=None, cats=False, pos_label=None)
    get_permutation_importances_df(topx=None, cutoff=None, cats=False, pos_label=None)
    get_importances_df(kind="shap", topx=None, cutoff=None, cats=False, pos_label=None)
    get_contrib_df(index, cats=True, topx=None, cutoff=None, pos_label=None)
    get_contrib_summary_df(index, cats=True, topx=None, cutoff=None, round=2, pos_label=None)
    get_interactions_df(col, cats=False, topx=None, cutoff=None, pos_label=None)

metrics
^^^^^^^

.. automethod:: explainerdashboard.explainers.BaseExplainer.metrics

metrics_descriptions
^^^^^^^^^^^^^^^^^^^^

.. automethod:: explainerdashboard.explainers.ClassifierExplainer.metrics_descriptions
.. automethod:: explainerdashboard.explainers.RegressionExplainer.metrics_descriptions

get_mean_abs_shap_df
^^^^^^^^^^^^^^^^^^^^

.. automethod:: explainerdashboard.explainers.BaseExplainer.get_mean_abs_shap_df

get_permutation_importances_df
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: explainerdashboard.explainers.BaseExplainer.get_permutation_importances_df

get_importances_df
^^^^^^^^^^^^^^^^^^

.. automethod:: explainerdashboard.explainers.BaseExplainer.get_importances_df

get_contrib_df
^^^^^^^^^^^^^^

.. automethod:: explainerdashboard.explainers.BaseExplainer.get_contrib_df

get_contrib_summary_df
^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: explainerdashboard.explainers.BaseExplainer.get_contrib_summary_df

get_interactions_df
^^^^^^^^^^^^^^^^^^^

.. automethod:: explainerdashboard.explainers.BaseExplainer.get_interactions_df



Classifier outputs
------------------

For ``ClassifierExplainer`` in addition::

    random_index(y_values=None, return_str=False,pred_proba_min=None, pred_proba_max=None,
                    pred_percentile_min=None, pred_percentile_max=None, pos_label=None)
    prediction_result_df(index, pos_label=None)
    cutoff_from_percentile(percentile, pos_label=None)
    get_precision_df(bin_size=None, quantiles=None, multiclass=False, round=3, pos_label=None)
    get_liftcurve_df(pos_label=None)


random_index
^^^^^^^^^^^^

.. automethod:: explainerdashboard.explainers.ClassifierExplainer.random_index


cutoff_from_percentile
^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: explainerdashboard.explainers.ClassifierExplainer.cutoff_from_percentile

percentile_from_cutoff
^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: explainerdashboard.explainers.ClassifierExplainer.percentile_from_cutoff

get_precision_df
^^^^^^^^^^^^^^^^

.. automethod:: explainerdashboard.explainers.ClassifierExplainer.get_precision_df

get_liftcurve_df
^^^^^^^^^^^^^^^^

.. automethod:: explainerdashboard.explainers.ClassifierExplainer.get_liftcurve_df

get_classification_df
^^^^^^^^^^^^^^^^^^^^^

.. automethod:: explainerdashboard.explainers.ClassifierExplainer.get_classification_df

roc_auc_curve
^^^^^^^^^^^^^

.. automethod:: explainerdashboard.explainers.ClassifierExplainer.roc_auc_curve

pr_auc_curve
^^^^^^^^^^^^

.. automethod:: explainerdashboard.explainers.ClassifierExplainer.pr_auc_curve

confusion_matrix
^^^^^^^^^^^^^^^^

.. automethod:: explainerdashboard.explainers.ClassifierExplainer.confusion_matrix


Regression outputs
------------------


For ``RegressionExplainer``::

    random_index(y_min=None, y_max=None, pred_min=None, pred_max=None, 
                    residuals_min=None, residuals_max=None,
                    abs_residuals_min=None, abs_residuals_max=None,
                    return_str=False)


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


get_decisionpath_df
^^^^^^^^^^^^^^^^^^^

.. automethod:: explainerdashboard.explainers.RandomForestExplainer.get_decisionpath_df

get_decisionpath_summary_df
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: explainerdashboard.explainers.RandomForestExplainer.get_decisionpath_summary_df

decisiontree_file
^^^^^^^^^^^^^^^^^^

.. automethod:: explainerdashboard.explainers.RandomForestExplainer.decisiontree_file

decisiontree_encoded
^^^^^^^^^^^^^^^^^^^^^

.. automethod:: explainerdashboard.explainers.RandomForestExplainer.decisiontree_encoded

decisiontree
^^^^^^^^^^^^^

.. automethod:: explainerdashboard.explainers.RandomForestExplainer.decisiontree


Calculated Properties
=====================

In general ``Explainers`` don't calculate any properties of the model or the 
data until they are needed for an output, so-called lazy calculation. When the
property is calculated once, it is stored for next time. So the first time 
you invoke a plot involving shap values may take a while to calculate. The next
time will be basically instant. 

You can access these properties directly from the explainer, e.g. ``explainer.get_shap_values_df()``. 
For classifier models if you want values for a particular ``pos_label`` you can
pass this label ``explainer.get_shap_values_df(0)`` would get the shap values for 
the 0'th class label.

In order to calculate all properties of the explainer at once, you can call
``explainer.calculate_properties()``. (``ExplainerComponents`` have a similar method
``component.calculate_dependencies()`` to calculate all properties that that specific
component will need). 

The various properties are::

    explainer.preds
    explainer.pred_percentiles
    explainer.permutation_importances(pos_label)
    explainer.mean_abs_shap_df(pos_label)
    explainer.shap_base_value(pos_label)
    explainer.get_shap_values_df(pos_label)
    explainer.shap_interaction_values
    

For ``ClassifierExplainer``::

    explainer.y_binary
    explainer.pred_probas_raw
    explainer.pred_percentiles_raw
    explainer.pred_probas(pos_label)
    explainer.roc_auc_curve(pos_label)
    explainer.pr_auc_curve(pos_label)
    explainer.get_classification_df(cutoff, pos_label)
    explainer.get_liftcurve_df(pos_label)
    explainer.confusion_matrix(cutoff, binary, pos_label)

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
manually to a specific method, the global ``self.pos_label`` will be used. You can set
this directly on the explainer (even us str labels if you have set these)::

    explainer.pos_label = 0
    explainer.plot_dependence("Fare") # will show plot for pos_label=0
    explainer.pos_label = 'Survived' 
    explainer.plot_dependence("Fare") # will now show plot for pos_label=1
    explainer.plot_dependence("Fare", pos_label=0) # show plot for label 0, without changing explainer.pos_label

The ``ExplainerDashboard`` will show a dropdown menu in the header to choose
a particular ``pos_label``. Changing this will basically update every single
plot in the dashboard. 


BaseExplainer
=============

.. autoclass:: explainerdashboard.explainers.BaseExplainer
   :members: get_shap_values_df, get_mean_abs_shap_df, get_permutation_importances_df, 
            get_importances_df, contrib_df, set_shap_values, set_shap_interaction_values, plot_importances, plot_contributions, 
            plot_importances_detailed, plot_interactions_detailed, plot_interactions_importances, 
            plot_dependence, plot_interaction, plot_pdp
   :member-order: bysource

ClassifierExplainer
===================

For classification (e.g. ``RandomForestClassifier``) models you use ``ClassifierExplainer``.

You can pass an additional parameter to ``__init__()`` with a list of label names. For
multilabel classifier you can set the positive class with e.g. ``explainer.pos_label=1``.
This will make sure that for example ``explainer.pred_probas`` will return the probability
of that label. 

More examples in the `notebook on the github repo. <https://github.com/oegedijk/explainerdashboard/blob/master/notebooks/explainer_examples.ipynb>`_


.. autoclass:: explainerdashboard.explainers.ClassifierExplainer
   :members: random_index, get_precision_df, get_classification_df, get_liftcurve_df,
        set_shap_values, set_shap_interaction_values,
        plot_precision, plot_cumulative_precision, plot_classification, 
        plot_lift_curve, plot_confusion_matrix, plot_roc_auc, plot_pr_auc
   :member-order: bysource
   :noindex:


RegressionExplainer
===================

For regression models (e.g. ``RandomForestRegressor``) models you use ``RegressionExplainer``.

You can pass ``units`` as an additional parameter for the units of the target variable (e.g. ``units="$"``). 

More examples in the `notebook on the github repo. <https://github.com/oegedijk/explainerdashboard/blob/master/notebooks/explainer_examples.ipynb>`_

.. autoclass:: explainerdashboard.explainers.RegressionExplainer
   :members: random_index, residuals, metrics, plot_predicted_vs_actual, 
                plot_residuals,  plot_residuals_vs_feature
   :member-order: bysource
   :noindex:






