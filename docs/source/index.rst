explainerdashboard
******************

Summary
=======

``explainerdashboard`` is a library for quickly building interactive dashboards
and interactive notebook components for analyzing and explaining the performance
of (scikit-learn compatible) machine learning models.

It allows you to investigate shap values, permutation importances, 
interaction effects, partial dependence plots, all kinds of performance plots,
and even individual trees in a random forest by deploying an interactive 
dashboard with just two lines of code. 

.. image:: screenshot.png

You first construct an ``explainer`` object out of your model and the test data. 
The ``explainer`` offers an interface that computes all relevent metrics behind 
the scenes and allows you to quickly plot feature importances,
shap dependence plots, pdp plots, etc, etc.

You then pass this ``explainer`` object to an ``ExplainerDashboard`` to start an 
interactive analytical web app to inspect the workings and performance of your model.
For a custom dashboard you can make use of the ``ExplainerComponents`` primitives to 
easily make your a dashboard interface to your own liking. For viewing 
individual components or tabs directly inside your jupyter notebook you use the 
``InlineExplainer``.

With ``explainerdashboard`` any datascientist can create an interactive web app
to display the workings of their ML model in minutes, without having to know
anything about web development or deployment. In addition it aids the model
development flow by offering interactive inline notebook components without
having to even launch a dashboard. 

Supports all scikit-learn compatible models (that are compatible with the shap library), 
including XGBoost, LightGBM, CatBoost, LinearRegression, LogisticRegression 
and RandomForests.

An Example
==========

Some example code, where we load some data, fit a model, construct an explainer, 
pass it on to an ``ExplainerDashboard`` and run the dashboard::

    from sklearn.ensemble import RandomForestClassifier

    from explainerdashboard.explainers import RandomForestClassifierExplainer
    from explainerdashboard.dashboards import ExplainerDashboard
    from explainerdashboard.datasets import titanic_survive, titanic_names

    X_train, y_train, X_test, y_test = titanic_survive()
    train_names, test_names = titanic_names()

    model = RandomForestClassifier(n_estimators=50, max_depth=5)
    model.fit(X_train, y_train)

    explainer = ClassifierExplainer(
                    model, X_test, y_test, 
                    cats=['Sex', 'Deck', 'Embarked'],
                    idxs=test_names,
                    labels=['Not survived', 'Survived'])

    ExplainerDashboard(explainer).run()

Or, as a one-liner::

    ExplainerDashboard(
        ClassifierExplainer(
            RandomForestClassifier().fit(X_train, y_train), 
            X_test, y_test
        )
    ).run()

The result of the lines above can be viewed on `this dashboard deployed to heroku. <http://titanicexplainer.herokuapp.com>`_


More examples of how to start dashboards for different types of models and with 
different parameters can be found in the `dashboard_examples notebook <https://github.com/oegedijk/explainerdashboard/blob/master/dashboard_examples.ipynb>`_ 
in the github repo.

For examples on how to interact with and get plots and dataframes out of the explainer
object check out `explainer_examples notebook  <https://github.com/oegedijk/explainerdashboard/blob/master/explainer_examples.ipynb>`_
in the github repo.


.. toctree::
   :maxdepth: 3

   explainers
   plots
   dashboards
   tabs
   inline
   components
   custom
   deployment
   license
   help


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
