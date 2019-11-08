
explainerdashboard
******************

Summary
=======

**explainerdashboard** is a library for quickly analyzing and explaining the performance
of a (scikit-learn compatible) machine learning models. 

It combines shap values, permutation importances, partial dependence plots,
and the visualisation of individual trees of random forests into a single package.

You can easily construct an ExplainerBunch object that computes all relevant
statistics behind the scenes and allows you to quickly plot feature importances,
shap dependence plots, pdp plots, etc.

Example
=======

You can then load this ExplainerBunch object into an ExplainerDashboard that
allows for the interactive investigation of the model. Some example code::

    from explainerdashboard.explainers import RandomForestClassifierBunch
    from explainerdashboard.dashboards import RandomForestDashboard
    from explainerdashboard.datasets import titanic_survive, titanic_names

    X_train, y_train, X_test, y_test = titanic_survive()
    train_names, test_names = titanic_names()

    model = RandomForestClassifier(n_estimators=50, max_depth=5)
    model.fit(X_train, y_train)

    explainer = RandomForestClassifierBunch(
                    model, X_test, y_test, roc_auc_score, 
                    cats=['Sex', 'Deck', 'Embarked'],
                    idxs=test_names, 
                    labels=['Not survived', 'Survived'])

    db = RandomForestDashboard(explainer,
                            model_summary=True,
                            contributions=True,
                            shap_dependence=True,
                            shap_interaction=True,
                            shadow_trees=True)
    db.run(port=8050)

The result can be viewed on `this dashboard deployed to heroku <titanicexplainer.herokuapp.com>`_ 



.. toctree::
   :maxdepth: 3

   explainers
   dashboards
   explainer_methods
   explainer_plots
   license
   help 


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
