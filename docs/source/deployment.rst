Deployment
**********

When deploying your dashboard it is better not to use the built-in flask
server but use more robust and scalable options like ``gunicorn`` and ``nginx``.

Deploying a single dashboard instance
=====================================

``Dash`` is built on top of ``Flask``, and so the dashboard instance 
contains a Flask server. You can simply expose this server to host your dashboard.

The server can be found in ``ExplainerDashboard().app.server`` or with
the methods ``ExplainerDashboard.flask_server()``.

The code below is from `the deployed example to heroku <https://github.com/oegedijk/explainingtitanic/blob/master/dashboard.py>`_::

    from sklearn.ensemble import RandomForestClassifier

    from explainerdashboard import ClassifierExplainer, ExplainerDashboard
    from explainerdashboard.datasets import titanic_survive, titanic_names

    X_train, y_train, X_test, y_test = titanic_survive()
    train_names, test_names = titanic_names()

    model = RandomForestClassifier(n_estimators=50, max_depth=5)
    model.fit(X_train, y_train)

    explainer = ClassifierExplainer(model, X_test, y_test, 
                                cats=['Sex', 'Deck', 'Embarked'],
                                idxs=test_names, 
                                labels=['Not survived', 'Survived'])

    db = ExplainerDashboard(explainer)

    server = db.app.server

If you name the file above ``dashboard.py``, then you can start the gunicorn
server with for example three workers and binding to port 8050 like this::

    gunicorn localhost:8050 dashboard:server


So here ``dashboard`` refers to ``dashboard.py`` and ``server`` refers to the ``server``
defined equal to ``db.app.server``.

If you want to have multiple workers to speed up your dashboard, you need
to preload the app before starting::

        gunicorn -w 3 --preload localhost:8050 dashboard:server


Deploying dashboard as part of Flask app on specific route
==========================================================

Another way to deploy the dashboard is to first start a ``Flask`` app, and then
use this app as the backend of the Dashboard, and host the dashboard on a specific
route. This way you can for example host multiple dashboard under different urls.
You need to pass the Flask ``server`` instance and the ``url_base_pathname`` to the
``ExplainerDashboard`` constructor, and then the dashboard itself can be found
under ``db.app.index``::

    from flask import Flask
    
    app = Flask(__name__)

    [...]
    
    db = ExplainerDashboard(explainer, server=app, url_base_pathname="/dashboard/")

    @app.route('/dashboard')
    def return_dashboard():
        return db.app.index()

Now you can start the dashboard by::

    gunicorn -w 3 --preload -b localhost:8050 dashboard:app

And you can visit the dashboard on ``http://localhost:8050/dashboard``.

Avoid timeout by precalculating explainers and loading with joblib
==================================================================

Some of the calculations in order to generate e.g. the SHAP values and permutation
importances can take quite a long time (especially shap interaction values). 
Long enough the break the startup timeout of ``gunicorn``. Therefore it is better
to first calculate all these values, save the explainer to disk, and then load
the explainer when starting the dashboard::

    import joblib
    from explainerdashboard import ClassifierExplainer
    
    explainer = ClassifierExplainer(model, X_test, y_test, 
                               cats=['Sex', 'Deck', 'Embarked'],
                               labels=['Not survived', 'Survived'])
    explainer.calculate_properties()
    joblib.dump(explainer, "explainer.pkl")

Then in ``dashboard.py`` load the explainer and start the dashboard:: 

    import joblib
    from explainerdashboard import ExplainerDashboard

    explainer = joblib.load("explainer.pkl")
    db = ExplainerDashboard(clas_explainer)
    server = db.app.server 

And start the thing with gunicorn::

    gunicorn -b localhost:8050 dashboard:server


Deploying to heroku
===================

In case you would like to deploy to `heroku <www.heroku.com>`_ (which is probably the simplest 
`deployment <https://dash.plotly.com/deployment>`_ option for dash apps), 
where the demonstration dashboard is hosted
at `titanicexplainer.herokuapp.com <titanicexplainer.herokuapp.com>`_ 
there are a number of issues to keep in mind.

Uninstalling and mocking xgboost
--------------------------------

A heroku deployment ("slug size") should not exeed 500MB after compression. Unfortunately
the ``xgboost`` library is >350MB, so this means it will be hard to deploy any
``xgboost`` models to heroku. Unfortunately however  ``xgboost`` gets automatically installed 
as a dependency of ``dtreeviz`` which is a dependency of ``explainerdashboard``. 

So in order to get even non-xgboost models to work you will
have to uninstall ``xgboost`` and then mock it. This is normally pretty easy 
(``pip uninstall xgboost``), but on heroku you first need to add a buildpack
in order to run shell instructions after the build phase.
So add the following shell buildpack:
`https://github.com/niteoweb/heroku-buildpack-shell.git <https://github.com/niteoweb/heroku-buildpack-shell.git>`_ ,
and then create a 
directory ``.heroku`` with a file ``run.sh`` with the
instructions to uninstall xgboost: ``pip install -y xgboost``. This script will
then be run at the end of your build process, ensuring that xgboost will be
uninstalled before the deployment is compressed to a slug.

However ``dtreeviz`` will still try to import ``xgboost`` so you need to 
mock the ``xgboost`` library by adding the following code before you import 
``explainerdashboard`` in your project::

    from unittest.mock import MagicMock
    import sys
    sys.modules["xgboost"] = MagicMock()


Graphviz buildpack
------------------

If you want to visualize indidividual trees in your ``RandomForest`` using
the ``dtreeviz`` package you will
need to make sure that ``graphviz`` is installed on your ``heroku`` dyno by
adding the following buildstack: 
``https://github.com/weibeld/heroku-buildpack-graphviz.git``


Setting logins and password
===========================

``explainerdashboard`` supports `dash basic auth functionality <https://dash.plotly.com/authentication>`_.

You can simply add a list of logins to the ExplainerDashboard to force a logins 
and prevent random users from accessing the details of your model dashboard::

    ExplainerDashboard(explainer, logins=[['login1', 'password1'], ['login2', 'password2']]).run()

Make sure not to check these login/password pairs into version control though, 
but store them somewhere safe! 
