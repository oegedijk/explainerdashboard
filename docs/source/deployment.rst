Deployment
**********

When deploying your dashboard it is better not to use the built-in flask
server but use more robust and scalable options like ``gunicorn`` and ``nginx``.
Install gunicorn through pip with ``pip install gunicorn``.

Storing explainer and running default dashboard with gunicorn
=============================================================

Before you start a dashboard with gunicorn you should first store your
explainer to disk with all properties calculated. You can do this by
either wrapping the explainer in a dashboard (which will calculate all properties
needed for that particular dashboard) or simply calculate all properties 
with ``explainer.calculate_properties()``::

    explainer = ClassifierExplainer(model, X, y)
    # calculate properties needed for this dashboard:
    db = ExplainerDashboard(explainer)
    # alternatively: 
    # explainer.calculate_properties()
    explainer.dump("explainer.joblib")

Now you define your dashboard in a file, e.g. ``dashboard.py``. You first
load the explainer from file, then start the dashboard, and expose the flask
server as ``app``::

    from explainerdashboard import ClassifierExplainer, ExplainerDashboard

    explainer = ClassifierExplainer.from_file("explainer.joblib")
    db = ExplainerDashboard(explainer)

    # need to define app so that gunicorn can find the flask server
    app = db.flask_server()


.. highlight:: bash

If you named the file above ``dashboard.py``, you can now start the gunicorn server with::

    $ gunicorn dashboard:app

If you want to run the server server with for example three workers, binding to port 8050,
and making sure all the workers are preloaded before starting the dashboard your 
launch gunicorn like::

    $ gunicorn -w 3 --preload -b localhost:8050 dashboard:app

If you now point your browser to ``http://localhost:8050`` you should see your dashboard. 
Next step is finding a nice url in your organization's domain, and forwarding it 
to your dashboard server.

.. highlight:: python

Storing custom dashboard config and running with gunicorn
=========================================================

If you have some custom settings on your ``ExplainerDashboard`` that you would like
to preserve, you need to export the settings to ``.yaml`` first. So for example if you build
a dashboard with two specific tabs and a particular title, you would store the 
explainer and dashboard settings like this::

    from explainerdashboard import ClassifierExplainer, ExplainerDashboard
    from explainerdashboard.dashboard_tabs import ShapDependenceTab, ImportancesTab

    explainer = ClassifierExplainer(model, X, y, labels=['Not Survived', 'Survived'])
    explainer.dump("explainer.joblib")

    db = ExplainerDashboard(explainer, [ShapDependenceTab, ImportancesTab], title="Custom Title")
    db.to_yaml("dashboard.yaml", explainerfile="explainer.joblib")

And then start the ExpalinerDashboard directly from the config file in ``dashboard.py``::

    from explainerdashboard import ExplainerDashboard

    db = ExplainerDashboard.from_config("dashboard.yaml")
    app = db.flask_server()

.. highlight:: bash

And start the server the same as before::

    $ gunicorn -w 3 --preload -b localhost:8050 dashboard:app

Automatically restart gunicorn server upon changes
==================================================

We can use the ``explainerdashboard`` CLI tool to automatically rebuild our
explainer whenever there is a change to the underlying
model, dataset or explainer configuration. And we we can use ``kill -HUP gunicorn.pid`` 
to force the gunicorn to restart and reload whenever a new ``explainer.joblib`` 
is generated or the dashboard configuration ``dashboard.yaml`` changes. These two 
processes together ensure that the dashboard automatically updates whenever there 
are underlying changes.

First we store the explainer config in ``explainer.yaml`` and the dashboard 
config in ``dashboard.yaml``. We also indicate which modelfiles and datafiles the
explainer depends on, and which columns in the datafile should be used as 
a target and which as index::

    explainer = ClassifierExplainer(model, X, y, labels=['Not Survived', 'Survived'])
    explainer.dump("explainer.joblib")
    explainer.to_yaml("explainer.yaml", 
                    modelfile="model.pkl",
                    datafile="data.csv",
                    index_col="Name",
                    target_col="Survival",
                    explainerfile="explainer.joblib",
                    dashboard_yaml="dashboard.yaml")

    db = ExplainerDashboard(explainer, [ShapDependenceTab, ImportancesTab], title="Custom Title")
    db.to_yaml("dashboard.yaml", explainerfile="explainer.joblib")

The ``dashboard.py`` is the same as before and simply loads an ``ExplainerDashboard``
directly from the config file::

    from explainerdashboard import ExplainerDashboard

    db = ExplainerDashboard.from_config("dashboard.yaml")
    app = db.flask_server()  

.. highlight:: bash

Now we would like to rebuild the ``explainer.joblib`` file whenever there is a 
change to ``model.pkl``, ``data.csv`` or ``explainer.yaml`` by running 
``explainerdashboard build``. And we restart the ``gunicorn`` server whenever 
there is a change in ``explainer.joblib`` or ``dashboard.yaml`` by killing 
the gunicorn server with ``kill -HUP pid`` To do that we need to install 
the python package ``watchdog`` (``pip install watchdog[watchmedo]``). This 
package can keep track of filechanges and execute shell-scripts upon file changes.

So we can start the gunicorn server and the two watchdog filechange trackers
from a shell script ``start_server.sh``::

    trap "kill 0" EXIT  # ensures that all three process are killed upon exit

    source venv/bin/activate # activate virtual environment first

    gunicorn --pid gunicorn.pid gunicorn_dashboard:app &
    watchmedo shell-command  -p "./model.pkl;./data.csv;./explainer.yaml" -c "explainerdashboard build explainer.yaml" &
    watchmedo shell-command -p "./explainer.joblib;./dashboard.yaml" -c 'kill -HUP $(cat gunicorn.pid)' &

    wait # wait till user hits ctrl-c to exit and kill all three processes

Now we can simply run ``chmod +x start_server.sh`` and ``./start_server.sh`` to 
get our server up and running.

Whenever we now make a change to either one of the source files 
(``model.pkl``, ``data.csv`` or ``explainer.yaml``), this produces a fresh
``explainer.joblib``. And whenever there is a change to either ``explainer.joblib``
or ``dashboard.yaml`` gunicorns restarts and rebuild the dashboard. 

So you can keep an explainerdashboard running without interuption and simply 
 an updated ``model.pkl`` or a fresh dataset ``data.csv`` into the directory and 
the dashboard will automatically update. 

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

    gunicorn --preload -b localhost:8050 dashboard:app

And you can visit the dashboard on ``http://localhost:8050/dashboard``.


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

If you want to visualize individual trees in your ``RandomForest`` using
the ``dtreeviz`` package you will
need to make sure that ``graphviz`` is installed on your ``heroku`` dyno by
adding the following buildstack: 
``https://github.com/weibeld/heroku-buildpack-graphviz.git``


Setting logins and password
===========================

``explainerdashboard`` supports `dash basic auth functionality <https://dash.plotly.com/authentication>`_.

You can simply add a list of logins to the ExplainerDashboard to force a login 
and prevent random users from accessing the details of your model dashboard::

    ExplainerDashboard(explainer, logins=[['login1', 'password1'], ['login2', 'password2']]).run()

Make sure not to check these login/password pairs into version control though, 
but store them somewhere safe! 
