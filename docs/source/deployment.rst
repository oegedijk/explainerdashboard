Deployment
**********

When deploying your dashboard it is better not to use the built-in flask
development server but use a more robust production server like ``gunicorn`` or ``waitress``.
Probably `gunicorn <https://gunicorn.org/>`_ is a bit more fully featured and 
faster but only works on unix/linux/osx, whereas
`waitress <https://docs.pylonsproject.org/projects/waitress/en/stable/>`_ also works 
on Windows and has very minimal dependencies. 

Install with either ``pip install gunicorn`` or ``pip install waitress``. 

Storing explainer and running default dashboard with gunicorn
=============================================================

Before you start a dashboard with gunicorn you need to store both the explainer 
instance and and a configuration for the dashboard::

    from explainerdashboard import ClassifierExplainer, ExplainerDashboard

    explainer = ClassifierExplainer(model, X, y)
    db = ExplainerDashboard(explainer, title="Cool Title", shap_interaction=False) 
    db.to_yaml("dashboard.yaml", explainerfile="explainer.joblib", dump_explainer=True)

Now you re-load your dashboard and expose a flask server as ``app`` in ``dashboard.py``::

    from explainerdashboard import ExplainerDashboard

    db = ExplainerDashboard.from_config("dashboard.yaml")
    app = db.flask_server() 


.. highlight:: bash

If you named the file above ``dashboard.py``, you can now start the gunicorn server with::

    $ gunicorn dashboard:app

If you want to run the server server with for example three workers, binding to 
port ``8050`` you launch gunicorn with::

    $ gunicorn -w 3 -b localhost:8050 dashboard:app

If you now point your browser to ``http://localhost:8050`` you should see your dashboard. 
Next step is finding a nice url in your organization's domain, and forwarding it 
to your dashboard server.

With waitress you would call::

    $ waitress-serve --port=8050 dashboard:app

.. highlight:: python

Although you can all use the ``waitress`` directly from the dashboard by passing
the ``use_waitress=True`` flag to ``.run()``::

    ExplainerDashboard(explainer).run(use_waitress=True)


Deploying dashboard as part of Flask app on specific route
==========================================================

.. highlight:: python

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


.. highlight:: bash 

Now you can start the dashboard by::

    $ gunicorn -b localhost:8050 dashboard:app

And you can visit the dashboard on ``http://localhost:8050/dashboard``.


Deploying to heroku
===================

In case you would like to deploy to `heroku <www.heroku.com>`_ (which is normally
the simplest option for dash apps, see 
`dash instructions here <https://dash.plotly.com/deployment>`_). The demonstration 
dashboard is also hosted on heroku at `titanicexplainer.herokuapp.com <http://titanicexplainer.herokuapp.com>`_.

In order to deploy the heroku there are a few things to keep in mind. First of 
all you need to add ``explainerdashboard`` and ``gunicorn`` to 
``requirements.txt`` (pinning is recommended to force a new build of your environment
whenever you upgrade versions)::

    explainerdashboard==0.3.1
    gunicorn

Select a python runtime compatible with the version that you used to pickle
your explainer in ``runtime.txt``::

    python-3.8.6

(supported versions as of this writing are ``python-3.9.0``, ``python-3.8.6``, 
``python-3.7.9`` and ``python-3.6.12``, but check the 
`heroku documentation <https://devcenter.heroku.com/articles/python-support#supported-runtimes>`_
for the latest)


And you need to tell heroku how to start your server in ``Procfile``::

    web: gunicorn dashboard:app


Graphviz buildpack
------------------

If you want to visualize individual trees inside your ``RandomForest`` or ``xgboost`` 
model using the ``dtreeviz`` package you will
need to make sure that ``graphviz`` is installed on your ``heroku`` dyno by
adding the following buildstack (as well as the ``python`` buildpack): 
``https://github.com/weibeld/heroku-buildpack-graphviz.git``

(you can add buildpacks through the "settings" page of your heroku project)

Docker deployment
=================
.. highlight:: python

You can also deploy a dashboard using docker. You can build the dashboard and store
it inside the container to make sure it is compatible with the container environment.
E.g. **generate_dashboard.py**::

    from sklearn.ensemble import RandomForestClassifier

    from explainerdashboard import *
    from explainerdashboard.datasets import *

    X_train, y_train, X_test, y_test = titanic_survive()
    model = RandomForestClassifier(n_estimators=50, max_depth=5).fit(X_train, y_train)

    explainer = ClassifierExplainer(model, X_test, y_test, 
                                    cats=["Sex", 'Deck', 'Embarked'],
                                    labels=['Not Survived', 'Survived'],
                                    descriptions=feature_descriptions)

    db = ExplainerDashboard(explainer)
    db.to_yaml("dashboard.yaml", explainerfile="explainer.joblib", dump_explainer=True)

**run_dashboard.py**::

    from explainerdashboard import ExplainerDashboard

    db = ExplainerDashboard.from_config("dashboard.yaml")
    db.run(host='0.0.0.0', port=9050, use_waitress=True)

.. highlight:: docker

**Dockerfile**::

    FROM python:3.8

    RUN pip install explainerdashboard

    COPY generate_dashboard.py ./
    COPY run_dashboard.py ./

    RUN python generate_dashboard.py

    EXPOSE 9050
    CMD ["python", "./run_dashboard.py"]

.. highlight:: bash

And build and run the container exposing port ``9050``::

    $ docker build -t explainerdashboard .
    $ docker run -p 9050:9050 explainerdashboard

Reducing memory usage
=====================

If you deploy the dashboard with a large dataset with a large number of rows (``n``)
and a large number of columns (``m``),
it can use up quite a bit of memory: the dataset itself, shap values, 
shap interaction values and any other calculated properties are alle kept in
memory in order to make the dashboard responsive. You can check the (approximate)
memory usage with ``explainer.memory_usage()``. In order to reduce the memory
footprint there are a number of things you can do:

1. Not including shap interaction tab.
    Shap interaction values are shape ``n*m*m``, so can take a subtantial amount 
    of memory, especially if you have a significant amount of columns ``m``. 
2. Setting a lower precision. 
    By default shap values are stored as ``'float64'``,
    but you can store them as ``'float32'`` instead and save half the space:
    ```ClassifierExplainer(model, X_test, y_test, precision='float32')```. You 
    can also set a lower precision on your ``X_test`` dataset yourself ofcourse.
3. Drop non-positive class shap values.
    For multi class classifiers, by default ``ClassifierExplainer`` calculates
    shap values for all classes. If you are only interested in a single class
    you can drop the other shap values with ``explainer.keep_shap_pos_label_only(pos_label)``
4. Storing row data externally and loading on the fly. 
    You can for example only store a subset of ``10.000`` rows in
    the ``explainer`` itself (enough to generate representative importance and dependence plots),
    and store the rest of your millions of rows of input data in an external file 
    or database that get loaded one by one with the following functions:

    - with ``explainer.set_X_row_func()`` you can set a function that takes 
      an `index` as argument and returns a single row dataframe with model
      compatible input data for that index. This function can include a query
      to a database or fileread. 
    - with ``explainer.set_y_func()`` you can set a function that takes 
      and `index` as argument and returns the observed outcome ``y`` for
      that index.
    - with ``explainer.set_index_list_func()`` you can set a function 
      that returns a list of available indexes that can be queried.
    
    If the number of indexes is too long to fit in a dropdown you can pass 
    ``index_dropdown=False`` which turns the dropdowns into free text fields.
    Instead of an ``index_list_func`` you can also set an 
    ``explainer.set_index_check_func(func)`` which should return a bool whether
    the ``index`` exists or not. 

    Important: these function can be called multiple times by multiple independent
    components, so probably best to implement some kind of caching functionality.
    The functions you pass can be also methods, so you have access to all of the
    internals of the explainer.


Setting logins and password
===========================

``ExplainerDashboard`` supports `dash basic auth functionality <https://dash.plotly.com/authentication>`_. 
``ExplainerHub`` uses ``flask_simple_login`` for its user authentication.

You can simply add a list of logins to the ``ExplainerDashboard`` to force a login 
and prevent random users from accessing the details of your model dashboard::

    ExplainerDashboard(explainer, logins=[['login1', 'password1'], ['login2', 'password2']]).run()

Whereas :ref:`ExplainerHub<ExplainerHub>` has somewhat more intricate user management 
using ``FlaskLogin``, but the basic syntax is the same. See the 
:ref:`ExplainerHub documetation<ExplainerHub>` for more details::

    hub = ExplainerHub([db1, db2], logins=[['login1', 'password1'], ['login2', 'password2']])

Make sure not to check these login/password pairs into version control though, 
but store them somewhere safe! ``ExplainerHub`` stores passwords into a hashed 
format by default.


Automatically restart gunicorn server upon changes
==================================================

We can use the ``explainerdashboard`` CLI tools to automatically rebuild our
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
    app = db.app.server

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



