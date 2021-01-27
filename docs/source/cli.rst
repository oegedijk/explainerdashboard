explainerdashboard CLI
**********************

The library comes with a ``explainerdashboard`` command line tool (CLI) that
you can use to build and run explainerdashboards from your terminal. 
This makes it easy to start a dashboard without having to run python code
or start a notebook first. You can also use it to build explainer objects 
as part of a CI/CD flow.

Run dashboard from stored explainer
===================================

In order to run a dashboard from a stored explainer from the commandline, \
we first need to store an explainer to disk. You can do this with::

    explainer = ClassifierExplainer(model, X, y)
    explainer.dump("explainer.joblib")

.. highlight:: bash

And then you can run the default dashboard and launch a browser tab 
from the command line by running::

    $ explainerdashboard run explainer.joblib

The CLI uses the ``waitress`` web server by default to run your dashboard.
To run on a specific port, not launch a browser or show help::

    $ explainerdashboard run explainer.joblib --port 8051
    $ explainerdashboard run explainer.joblib --no-browser
    $ explainerdashboard run --help


Run custom dashboard from dashboard.yaml
========================================

.. highlight:: python

If you'd like to launch a custom dashboard with custom tabs and parameters,
you can do so by storing the configuration to `.yaml`::

    db = ExplainerDashboard(explainer, [ShapDependenceTab, "importances"],
            port=9000, title="Custom Dashboard", header_hide_title=True)
    db.to_yaml("dashboard.yaml", explainerfile="explainer.joblib")

.. highlight:: bash

You can edit ``dashboard.yaml`` directly to make further configuration
changes if you wish. Start the dashboard from the commandline with::

    $ explainerdashboard run dashboard.yaml

.. highlight:: python


Building explainers from explainer.yaml
=======================================

You can build also explainers from the commandline by storing the model (e.g. ``model.pkl``)
and datafile (e.g. ``data.csv``), indicating which column is ``y`` (e.g. ``'Survival'``),
and which is the index (e.g. ``'Name'``), along with the other parameters 
of the explainer. 

You can get this configuration by storing the configuration as before::

    explainer = ClassifierExplainer(model, X, y, 
                    labels=['Not survived', 'Survived'])
    pickle.dump(model, open("model.pkl", "wb))

    explainer.to_yaml("explainer.yaml", 
                explainerfile="explainer.joblib",
                modelfile="model.pkl",
                datafile="data.csv",
                target_col="Survival",
                index_col="Name",
                dashboard_yaml="dashboard.yaml")

.. highlight:: bash

You can then build the ``explainer.joblib`` file by running::

    $ explainerdashboard build explainer.yaml

This will load the model and dataset, construct an explainer, construct the
custom dashboard, calculate all properties needed for that specific dashboard, 
and store the explainer to disk. This can be useful when you for example 
would like to populate the dashboard with a new set of data: you can simply
update ``data.csv`` and run ``explainerdashboard build``. To start the dashboard 
you can then run::

    $ explainerdashboard run dashboard.yaml

To build the explainer for a specific dashboard (other than the one 
specified in ``dashboard_yaml``, pass it as a second argument::

    $ explainerdashboard build explainer.yaml dashboard.yaml


.. note:: 
    If you use the default naming scheme of ``explainer.joblib``, ``dashboard.yaml``
    and ``explainer.yaml``, you can omit these arguments and simply run e.g.::

        $ explainerdashboard build
        $ explainerdashboard run

.. highlight:: python


dump, from_file, to_yaml
========================

Explainer.dump()
----------------

.. automethod:: explainerdashboard.explainers.BaseExplainer.dump

Explainer.from_file()
---------------------

.. automethod:: explainerdashboard.explainers.BaseExplainer.from_file

Explainer.to_yaml()
-------------------

.. automethod:: explainerdashboard.explainers.BaseExplainer.to_yaml

ExplainerDashboard.to_yaml()
----------------------------

.. automethod:: explainerdashboard.dashboards.ExplainerDashboard.to_yaml

ExplainerDashboard.from_config
------------------------------

.. automethod:: explainerdashboard.dashboards.ExplainerDashboard.from_config