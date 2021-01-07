ExplainerHub (beta)
*******************


If you are hosting multiple ``ExplainerDashboards`` it becomes convenient 
to host them at a single place. This is made easy with ``ExplainerHub``. 

Possible use cases are: 

1. Showcasing multiple models to compare and decide which 
   one to put into production
2. Keeping ExplainerDashboards up and running for all models in production 
   in a single place

You can initialize an ``ExplainerHub`` by passing in a list of ``ExplainerDashboards``::

    db1 = ExplainerDashboard(explainer1)
    db2 = ExplainerDashboard(explainer2)
    hub = ExplainerHub([db1, db2])
    hub.run()

Each dashboard is hosted on it's own url path (e.g. ``localhost:8050/dashboard1``),
and a front end dashboard with links and descriptions for every dashboard is hosted
at e.g. ``localhost:8050``:

.. image:: screenshots/explainerhub.png


Adjusting title and descriptions
================================

You can adjust the title of the ``ExplainerHub`` and the description in the
jumbotron, by passing ``title`` and ``description``. You can also adjust
the ``title`` and ``description`` of each ``ExplainerDashboard``, and set the
url path with ``name``::

    db1 = ExplainerDashboard(explainer1, title="Model One", name="db1", 
                description="This is model option one")
    db2 = ExplainerDashboard(explainer2, title="Model Two", name="db2", 
                description="This is model option two")
    hub = ExplainerHub([db1, db2], title="Model Comparison", 
                description="Showing dashboards for both model one and two")
    hub.run()

Changing size, theme, etc
==========================

By default the hub fills the entire width of the browser, you can make it more slim
by passing ``fluid=False``. You can also pass other bootstrap themes: 
``bootstrap=dbc.themes.SKETCHY``. You can adjust the size of the iFrame with e.g. 
``min_height=2000``.

You can also build your own front end if you want. If you pass ``no_index=True``,
the index page and navbars will not get loaded, while the dashboards are still
loaded on their respective routes. E.g.::

    hub = ExplainerHub([db1, db2], no_index=True)
    app = hub.flask_server()

    @app.route("/")
    def custom_index():
        return render_template("custom_index.html")


Managing users
==============

You can manage logins and which usernames have access to particular dashboards
by passing ``logins`` and ``db_users``. Here we create two users (``user1`` 
and ``user2``) and only give ``user1`` access to the first dashboard, 
and only give ``user2`` access to the second dashboard::

    hub = ExplainerHub([db1, db2], 
            logins=[['user1', 'password1'],['user2','password2']],
            db_users=dict(db1=['user1'], db2=['user2'])

If you had defined users in your ``ExplainerDashboard`` then these get automatically
carried over to the`` ExplainerHub``::

    db1 = ExplainerDashboard(explainer1, logins=[['user1', 'password1']])
    db2 = ExplainerDashboard(explainer2, logins=[['user2', 'password2']])
    hub = ExplainerHub([db1, db2])

You can also add users from the hub itself::

    hub.add_user("user3", "password3")
    hub.add_user_to_dashboard("db2", "user3")

Or by adding ``users.yaml`` or by using the ``explainerhub`` CLI tool (see below).


Storing to config
=================

You can store an ``ExplainerHub`` to disk with ``ExplainerHub.to_yaml()``. This
will also dump all the explainers to disk and store the configuration of dashboards
that make up the hub to individual .yaml files. You reload a hub with the ``from_config`` 
classmethod::

    hub.to_yaml("hub.yaml")

    hub2 = ExplainerHub.from_config("hub.yaml")

The `hub.yaml` file looks something like this::

    explainerhub:
        title: ExplainerHub
        description: Showing dashboards for both model one and two
        masonry: false
        n_dashboard_cols: 3
        users_file: users.yaml
        db_users: null
        port: 8050
        kwargs: {}
        dashboards:
        - db1_dashboard.yaml
        - db2_dashboard.yaml

If you pass ``integrate_dashboard_yamls=True``, then the configuration of the 
dashboards get integrated into a single ``hub.yaml`` file instead of being
stored in separate files. 

When you store the hub, all users and (hashed) passwords get stored in a 
``users.yaml`` file. 

explainerhub CLI
================

.. highlight:: bash

You can also use the ``explainerhub`` CLI tool to start your ``ExplainerHub`` 
and manage your users straight from the commandline::

    $ explainerhub run hub.yaml
    $ explainerhub add-user
    $ explainerhub delete-user
    $ explainerhub add-dashboard-user
    $ explainerhub delete-dashboard-user

SECRET_KEY
==========

.. highlight:: bash
In order to make the logins persist when you reboot the server, you need to
pass a ``SECRET_KEY`` to the hub. Like with any Flask app you should be very
careful not to store this key somewhere easily findable. Ususally people store
it as an environmental variable::

    $ export SECRET_KEY='5f352379324c22463451387a0aec5d2f'

.. highlight:: python

Then you load it with the `os` module and pass it to the hub::

    ExplainerHub([db1, db2], secret_key=os.environ.get("SECRET_KEY"))

If you do not pass a secret key, a random uuid key is generated each time
you initialize the hub (which means you'll have to log in every time).


.. autoclass:: explainerdashboard.dashboards.ExplainerHub
   :members: to_yaml, from_config, add_user, add_user_to_dashboard, get_dashboard_users, flask_server, run