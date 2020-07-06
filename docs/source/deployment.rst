Deployment
**********

When deploying your dashboard it is better not to use the built-in flask
server but use more robust and scalable options like ``gunicorn`` and ``nginx``.

In order to deploy the dashboard you need to expose the flask server inside
the ``ExplainerDashboard`` through your web service. This can be found
inside the ``app`` inside the ``ExplainerDashboard``.

The code below is from `the deployed example to heroku <https://github.com/oegedijk/explainingtitanic/blob/master/dashboard.py>`_::

    from sklearn.ensemble import RandomForestClassifier

    from explainerdashboard.explainers import *
    from explainerdashboard.dashboards import *
    from explainerdashboard.datasets import *

    print('loading data...')
    X_train, y_train, X_test, y_test = titanic_survive()
    train_names, test_names = titanic_names()

    print('fitting model...')
    model = RandomForestClassifier(n_estimators=50, max_depth=5)
    model.fit(X_train, y_train)

    print('building Explainer...')
    explainer = RandomForestClassifierExplainer(model, X_test, y_test, 
                                cats=['Sex', 'Deck', 'Embarked'],
                                idxs=test_names, 
                                labels=['Not survived', 'Survived'])

    print('Building ExplainerDashboard...')
    db = ExplainerDashboard(explainer)

    server = db.app.server

If you name the file above ``dashboard.py``, then you can start the gunicorn
server with for example three workers and binding to port 8050 like this::

    gunicorn -w 3 -b localhost:8050 dashboard:server


So here ``dashboard`` refers to ``dashboard.py`` and ``server`` refers to the ``server``
defined equal to ``db.app.server``.