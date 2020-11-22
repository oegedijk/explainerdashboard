

import dash

from xgboost import XGBClassifier, XGBRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from explainerdashboard.explainers import ClassifierExplainer, RegressionExplainer
from explainerdashboard.datasets import titanic_survive, titanic_fare, titanic_embarked, titanic_names
from explainerdashboard.dashboards import ExplainerDashboard


def get_classification_explainer(xgboost=False, include_y=True):
    X_train, y_train, X_test, y_test = titanic_survive()
    train_names, test_names = titanic_names()
    if xgboost:
        model = XGBClassifier().fit(X_train, y_train)
    else:
        model = RandomForestClassifier(n_estimators=50, max_depth=10).fit(X_train, y_train)
    if include_y:
        explainer = ClassifierExplainer(
                            model, X_test, y_test, 
                            cats=['Sex', 'Cabin', 'Embarked'],
                            labels=['Not survived', 'Survived'])
    else:
        explainer = ClassifierExplainer(
                            model, X_test, 
                            cats=['Sex', 'Cabin', 'Embarked'],
                            labels=['Not survived', 'Survived'])

    explainer.calculate_properties()
    return explainer


def get_regression_explainer(xgboost=False, include_y=True):
    X_train, y_train, X_test, y_test = titanic_fare()
    train_names, test_names = titanic_names()
    if xgboost:
        model = XGBRegressor().fit(X_train, y_train)
    else:
        model = RandomForestRegressor(n_estimators=50, max_depth=10).fit(X_train, y_train)

    if include_y:
        reg_explainer = RegressionExplainer(model, X_test, y_test, 
                                        cats=['Sex', 'Deck', 'Embarked'], 
                                        idxs=test_names, 
                                        units="$")
    else:
        reg_explainer = RegressionExplainer(model, X_test, 
                                        cats=['Sex', 'Deck', 'Embarked'], 
                                        idxs=test_names, 
                                        units="$")

    reg_explainer.calculate_properties()
    return reg_explainer

def get_multiclass_explainer(xgboost=False, include_y=True):
    X_train, y_train, X_test, y_test = titanic_embarked()
    train_names, test_names = titanic_names()
    if xgboost:
        model = XGBClassifier().fit(X_train, y_train)
    else:
        model = RandomForestClassifier(n_estimators=50, max_depth=10).fit(X_train, y_train)

    if include_y:
        if xgboost:
            multi_explainer = ClassifierExplainer(model, X_test, y_test,
                                            model_output='logodds',
                                            cats=['Sex', 'Deck'], 
                                            labels=['Queenstown', 'Southampton', 'Cherbourg'])
        else:
            multi_explainer = ClassifierExplainer(model, X_test, y_test,
                                            cats=['Sex', 'Deck'], 
                                            labels=['Queenstown', 'Southampton', 'Cherbourg'])
    else:
        if xgboost:
            multi_explainer = ClassifierExplainer(model, X_test, 
                                            model_output='logodds',
                                            cats=['Sex', 'Deck'], 
                                            labels=['Queenstown', 'Southampton', 'Cherbourg'])
        else:
            multi_explainer = ClassifierExplainer(model, X_test, 
                                            cats=['Sex', 'Deck'], 
                                            labels=['Queenstown', 'Southampton', 'Cherbourg'])

    multi_explainer.calculate_properties()
    return multi_explainer


def test_classification_dashboard(dash_duo):
    explainer = get_classification_explainer()
    db = ExplainerDashboard(explainer, title="testing", responsive=False)
    dash_duo.start_server(db.app)
    dash_duo.wait_for_text_to_equal("h1", "testing", timeout=30)
    assert dash_duo.get_logs() == [], "browser console should contain no error"


def test_regression_dashboard(dash_duo):
    explainer = get_regression_explainer()
    db = ExplainerDashboard(explainer, title="testing", responsive=False)
    dash_duo.start_server(db.app)
    dash_duo.wait_for_text_to_equal("h1", "testing", timeout=30)
    assert dash_duo.get_logs() == [], "browser console should contain no error"


def test_multiclass_dashboard(dash_duo):
    explainer = get_multiclass_explainer()
    db = ExplainerDashboard(explainer, title="testing", responsive=False)
    dash_duo.start_server(db.app)
    dash_duo.wait_for_text_to_equal("h1", "testing", timeout=30)
    assert dash_duo.get_logs() == [], "browser console should contain no error"


def test_xgboost_classification_dashboard(dash_duo):
    explainer = get_classification_explainer(xgboost=True)
    db = ExplainerDashboard(explainer, title="testing", responsive=False)
    dash_duo.start_server(db.app)
    dash_duo.wait_for_text_to_equal("h1", "testing", timeout=30)
    assert dash_duo.get_logs() == [], "browser console should contain no error"


def test_xgboost_regression_dashboard(dash_duo):
    explainer = get_regression_explainer(xgboost=True)
    db = ExplainerDashboard(explainer, title="testing", responsive=False)
    dash_duo.start_server(db.app)
    dash_duo.wait_for_text_to_equal("h1", "testing", timeout=30)
    assert dash_duo.get_logs() == [], "browser console should contain no error"


def test_xgboost_multiclass_dashboard(dash_duo):
    explainer = get_multiclass_explainer(xgboost=True)
    db = ExplainerDashboard(explainer, title="testing", responsive=False)
    dash_duo.start_server(db.app)
    dash_duo.wait_for_text_to_equal("h1", "testing", timeout=30)
    assert dash_duo.get_logs() == [], "browser console should contain no error"


def test_classification_dashboard_no_y(dash_duo):
    explainer = get_classification_explainer(include_y=False)
    db = ExplainerDashboard(explainer, title="testing", responsive=False)
    dash_duo.start_server(db.app)
    dash_duo.wait_for_text_to_equal("h1", "testing", timeout=30)
    assert dash_duo.get_logs() == [], "browser console should contain no error"


def test_regression_dashboard_no_y(dash_duo):
    explainer = get_regression_explainer(include_y=False)
    db = ExplainerDashboard(explainer, title="testing", responsive=False)
    dash_duo.start_server(db.app)
    dash_duo.wait_for_text_to_equal("h1", "testing", timeout=30)
    assert dash_duo.get_logs() == [], "browser console should contain no error"


def test_multiclass_dashboard_no_y(dash_duo):
    explainer = get_multiclass_explainer(include_y=False)
    db = ExplainerDashboard(explainer, title="testing", responsive=False)
    dash_duo.start_server(db.app)
    dash_duo.wait_for_text_to_equal("h1", "testing", timeout=30)
    assert dash_duo.get_logs() == [], "browser console should contain no error"