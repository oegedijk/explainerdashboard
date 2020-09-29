

import dash

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from explainerdashboard.explainers import ClassifierExplainer, RegressionExplainer
from explainerdashboard.datasets import titanic_survive, titanic_fare, titanic_embarked, titanic_names
from explainerdashboard.dashboards import ExplainerDashboard


def get_classification_explainer():
    X_train, y_train, X_test, y_test = titanic_survive()
    train_names, test_names = titanic_names()
    model = XGBClassifier().fit(X_train, y_train)
    explainer = ClassifierExplainer(
                        model, X_test, y_test, 
                        cats=['Sex', 'Cabin', 'Embarked'],
                        labels=['Not survived', 'Survived'],
                        idxs=test_names)
    explainer.calculate_properties()
    return explainer


def get_regression_explainer():
    X_train, y_train, X_test, y_test = titanic_fare()
    train_names, test_names = titanic_names()
    model = RandomForestRegressor(n_estimators=50, max_depth=10).fit(X_train, y_train)
    reg_explainer = RegressionExplainer(model, X_test, y_test, 
                                    cats=['Sex', 'Deck', 'Embarked'], 
                                    idxs=test_names, 
                                    units="$")
    reg_explainer.calculate_properties()
    return reg_explainer


def get_multiclass_explainer():
    X_train, y_train, X_test, y_test = titanic_embarked()
    train_names, test_names = titanic_names()
    model = RandomForestClassifier(n_estimators=50, max_depth=10).fit(X_train, y_train)
    multi_explainer = ClassifierExplainer(model, X_test, y_test, 
                                    cats=['Sex', 'Deck'], 
                                    idxs=test_names,
                                    labels=['Queenstown', 'Southampton', 'Cherbourg'])
    multi_explainer.calculate_properties()
    return multi_explainer


def test_classification_dashboard(dash_duo):
    explainer = get_classification_explainer()
    db = ExplainerDashboard(explainer, title="testing")
    dash_duo.start_server(db.app)
    dash_duo.wait_for_text_to_equal("h1", "testing", timeout=30)
    assert dash_duo.get_logs() == [], "browser console should contain no error"


def test_regression_dashboard(dash_duo):
    explainer = get_regression_explainer()
    db = ExplainerDashboard(explainer, title="testing")
    dash_duo.start_server(db.app)
    dash_duo.wait_for_text_to_equal("h1", "testing", timeout=30)
    assert dash_duo.get_logs() == [], "browser console should contain no error"


def test_multiclass_dashboard(dash_duo):
    explainer = get_multiclass_explainer()
    db = ExplainerDashboard(explainer, title="testing")
    dash_duo.start_server(db.app)
    dash_duo.wait_for_text_to_equal("h1", "testing", timeout=30)
    assert dash_duo.get_logs() == [], "browser console should contain no error"