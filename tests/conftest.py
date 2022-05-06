import pytest

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from explainerdashboard import RegressionExplainer, ClassifierExplainer, ExplainerDashboard
from explainerdashboard.datasets import titanic_survive, titanic_fare, titanic_names

@pytest.fixture()
def test_names():
    return titanic_names()[1]

@pytest.fixture(scope="session")
def testlen():
    _, _, X_test, _ = titanic_survive()
    return len(X_test)

@pytest.fixture(scope="session")
def fitted_rf_classifier_model():
    X_train, y_train, _, _ = titanic_survive()
    model = RandomForestClassifier(n_estimators=5, max_depth=2)
    model.fit(X_train, y_train)
    return model


@pytest.fixture(scope="session")
def rf_classifier_explainer(fitted_rf_classifier_model):
    _, _, X_test, y_test = titanic_survive()
    explainer = ClassifierExplainer(
        fitted_rf_classifier_model, 
        X_test, 
        y_test, 
        cats=[{'Gender': ['Sex_female', 'Sex_male', 'Sex_nan']}, 'Deck', 'Embarked'],
        cats_notencoded={'Gender': 'No Gender'},
        labels=['Not survived', 'Survived']
    )
    return explainer


@pytest.fixture(scope="session")
def precalculated_rf_classifier_explainer(rf_classifier_explainer):
    db = ExplainerDashboard(rf_classifier_explainer)
    return rf_classifier_explainer


@pytest.fixture(scope="session")
def fitted_rf_regression_model():
    X_train, y_train, _, _ = titanic_fare()
    model = RandomForestRegressor(n_estimators=5, max_depth=2)
    model.fit(X_train, y_train)
    return model


@pytest.fixture(scope="session")
def rf_regression_explainer(fitted_rf_regression_model):
    _, _, X_test, y_test = titanic_fare()
    _, test_names = titanic_names()
    explainer = RegressionExplainer(
        fitted_rf_regression_model, 
        X_test, y_test,
        cats=[{'Gender': ['Sex_female', 'Sex_male', 'Sex_nan']}, 'Deck', 'Embarked'],
        idxs=test_names)
    return explainer


@pytest.fixture(scope="session")
def precalculated_rf_regression_explainer(rf_regression_explainer):
    db = ExplainerDashboard(rf_regression_explainer)
    return rf_regression_explainer


