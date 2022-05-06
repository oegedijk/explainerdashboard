import pytest

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBClassifier, XGBRegressor
from lightgbm.sklearn import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

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


@pytest.fixture(scope="session")
def fitted_xgb_classifier_model():
    X_train, y_train, _, _ = titanic_survive()
    model = XGBClassifier(n_estimators=5, max_depth=2)
    model.fit(X_train, y_train)
    return model


@pytest.fixture(scope="session")
def xgb_classifier_explainer(fitted_xgb_classifier_model):
    _, _, X_test, y_test = titanic_survive()
    explainer = ClassifierExplainer(
        fitted_xgb_classifier_model, 
        X_test, 
        y_test, 
        cats=[{'Gender': ['Sex_female', 'Sex_male', 'Sex_nan']}, 'Deck', 'Embarked'],
        cats_notencoded={'Gender': 'No Gender'},
        labels=['Not survived', 'Survived']
    )
    return explainer


@pytest.fixture(scope="session")
def precalculated_xgb_classifier_explainer(xgb_classifier_explainer):
    ExplainerDashboard(xgb_classifier_explainer)
    return xgb_classifier_explainer


@pytest.fixture(scope="session")
def fitted_xgb_regression_model():
    X_train, y_train, _, _ = titanic_fare()
    model = XGBRegressor(n_estimators=5, max_depth=2)
    model.fit(X_train, y_train)
    return model


@pytest.fixture(scope="session")
def xgb_regression_explainer(fitted_xgb_regression_model):
    _, _, X_test, y_test = titanic_fare()
    _, test_names = titanic_names()
    explainer = RegressionExplainer(
        fitted_xgb_regression_model, 
        X_test, y_test,
        cats=[{'Gender': ['Sex_female', 'Sex_male', 'Sex_nan']}, 'Deck', 'Embarked'],
        idxs=test_names)
    return explainer


@pytest.fixture(scope="session")
def precalculated_xgb_regression_explainer(xgb_regression_explainer):
    _ = ExplainerDashboard(xgb_regression_explainer)
    return xgb_regression_explainer


@pytest.fixture(scope="session")
def fitted_lgbm_classifier_model():
    X_train, y_train, _, _ = titanic_survive()
    model = LGBMClassifier(n_estimators=5, max_depth=2)
    model.fit(X_train, y_train)
    return model


@pytest.fixture(scope="session")
def lgbm_classifier_explainer(fitted_lgbm_classifier_model):
    _, _, X_test, y_test = titanic_survive()
    explainer = ClassifierExplainer(
        fitted_lgbm_classifier_model, 
        X_test, 
        y_test, 
        cats=[{'Gender': ['Sex_female', 'Sex_male', 'Sex_nan']}, 'Deck', 'Embarked'],
        cats_notencoded={'Gender': 'No Gender'},
        labels=['Not survived', 'Survived']
    )
    return explainer


@pytest.fixture(scope="session")
def precalculated_lgbm_classifier_explainer(lgbm_classifier_explainer):
    ExplainerDashboard(lgbm_classifier_explainer)
    return lgbm_classifier_explainer


@pytest.fixture(scope="session")
def fitted_lgbm_regression_model():
    X_train, y_train, _, _ = titanic_fare()
    model = LGBMRegressor(n_estimators=5, max_depth=2)
    model.fit(X_train, y_train)
    return model


@pytest.fixture(scope="session")
def lgbm_regression_explainer(fitted_lgbm_regression_model):
    _, _, X_test, y_test = titanic_fare()
    _, test_names = titanic_names()
    explainer = RegressionExplainer(
        fitted_lgbm_regression_model, 
        X_test, y_test,
        cats=[{'Gender': ['Sex_female', 'Sex_male', 'Sex_nan']}, 'Deck', 'Embarked'],
        idxs=test_names)
    return explainer


@pytest.fixture(scope="session")
def precalculated_lgbm_regression_explainer(lgbm_regression_explainer):
    _ = ExplainerDashboard(lgbm_regression_explainer)
    return lgbm_regression_explainer


@pytest.fixture(scope="session")
def catboost_classifier_explainer():
    X_train, y_train, X_test, y_test = titanic_survive()

    model = CatBoostClassifier(iterations=100, verbose=0).fit(X_train, y_train)
    explainer = ClassifierExplainer(
                        model, X_test, y_test, 
                        cats=['Deck', 'Embarked'],
                        labels=['Not survived', 'Survived'])

    X_cats, y_cats = explainer.X_merged, explainer.y.astype("int")
    model = CatBoostClassifier(iterations=5, verbose=0).fit(X_cats, y_cats, cat_features=[8, 9])
    return ClassifierExplainer(model, X_cats, y_cats, 
                            cats=['Sex'], 
                            labels=['Not survived', 'Survived'],
                            idxs=X_test.index)

@pytest.fixture(scope="session")
def precalculated_catboost_classifier_explainer(catboost_classifier_explainer):
    ExplainerDashboard(catboost_classifier_explainer)
    return catboost_classifier_explainer


@pytest.fixture(scope="session")
def catboost_regression_explainer():
    X_train, y_train, X_test, y_test = titanic_fare()
    model = CatBoostRegressor(iterations=5, verbose=0).fit(X_train, y_train)
    explainer = RegressionExplainer(model, X_test, y_test, cats=['Deck', 'Embarked'])
    X_cats, y_cats = explainer.X_merged, explainer.y
    model = CatBoostRegressor(iterations=5, verbose=0).fit(X_cats, y_cats, cat_features=[8, 9])
    return RegressionExplainer(model, X_cats, y_cats, cats=['Sex'], idxs=X_test.index)

@pytest.fixture(scope="session")
def precalculated_catboost_regression_explainer(catboost_regression_explainer):
    ExplainerDashboard(catboost_regression_explainer)
    return catboost_regression_explainer

