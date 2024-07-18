import pytest

from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBClassifier, XGBRegressor
from lightgbm.sklearn import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

from explainerdashboard import RegressionExplainer, ClassifierExplainer, ExplainerDashboard, ExplainerHub
from explainerdashboard.custom import ShapDependenceComposite
from explainerdashboard.datasets import titanic_survive, titanic_fare, titanic_names, titanic_embarked



### DATA FIXTURES


@pytest.fixture(scope="session")
def classifier_data():
    """X_train, y_train, X_test, y_test"""
    X_train, y_train, X_test, y_test = titanic_survive()
    return X_train, y_train, X_test, y_test

@pytest.fixture(scope="session")
def regression_data():
    X_train, y_train, X_test, y_test = titanic_fare()
    return X_train, y_train, X_test, y_test

@pytest.fixture(scope="session")
def multiclass_data():
    X_train, y_train, X_test, y_test = titanic_embarked()
    return X_train, y_train, X_test, y_test

@pytest.fixture(scope='session')
def categorical_classifier_data():
    df = pd.read_csv(Path.cwd() / "tests" / "test_assets" / "pipeline_data.csv")
    X = df[['age', 'fare', 'embarked', 'sex', 'pclass']]
    y = df['survived'].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return X_train, y_train, X_test, y_test


@pytest.fixture(scope="session")
def test_names():
    return titanic_names()[1]

@pytest.fixture(scope="session")
def testlen(classifier_data):
    _, _, X_test, _ = classifier_data
    return len(X_test)

@pytest.fixture(scope="session")
def dashboard_dumps_folder(tmp_path_factory, precalculated_rf_classifier_explainer, custom_dashboard):
    dump_path = tmp_path_factory.mktemp("dump_tests")

    precalculated_rf_classifier_explainer.dump(dump_path / "explainer.joblib")
    precalculated_rf_classifier_explainer.to_yaml(dump_path / "explainer.yaml")
    custom_dashboard.to_yaml(dump_path / "dashboard.yaml", explainerfile=str(dump_path / "explainer.joblib"))
    return dump_path

@pytest.fixture(scope="session")
def explainer_hub_dump_folder(tmp_path_factory, explainer_hub):
    dump_path = tmp_path_factory.mktemp("hub_dump")
    explainer_hub.to_yaml(dump_path / "hub.yaml")
    return dump_path

### MODEL FIXTURES


###### CLASSIFIER MODEL FIXTURES

@pytest.fixture(scope="session")
def fitted_rf_classifier_model(classifier_data):
    X_train, y_train, _, _ = classifier_data
    model = RandomForestClassifier(n_estimators=5, max_depth=2)
    model.fit(X_train, y_train)
    return model

@pytest.fixture(scope="session")
def fitted_xgb_classifier_model(classifier_data):
    X_train, y_train, _, _ = classifier_data
    model = XGBClassifier(n_estimators=5, max_depth=2)
    model.fit(X_train, y_train)
    return model

@pytest.fixture(scope="session")
def fitted_lgbm_classifier_model(classifier_data):
    X_train, y_train, _, _ = classifier_data
    model = LGBMClassifier(n_estimators=5, max_depth=2)
    model.fit(X_train, y_train)
    return model

@pytest.fixture(scope="session")
def fitted_dt_classifier_model(classifier_data):
    X_train, y_train, _, _ = classifier_data
    model = DecisionTreeClassifier(max_depth=2)
    model.fit(X_train, y_train)
    return model

@pytest.fixture(scope="session")
def fitted_et_classifier_model(classifier_data):
    X_train, y_train, _, _ = classifier_data
    model = ExtraTreesClassifier(n_estimators=5, max_depth=2)
    model.fit(X_train, y_train)
    return model

@pytest.fixture(scope="session")
def fitted_logistic_regression_model(classifier_data):
    X_train, y_train, _, _ = classifier_data
    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)
    return model

####### REGRESSION MODEL FIXTURES

@pytest.fixture(scope="session")
def fitted_rf_regression_model(regression_data):
    X_train, y_train, _, _ = regression_data
    model = RandomForestRegressor(n_estimators=5, max_depth=2)
    model.fit(X_train, y_train)
    return model

@pytest.fixture(scope="session")
def fitted_xgb_regression_model(regression_data):
    X_train, y_train, _, _ = regression_data
    model = XGBRegressor(n_estimators=5, max_depth=2)
    model.fit(X_train, y_train)
    return model

@pytest.fixture(scope="session")
def fitted_lgbm_regression_model(regression_data):
    X_train, y_train, _, _ = regression_data
    model = LGBMRegressor()
    model.fit(X_train, y_train)
    return model

@pytest.fixture(scope="session")
def fitted_linear_regression_model(regression_data):
    X_train, y_train, _, _ = regression_data
    model = LinearRegression()   
    model.fit(X_train, y_train)
    return model

@pytest.fixture(scope="session")
def fitted_dt_regression_model(regression_data):
    "DecisionTreeRegressor"
    X_train, y_train, _, _ = regression_data
    model = DecisionTreeRegressor(max_depth=2)
    model.fit(X_train, y_train)
    return model

@pytest.fixture(scope="session")
def fitted_et_regression_model(regression_data):
    X_train, y_train, _, _ = regression_data
    model = ExtraTreesRegressor(n_estimators=5, max_depth=2)
    model.fit(X_train, y_train)
    return model




###### MULTICLASS MODEL FIXTURES

@pytest.fixture(scope="session")
def fitted_rf_multiclass_model(multiclass_data):
    X_train, y_train, _, _ = multiclass_data
    model = RandomForestClassifier(n_estimators=5, max_depth=2)
    model.fit(X_train, y_train)
    return model

@pytest.fixture(scope="session")
def fitted_xgb_multiclass_model(multiclass_data):
    X_train, y_train, _, _ = multiclass_data
    model = XGBClassifier(n_estimators=5, max_depth=2)
    model.fit(X_train, y_train)
    return model

###### PIPELINE MODEL FIXTURES

@pytest.fixture(scope='session')
def fitted_classifier_pipeline(categorical_classifier_data):
    X_train,  y_train, _, _ = categorical_classifier_data

    numeric_features = ['age', 'fare']
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    categorical_features = ['embarked', 'sex', 'pclass']
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ordinal', OrdinalEncoder())])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)])

    # Append classifier to preprocessing pipeline.
    # Now we have a full prediction pipeline.
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                        ('classifier', RandomForestClassifier(n_estimators=5, max_depth=3))])

    pipeline.fit(X_train, y_train)
    return pipeline


### EXPLAINER FIXTURES


@pytest.fixture(scope="session")
def rf_classifier_explainer(fitted_rf_classifier_model, classifier_data):
    _, _, X_test, y_test = classifier_data
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
def logistic_regression_explainer(fitted_logistic_regression_model, classifier_data):
    _, _, X_test, y_test = classifier_data
    explainer = ClassifierExplainer(
        fitted_logistic_regression_model, 
        X_test, 
        y_test, 
        cats=[{'Gender': ['Sex_female', 'Sex_male', 'Sex_nan']}, 'Deck', 'Embarked'],
        cats_notencoded={'Gender': 'No Gender'},
        labels=['Not survived', 'Survived']
    )
    return explainer

@pytest.fixture(scope="session")
def logistic_regression_kernel_explainer(fitted_logistic_regression_model, classifier_data):
    _, _, X_test, y_test = classifier_data
    explainer = ClassifierExplainer(
        fitted_logistic_regression_model, 
        X_test.iloc[:10], y_test.iloc[:10],
        cats=[{'Gender': ['Sex_female', 'Sex_male', 'Sex_nan']}, 'Deck', 'Embarked'],
        cats_notencoded={'Gender': 'No Gender'},
        labels=['Not survived', 'Survived'],
        shap='kernel', 
    )
    return explainer


@pytest.fixture(scope="session")
def rf_classifier_explainer_no_y(fitted_rf_classifier_model, classifier_data):
    _, _, X_test, _ = classifier_data
    explainer = ClassifierExplainer(
        fitted_rf_classifier_model, 
        X_test, 
        cats=[{'Gender': ['Sex_female', 'Sex_male', 'Sex_nan']}, 'Deck', 'Embarked'],
        cats_notencoded={'Gender': 'No Gender'},
        labels=['Not survived', 'Survived']
    )
    return explainer

@pytest.fixture(scope="session")
def rf_regression_explainer(fitted_rf_regression_model, regression_data, test_names):
    _, _, X_test, y_test = regression_data
    explainer = RegressionExplainer(
        fitted_rf_regression_model, 
        X_test, y_test,
        cats=[{'Gender': ['Sex_female', 'Sex_male', 'Sex_nan']}, 'Deck', 'Embarked'],
        idxs=test_names)
    return explainer

@pytest.fixture(scope="session")
def linear_regression_explainer(fitted_linear_regression_model, regression_data, test_names):
    _, _, X_test, y_test = regression_data
    explainer = RegressionExplainer(
        fitted_linear_regression_model, 
        X_test, y_test,
        cats=[{'Gender': ['Sex_female', 'Sex_male', 'Sex_nan']}, 'Deck', 'Embarked'],
        idxs=test_names)
    return explainer


@pytest.fixture(scope="session")
def linear_regression_kernel_explainer(fitted_linear_regression_model, regression_data, test_names):
    _, _, X_test, y_test = regression_data
    explainer = RegressionExplainer(
        fitted_linear_regression_model, 
        X_test.iloc[:10], y_test.iloc[:10],
        cats=[{'Gender': ['Sex_female', 'Sex_male', 'Sex_nan']}, 'Deck', 'Embarked'],
        shap='kernel')
    return explainer

@pytest.fixture(scope="session")
def rf_regression_explainer_no_y(fitted_rf_regression_model, regression_data, test_names):
    _, _, X_test, _ = regression_data
    explainer = RegressionExplainer(
        fitted_rf_regression_model, 
        X_test, 
        cats=[{'Gender': ['Sex_female', 'Sex_male', 'Sex_nan']}, 'Deck', 'Embarked'],
        idxs=test_names)
    return explainer

@pytest.fixture(scope="session")
def xgb_classifier_explainer(fitted_xgb_classifier_model, classifier_data):
    _, _, X_test, y_test = classifier_data
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
def xgb_regression_explainer(fitted_xgb_regression_model, regression_data, test_names):
    _, _, X_test, y_test = regression_data
    explainer = RegressionExplainer(
        fitted_xgb_regression_model, 
        X_test, y_test,
        cats=[{'Gender': ['Sex_female', 'Sex_male', 'Sex_nan']}, 'Deck', 'Embarked'],
        idxs=test_names)
    return explainer

@pytest.fixture(scope="session")
def lgbm_classifier_explainer(fitted_lgbm_classifier_model, classifier_data):
    _, _, X_test, y_test = classifier_data
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
def lgbm_regression_explainer(fitted_lgbm_regression_model, regression_data, test_names):
    _, _, X_test, y_test = regression_data
    explainer = RegressionExplainer(
        fitted_lgbm_regression_model, 
        X_test, y_test,
        cats=[{'Gender': ['Sex_female', 'Sex_male', 'Sex_nan']}, 'Deck', 'Embarked'],
        idxs=test_names)
    return explainer

@pytest.fixture(scope="session")
def catboost_classifier_explainer(classifier_data):
    X_train, y_train, X_test, y_test = classifier_data

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
def catboost_regression_explainer(regression_data):
    X_train, y_train, X_test, y_test = regression_data
    model = CatBoostRegressor(iterations=5, verbose=0).fit(X_train, y_train)
    explainer = RegressionExplainer(model, X_test, y_test, cats=['Deck', 'Embarked'])
    X_cats, y_cats = explainer.X_merged, explainer.y
    model = CatBoostRegressor(iterations=5, verbose=0).fit(X_cats, y_cats, cat_features=[8, 9])
    return RegressionExplainer(model, X_cats, y_cats, cats=['Sex'], idxs=X_test.index)

@pytest.fixture(scope="session")
def dt_classifier_explainer(fitted_dt_classifier_model, classifier_data):
    _, _, X_test, y_test = classifier_data
    explainer = ClassifierExplainer(
        fitted_dt_classifier_model, 
        X_test, 
        y_test, 
        cats=[{'Gender': ['Sex_female', 'Sex_male', 'Sex_nan']}, 'Deck', 'Embarked'],
        cats_notencoded={'Gender': 'No Gender'},
        labels=['Not survived', 'Survived']
    )
    return explainer

@pytest.fixture(scope="session")
def dt_regression_explainer(fitted_dt_regression_model, test_names, regression_data):
    "DecisionTreeRegressor explainer"
    _, _, X_test, y_test = regression_data
    explainer = RegressionExplainer(
        fitted_dt_regression_model, 
        X_test, y_test,
        cats=[{'Gender': ['Sex_female', 'Sex_male', 'Sex_nan']}, 'Deck', 'Embarked'],
        idxs=test_names)
    return explainer

@pytest.fixture(scope="session")
def et_classifier_explainer(fitted_et_classifier_model, classifier_data):
    _, _, X_test, y_test = classifier_data
    explainer = ClassifierExplainer(
        fitted_et_classifier_model, 
        X_test, 
        y_test, 
        cats=[{'Gender': ['Sex_female', 'Sex_male', 'Sex_nan']}, 'Deck', 'Embarked'],
        cats_notencoded={'Gender': 'No Gender'},
        labels=['Not survived', 'Survived']
    )
    return explainer

@pytest.fixture(scope="session")
def et_regression_explainer(fitted_et_regression_model, test_names, regression_data):
    _, _, X_test, y_test = regression_data
    explainer = RegressionExplainer(
        fitted_et_regression_model, 
        X_test, y_test,
        cats=[{'Gender': ['Sex_female', 'Sex_male', 'Sex_nan']}, 'Deck', 'Embarked'],
        idxs=test_names)
    return explainer

@pytest.fixture(scope="session")
def rf_multiclass_explainer(fitted_rf_multiclass_model, test_names, multiclass_data):
    _, _, X_test, y_test = multiclass_data
    explainer = ClassifierExplainer(fitted_rf_multiclass_model, X_test, y_test,  
        cats=[{'Gender': ['Sex_female', 'Sex_male', 'Sex_nan']}, 'Deck'],
        idxs=test_names, 
        labels=['Queenstown', 'Southampton', 'Cherbourg'])
    return explainer

@pytest.fixture(scope="session")
def rf_multiclass_explainer_no_y(fitted_rf_multiclass_model, test_names, multiclass_data):
    _, _, X_test, _ = multiclass_data
    explainer = ClassifierExplainer(fitted_rf_multiclass_model, X_test, 
        cats=[{'Gender': ['Sex_female', 'Sex_male', 'Sex_nan']}, 'Deck'],
        idxs=test_names, 
        labels=['Queenstown', 'Southampton', 'Cherbourg'])
    return explainer

@pytest.fixture(scope="session")
def xgb_multiclass_explainer(fitted_xgb_multiclass_model, test_names, multiclass_data):
    _, _, X_test, y_test = multiclass_data
    explainer = ClassifierExplainer(fitted_xgb_multiclass_model, X_test, y_test,  
                            cats=[{'Gender': ['Sex_female', 'Sex_male', 'Sex_nan']}, 'Deck'],
                            idxs=test_names, 
                            labels=['Queenstown', 'Southampton', 'Cherbourg'])
    return explainer

@pytest.fixture(scope="session")
def classifier_pipeline_explainer(fitted_classifier_pipeline, categorical_classifier_data):
    _, _, X_test, y_test = categorical_classifier_data
    return ClassifierExplainer(fitted_classifier_pipeline, X_test, y_test)


@pytest.fixture(scope="session")
def classifier_pipeline_kernel_explainer(fitted_classifier_pipeline, categorical_classifier_data):
    _, _, X_test, y_test = categorical_classifier_data
    return ClassifierExplainer(fitted_classifier_pipeline, X_test, y_test, shap='kernel')

### PRECALCULATED EXPLAINER FIXTURES

@pytest.fixture(scope="session")
def precalculated_rf_classifier_explainer(rf_classifier_explainer):
    _ = ExplainerDashboard(rf_classifier_explainer)
    return rf_classifier_explainer

@pytest.fixture(scope="session")
def precalculated_logistic_regression_explainer(logistic_regression_explainer):
    _ = ExplainerDashboard(logistic_regression_explainer)
    return logistic_regression_explainer

@pytest.fixture(scope="session")
def precalculated_rf_classifier_explainer_no_y(rf_classifier_explainer_no_y):
    _ = ExplainerDashboard(rf_classifier_explainer_no_y)
    return rf_classifier_explainer_no_y

@pytest.fixture(scope="session")
def precalculated_rf_regression_explainer(rf_regression_explainer):
    _ = ExplainerDashboard(rf_regression_explainer)
    return rf_regression_explainer

@pytest.fixture(scope="session")
def precalculated_linear_regression_explainer(linear_regression_explainer):
    _ = ExplainerDashboard(linear_regression_explainer)
    return linear_regression_explainer

@pytest.fixture(scope="session")
def precalculated_rf_regression_explainer_no_y(rf_regression_explainer_no_y):
    _ = ExplainerDashboard(rf_regression_explainer_no_y)
    return rf_regression_explainer_no_y

@pytest.fixture(scope="session")
def precalculated_xgb_classifier_explainer(xgb_classifier_explainer):
    ExplainerDashboard(xgb_classifier_explainer)
    return xgb_classifier_explainer

@pytest.fixture(scope="session")
def precalculated_xgb_regression_explainer(xgb_regression_explainer):
    _ = ExplainerDashboard(xgb_regression_explainer)
    return xgb_regression_explainer

@pytest.fixture(scope="session")
def precalculated_lgbm_classifier_explainer(lgbm_classifier_explainer):
    _ = ExplainerDashboard(lgbm_classifier_explainer)
    return lgbm_classifier_explainer

@pytest.fixture(scope="session")
def precalculated_lgbm_regression_explainer(lgbm_regression_explainer):
    _ = ExplainerDashboard(lgbm_regression_explainer)
    return lgbm_regression_explainer

@pytest.fixture(scope="session")
def precalculated_catboost_classifier_explainer(catboost_classifier_explainer):
    _ = ExplainerDashboard(catboost_classifier_explainer)
    return catboost_classifier_explainer

@pytest.fixture(scope="session")
def precalculated_catboost_regression_explainer(catboost_regression_explainer):
    _ = ExplainerDashboard(catboost_regression_explainer)
    return catboost_regression_explainer

@pytest.fixture(scope="session")
def precalculated_dt_classifier_explainer(dt_classifier_explainer):
    _ = ExplainerDashboard(dt_classifier_explainer)
    return dt_classifier_explainer

@pytest.fixture(scope="session")
def precalculated_dt_regression_explainer(dt_regression_explainer):
    _ = ExplainerDashboard(dt_regression_explainer)
    return dt_regression_explainer

@pytest.fixture(scope="session")
def precalculated_et_classifier_explainer(et_classifier_explainer):
    _ = ExplainerDashboard(et_classifier_explainer)
    return et_classifier_explainer

@pytest.fixture(scope="session")
def precalculated_et_regression_explainer(et_regression_explainer):
    _ = ExplainerDashboard(et_regression_explainer)
    return et_regression_explainer

@pytest.fixture(scope="session")
def precalculated_rf_multiclass_explainer(rf_multiclass_explainer):
    _ = ExplainerDashboard(rf_multiclass_explainer)
    return rf_multiclass_explainer

@pytest.fixture(scope="session")
def precalculated_rf_multiclass_explainer_no_y(rf_multiclass_explainer_no_y):
    _ = ExplainerDashboard(rf_multiclass_explainer_no_y)
    return rf_multiclass_explainer_no_y

@pytest.fixture(scope="session")
def precalculated_xgb_multiclass_explainer(xgb_multiclass_explainer):
    _ = ExplainerDashboard(xgb_multiclass_explainer)
    return xgb_multiclass_explainer


### OTHER FIXTURES

@pytest.fixture(scope="session")
def custom_dashboard(precalculated_rf_classifier_explainer):
    custom_dashboard = ExplainerDashboard(precalculated_rf_classifier_explainer, 
            [
                ShapDependenceComposite(precalculated_rf_classifier_explainer, title="Test Tab!"),
                ShapDependenceComposite, 
                "importances"
            ], title="Test Title!"
    )
    return custom_dashboard


@pytest.fixture(scope="session")
def explainer_hub(precalculated_rf_classifier_explainer, precalculated_rf_regression_explainer):
    hub = ExplainerHub([
                ExplainerDashboard(precalculated_rf_classifier_explainer, description="Super interesting dashboard"),
                ExplainerDashboard(precalculated_rf_regression_explainer, title="Dashboard Two", 
                        name='db2', logins=[['user2', 'password2']])
            ], 
            users_file=str(Path.cwd() / "tests" / "test_assets" / "users.yaml")
    )
    return hub


