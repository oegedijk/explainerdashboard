from lightgbm.sklearn import LGBMClassifier
from lightgbm.sklearn import LGBMRegressor

from explainerdashboard import ClassifierExplainer, ExplainerDashboard
from explainerdashboard import RegressionExplainer


def _cast_string_cats_to_category(X):
    X = X.copy()
    for col in ["embarked", "sex"]:
        X[col] = X[col].astype("category")
    return X


def test_lgbm_string_categorical_values_do_not_crash_dashboard(
    categorical_classifier_data,
):
    X_train, y_train, X_test, y_test = categorical_classifier_data

    X_train = _cast_string_cats_to_category(X_train)
    X_test = _cast_string_cats_to_category(X_test)

    model = LGBMClassifier(n_estimators=5, max_depth=2, random_state=0)
    model.fit(X_train, y_train, feature_name="auto", categorical_feature="auto")

    explainer = ClassifierExplainer(
        model,
        X_test,
        y_test,
        model_output="logodds",
        labels=["Not survived", "Survived"],
    )
    dashboard = ExplainerDashboard(
        explainer,
        title="String categorical LGBM test",
        whatif=False,
        shap_interaction=False,
        decision_trees=False,
    )
    assert dashboard is not None


def test_lgbm_string_categorical_values_do_not_crash_get_shap_row(
    categorical_classifier_data,
):
    X_train, y_train, X_test, y_test = categorical_classifier_data

    X_train = _cast_string_cats_to_category(X_train)
    X_test = _cast_string_cats_to_category(X_test)

    model = LGBMClassifier(n_estimators=5, max_depth=2, random_state=0)
    model.fit(X_train, y_train, feature_name="auto", categorical_feature="auto")

    explainer = ClassifierExplainer(
        model,
        X_test,
        y_test,
        model_output="logodds",
        labels=["Not survived", "Survived"],
    )
    shap_row = explainer.get_shap_row(X_row=X_test.iloc[[0]])
    assert shap_row is not None


def test_lgbm_string_categorical_values_unseen_category_in_xrow_does_not_crash(
    categorical_classifier_data,
):
    X_train, y_train, X_test, y_test = categorical_classifier_data

    X_train = _cast_string_cats_to_category(X_train)
    X_test = _cast_string_cats_to_category(X_test)

    model = LGBMClassifier(n_estimators=5, max_depth=2, random_state=0)
    model.fit(X_train, y_train, feature_name="auto", categorical_feature="auto")

    explainer = ClassifierExplainer(
        model,
        X_test,
        y_test,
        model_output="logodds",
        labels=["Not survived", "Survived"],
    )

    X_row = X_test.iloc[[0]].copy()
    X_row["embarked"] = X_row["embarked"].cat.add_categories(["unseen_port"])
    X_row.loc[:, "embarked"] = "unseen_port"
    shap_row = explainer.get_shap_row(X_row=X_row)
    assert shap_row is not None


def test_lgbm_regression_string_categorical_values_do_not_crash_dashboard(
    categorical_classifier_data,
):
    X_train, _, X_test, _ = categorical_classifier_data

    X_train = _cast_string_cats_to_category(X_train)
    X_test = _cast_string_cats_to_category(X_test)

    y_train = X_train["fare"]
    y_test = X_test["fare"]
    X_train = X_train.drop(columns=["fare"])
    X_test = X_test.drop(columns=["fare"])

    model = LGBMRegressor(n_estimators=5, max_depth=2, random_state=0)
    model.fit(X_train, y_train, feature_name="auto", categorical_feature="auto")

    explainer = RegressionExplainer(model, X_test, y_test)
    dashboard = ExplainerDashboard(
        explainer,
        title="String categorical LGBM regressor test",
        whatif=False,
        shap_interaction=False,
        decision_trees=False,
    )
    assert dashboard is not None
