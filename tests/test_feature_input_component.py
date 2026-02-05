from sklearn.ensemble import RandomForestClassifier

from explainerdashboard import ClassifierExplainer
from explainerdashboard.dashboard_components.overview_components import (
    FeatureInputComponent,
)


def test_feature_input_component_handles_bool_columns(classifier_data):
    X_train, y_train, X_test, y_test = classifier_data

    X_train = X_train.copy()
    X_test = X_test.copy()

    cutoff = X_train["Age"].median()
    X_train["is_older"] = X_train["Age"] > cutoff
    X_test["is_older"] = X_test["Age"] > cutoff

    model = RandomForestClassifier(n_estimators=5, max_depth=2)
    model.fit(X_train, y_train)

    explainer = ClassifierExplainer(model, X_test, y_test)
    component = FeatureInputComponent(explainer)

    layout = component.layout()
    assert layout is not None
