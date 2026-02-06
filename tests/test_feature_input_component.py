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


def test_feature_input_component_respects_custom_range_and_rounding(classifier_data):
    X_train, y_train, X_test, y_test = classifier_data

    model = RandomForestClassifier(n_estimators=5, max_depth=2)
    model.fit(X_train, y_train)

    explainer = ClassifierExplainer(model, X_test, y_test)
    component = FeatureInputComponent(
        explainer, feature_input_ranges={"Age": (0, 50)}, round=1
    )

    age_div = next(
        div
        for div in component._feature_inputs
        if getattr(div.children[0], "children", None) == "Age"
    )
    age_input = age_div.children[1]
    range_text = age_div.children[2].children

    props = age_input.to_plotly_json()["props"]

    assert props.get("min") == 0
    assert props.get("max") == 50
    assert props.get("step") == 0.1
    assert range_text == "Range: 0-50"
