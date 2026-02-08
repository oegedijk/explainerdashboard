import pytest
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


def test_feature_input_component_input_features_sets_order(classifier_data):
    X_train, y_train, X_test, y_test = classifier_data

    model = RandomForestClassifier(n_estimators=5, max_depth=2)
    model.fit(X_train, y_train)

    explainer = ClassifierExplainer(model, X_test, y_test)
    custom_features = ["Fare", "Age"]
    component = FeatureInputComponent(explainer, input_features=custom_features)

    labels = [div.children[0].children for div in component._feature_inputs]
    assert labels == custom_features


def test_feature_input_component_hide_features_hides_from_visible_inputs(
    classifier_data,
):
    X_train, y_train, X_test, y_test = classifier_data

    model = RandomForestClassifier(n_estimators=5, max_depth=2)
    model.fit(X_train, y_train)

    explainer = ClassifierExplainer(model, X_test, y_test)
    component = FeatureInputComponent(explainer, hide_features=["Age"])

    labels = [div.children[0].children for div in component._feature_inputs]
    assert "Age" not in labels
    assert len(component._hidden_feature_inputs) >= 1


def test_feature_input_component_hide_features_keeps_full_callback_contract(
    classifier_data,
):
    X_train, y_train, X_test, y_test = classifier_data

    model = RandomForestClassifier(n_estimators=5, max_depth=2)
    model.fit(X_train, y_train)

    explainer = ClassifierExplainer(model, X_test, y_test)
    component = FeatureInputComponent(explainer, hide_features=["Age"])

    expected_len = len(explainer.columns_ranked_by_shap())
    assert len(component._feature_callback_inputs) == expected_len
    assert len(component._feature_callback_outputs) == expected_len


@pytest.mark.parametrize(
    "kwargs",
    [
        {"input_features": ["feature_does_not_exist"]},
        {"hide_features": ["feature_does_not_exist"]},
        {"input_features": ["Age", "Age"]},
        {"hide_features": ["Age", "Age"]},
    ],
)
def test_feature_input_component_invalid_feature_configuration_raises(
    classifier_data, kwargs
):
    X_train, y_train, X_test, y_test = classifier_data

    model = RandomForestClassifier(n_estimators=5, max_depth=2)
    model.fit(X_train, y_train)

    explainer = ClassifierExplainer(model, X_test, y_test)
    with pytest.raises(ValueError):
        FeatureInputComponent(explainer, **kwargs)


def test_feature_input_component_empty_visible_set_raises(classifier_data):
    X_train, y_train, X_test, y_test = classifier_data

    model = RandomForestClassifier(n_estimators=5, max_depth=2)
    model.fit(X_train, y_train)

    explainer = ClassifierExplainer(model, X_test, y_test)
    with pytest.raises(ValueError):
        FeatureInputComponent(
            explainer, hide_features=list(explainer.merged_cols.tolist())
        )
