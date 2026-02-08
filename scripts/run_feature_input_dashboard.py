#!/usr/bin/env python3
import argparse

from sklearn.ensemble import RandomForestClassifier

from explainerdashboard import ClassifierExplainer, ExplainerDashboard
from explainerdashboard.datasets import titanic_survive, titanic_names


def _parse_csv_list(value):
    if value is None:
        return None
    items = [item.strip() for item in value.split(",")]
    return [item for item in items if item]


def build_dashboard(mode, input_features=None, hide_features=None):
    X_train, y_train, X_test, y_test = titanic_survive()
    _, test_names = titanic_names()
    feature_descriptions = {
        "Sex": "Gender of passenger",
        "Deck": "Deck of passenger cabin",
        "Embarked": "Port of embarkation",
        "PassengerClass": "Ticket class (1st, 2nd, 3rd)",
        "Fare": "Ticket fare",
        "Age": "Age of passenger",
        "No_of_siblings_plus_spouses_on_board": "Siblings/spouses aboard",
        "No_of_parents_plus_children_on_board": "Parents/children aboard",
    }
    model = RandomForestClassifier(n_estimators=20, max_depth=3, random_state=42)
    model.fit(X_train, y_train)

    explainer = ClassifierExplainer(
        model,
        X_test,
        y_test,
        cats=["Sex", "Deck", "Embarked"],
        idxs=test_names,
        descriptions=feature_descriptions,
        target="Survival",
        labels=["Not survived", "Survived"],
    )

    dashboard_kwargs = {}
    if mode == "custom":
        # Provide a safe default custom setup from actual feature names.
        if input_features is None:
            input_features = explainer.columns_ranked_by_shap()[:8]
        if hide_features is None:
            hide_features = input_features[-2:]
        dashboard_kwargs.update(
            input_features=input_features,
            hide_features=hide_features,
        )

    db = ExplainerDashboard(
        explainer,
        title=f"Feature Input Demo ({mode})",
        shap_interaction=False,
        **dashboard_kwargs,
    )
    return db, dashboard_kwargs


def main():
    parser = argparse.ArgumentParser(
        description="Launch a demo dashboard for FeatureInputComponent."
    )
    parser.add_argument(
        "--mode",
        choices=["vanilla", "custom"],
        default="vanilla",
        help="vanilla: no extra params, custom: pass input_features/hide_features",
    )
    parser.add_argument(
        "--input-features",
        default=None,
        help="Comma-separated feature names for input_features (custom mode only).",
    )
    parser.add_argument(
        "--hide-features",
        default=None,
        help="Comma-separated feature names for hide_features (custom mode only).",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument(
        "--build-only",
        action="store_true",
        help="Build dashboard and print config without launching server.",
    )
    args = parser.parse_args()

    input_features = _parse_csv_list(args.input_features)
    hide_features = _parse_csv_list(args.hide_features)
    db, dashboard_kwargs = build_dashboard(
        args.mode, input_features=input_features, hide_features=hide_features
    )

    print(f"Built dashboard mode={args.mode}")
    if dashboard_kwargs:
        print(f"kwargs forwarded to components: {dashboard_kwargs}")

    if args.build_only:
        return

    db.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
