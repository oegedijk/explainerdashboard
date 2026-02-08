#!/usr/bin/env python3
"""Launch a local ExplainerDashboard for a LightGBM classifier."""

import os
import argparse

# DTreeViz renders legends with Matplotlib inside Dash callback threads.
# On macOS the default GUI backend can crash outside the main thread.
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib
from lightgbm import LGBMClassifier

from explainerdashboard import ClassifierExplainer, ExplainerDashboard
from explainerdashboard.datasets import titanic_survive, titanic_names

matplotlib.use("Agg", force=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a LightGBM model and launch an ExplainerDashboard."
    )
    parser.add_argument("--port", type=int, default=8050, help="Dashboard port.")
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Host interface to bind."
    )
    parser.add_argument(
        "--n-estimators", type=int, default=50, help="Number of boosting rounds."
    )
    parser.add_argument("--max-depth", type=int, default=3, help="Max tree depth.")
    return parser.parse_args()


def main():
    args = parse_args()

    X_train, y_train, X_test, y_test = titanic_survive()
    test_names = titanic_names()[1]

    model = LGBMClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=42,
        verbosity=-1,
    )
    model.fit(X_train, y_train)

    explainer = ClassifierExplainer(
        model,
        X_test,
        y_test,
        idxs=test_names,
        labels=["Not survived", "Survived"],
        cats=[{"Gender": ["Sex_female", "Sex_male", "Sex_nan"]}, "Deck", "Embarked"],
        cats_notencoded={"Gender": "No Gender"},
    )

    dashboard = ExplainerDashboard(
        explainer,
        title="LightGBM Explainer (TreeViz Demo)",
        decision_trees=True,
        shap_interaction=False,
    )
    dashboard.run(port=args.port, host=args.host)


if __name__ == "__main__":
    main()
