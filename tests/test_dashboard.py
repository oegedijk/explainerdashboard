import unittest
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier

from explainerdashboard import ClassifierExplainer, ExplainerDashboard
from explainerdashboard.datasets import titanic_survive, titanic_names
from explainerdashboard.dashboard_tabs import ShapDependenceTab


class DashboardTests(unittest.TestCase):
    def setUp(self):
        X_train, y_train, X_test, y_test = titanic_survive()
        train_names, test_names = titanic_names()

        self.model = RandomForestClassifier(n_estimators=5, max_depth=2)
        self.model.fit(X_train, y_train)

        self.explainer = ClassifierExplainer(
                            self.model, X_test, y_test, 
                            cats=[{'Gender': ['Sex_female', 'Sex_male', 'Sex_nan']}, 
                                                'Deck', 'Embarked'],
                            labels=['Not survived', 'Survived'])

        self.dashboard = ExplainerDashboard(self.explainer, 
            [
                ShapDependenceTab(self.explainer, title="Test Tab!"),
                ShapDependenceTab, 
                "importances"
            ], title="Test Title!")

        self.pkl_dir = Path.cwd() / "tests" / "test_assets" 
        self.explainer.dump(self.pkl_dir / "explainer.joblib")
        self.explainer.to_yaml(self.pkl_dir / "explainer.yaml")
        self.dashboard.to_yaml(self.pkl_dir / "dashboard.yaml", 
                    explainerfile=str(self.pkl_dir / "explainer.joblib"))

    def test_yaml(self):
        yaml = self.dashboard.to_yaml()
        self.assertIsInstance(yaml, str)

    def test_yaml_dict(self):
        yaml_dict = self.dashboard.to_yaml(return_dict=True)
        self.assertIsInstance(yaml_dict, dict)
        self.assertIn("dashboard", yaml_dict)

    def test_load_config_joblib(self):
        db = ExplainerDashboard.from_config(
            self.pkl_dir / "explainer.joblib",
            self.pkl_dir / "dashboard.yaml")
        self.assertIsInstance(db, ExplainerDashboard)

    def test_load_config_yaml(self):
        db = ExplainerDashboard.from_config(
            self.pkl_dir / "dashboard.yaml")
        self.assertIsInstance(db, ExplainerDashboard)

    def test_load_config_explainer(self):
        db = ExplainerDashboard.from_config(
            self.explainer, self.pkl_dir / "dashboard.yaml")
        self.assertIsInstance(db, ExplainerDashboard)
        