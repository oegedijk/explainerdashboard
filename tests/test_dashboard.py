import unittest

from sklearn.ensemble import RandomForestClassifier

from explainerdashboard import ClassifierExplainer, ExplainerDashboard
from explainerdashboard.datasets import titanic_survive, titanic_names
from explainerdashboard.dashboard_tabs import ShapDependenceTab


class DashboardTests(unittest.TestCase):
    def setUp(self):
        X_train, y_train, X_test, y_test = titanic_survive()
        train_names, test_names = titanic_names()

        model = RandomForestClassifier(n_estimators=5, max_depth=2)
        model.fit(X_train, y_train)

        self.explainer = ClassifierExplainer(
                            model, X_test, y_test, 
                            cats=['Sex', 'Cabin', 'Embarked'],
                            labels=['Not survived', 'Survived'])

        self.dashboard = ExplainerDashboard(self.explainer, [ShapDependenceTab, "importances"])

    def test_yaml(self):
        yaml = self.dashboard.to_yaml()
        self.assertIsInstance(yaml, str)
        