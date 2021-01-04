import unittest
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from explainerdashboard import *
from explainerdashboard.datasets import *
from explainerdashboard.custom import *

class ExplainerHubTests(unittest.TestCase):
    def setUp(self):
        X_train, y_train, X_test, y_test = titanic_survive()

        model = RandomForestClassifier(n_estimators=5, max_depth=2).fit(X_train, y_train)
        self.explainer = ClassifierExplainer(model, X_test, y_test)
        self.db1 = ExplainerDashboard(self.explainer, description="Super interesting dashboard")
        self.db2 = ExplainerDashboard(self.explainer, title="Dashboard Two", 
                        name='db2', logins=[['user2', 'password2']])
        self.hub = ExplainerHub([self.db1, self.db2], users_file=str(Path.cwd() / "tests" / "test_assets" / "users.yaml"))
        
    def test_hub_users(self):
        self.assertGreater(len(self.hub.users), 0)
        self.assertIn("db2", self.hub.dashboards_with_users)
        self.hub.add_user("user3", "password")
        self.hub.add_user_to_dashboard("db2", "user3")
        self.assertIn("user3", self.hub.dashboard_users['db2'])
        self.hub.add_user("user4", "password", add_to_users_file=True)
        self.hub.add_user_to_dashboard("db2", "user4", add_to_users_file=True)
        self.assertIn("user4", self.hub.dashboard_users['db2'])
        self.assertIn("user4", self.hub.get_dashboard_users("db2"))

    def test_load_from_config(self):
        self.hub.to_yaml(Path.cwd() / "tests" / "test_assets" / "hub.yaml")
        self.hub2 = ExplainerHub.from_config(Path.cwd() / "tests" / "test_assets" / "hub.yaml")
