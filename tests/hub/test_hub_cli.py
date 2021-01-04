import pytest
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier

from explainerdashboard import *
from explainerdashboard.datasets import *
from explainerdashboard.custom import *

@pytest.fixture
def generate_assets():
    X_train, y_train, X_test, y_test = titanic_survive()
    model = RandomForestClassifier(n_estimators=5, max_depth=2).fit(X_train, y_train)
    explainer = ClassifierExplainer(model, X_test, y_test)
    db1 = ExplainerDashboard(explainer, description="Super interesting dashboard")
    db2 = ExplainerDashboard(explainer, title="Dashboard Two", 
                                name='db2', logins=[['user2', 'password2']])
    hub = ExplainerHub([db1, db2])
    hub.to_yaml(Path.cwd() / "tests" / "test_assets" / "hub.yaml")
    return None


def test_explainerhub_cli_help(generate_assets, script_runner):
    ret = script_runner.run('explainerhub', ' --help', 
                cwd=str(Path().cwd() / "tests" / "test_assets"))
    assert ret.success
    assert ret.stderr == ''

