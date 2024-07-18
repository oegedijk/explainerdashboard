import pytest
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier

from explainerdashboard import ClassifierExplainer, ExplainerDashboard
from explainerdashboard.datasets import titanic_survive
from explainerdashboard.custom import ShapDependenceComposite

pytestmark = pytest.mark.cli

@pytest.fixture(scope="session")
def generate_assets():
    X_train, y_train, X_test, y_test = titanic_survive()

    model = RandomForestClassifier(n_estimators=5, max_depth=2)
    model.fit(X_train, y_train)

    explainer = ClassifierExplainer(
                        model, X_test, y_test, 
                        cats=[{'Gender': ['Sex_female', 'Sex_male', 'Sex_nan']}, 
                                            'Deck', 'Embarked'],
                        labels=['Not survived', 'Survived'])

    dashboard = ExplainerDashboard(explainer, 
        [
            ShapDependenceComposite(explainer, title="Test Tab!"),
            ShapDependenceComposite, 
            "importances"
        ], title="Test Title!")

    pkl_dir = Path.cwd() / "tests" / "test_assets" 
    explainer.to_yaml(pkl_dir / "explainer.yaml")
    dashboard.to_yaml(pkl_dir / "dashboard.yaml", 
                explainerfile=str(pkl_dir / "explainer.joblib"), 
                dump_explainer=True)
    return None


def test_explainerdashboard_cli_help(generate_assets, script_runner):
    ret = script_runner.run(['explainerdashboard', ' --help'], 
                cwd=str(Path().cwd() / "tests" / "test_assets"))
    assert ret.success
    assert ret.stderr == ''


def test_explainerdashboard_cli_explainer(generate_assets, script_runner):
    ret = script_runner.run(['explainerdashboard', ' test explainer.joblib'], 
                cwd=str(Path().cwd() / "tests" / "test_assets"))
    assert ret.success
    assert ret.stderr == ''


def test_explainerdashboard_cli_yaml(generate_assets, script_runner):
    ret = script_runner.run(['explainerdashboard', ' test dashboard.yaml'], 
                cwd=str(Path().cwd() / "tests" / "test_assets"))
    assert ret.success
    assert ret.stderr == ''


def test_explainerdashboard_cli_build(generate_assets, script_runner):
    ret = script_runner.run(['explainerdashboard', ' build'], 
                cwd=str(Path().cwd() / "tests" / "test_assets"))
    assert ret.success
    assert ret.stderr == ''


def test_explainerdashboard_cli_build_explainer(generate_assets, script_runner):
    ret = script_runner.run(['explainerdashboard', ' build explainer.yaml'], 
                cwd=str(Path().cwd() / "tests" / "test_assets"))
    assert ret.success
    assert ret.stderr == ''


def test_explainerdashboard_cli_build_explainer_dashboard(generate_assets, script_runner):
    ret = script_runner.run(['explainerdashboard', ' build explainer.yaml dashboard.yaml'], 
                cwd=str(Path().cwd() / "tests" / "test_assets"))
    assert ret.success
    assert ret.stderr == ''