from pathlib import Path

def test_explainerdashboard_cli_help(script_runner):
    ret = script_runner.run('explainerdashboard', ' --help', 
                cwd=str(Path().cwd() / "tests" / "cli_assets"))
    assert ret.success
    assert ret.stderr == ''


def test_explainerdashboard_cli_explainer(script_runner):
    ret = script_runner.run('explainerdashboard', ' test explainer.joblib', 
                cwd=str(Path().cwd() / "tests" / "cli_assets"))
    assert ret.success
    assert ret.stderr == ''


def test_explainerdashboard_cli_yaml(script_runner):
    ret = script_runner.run('explainerdashboard', ' test dashboard.yaml', 
                cwd=str(Path().cwd() / "tests" / "cli_assets"))
    assert ret.success
    assert ret.stderr == ''


def test_explainerdashboard_cli_build(script_runner):
    ret = script_runner.run('explainerdashboard', ' build', 
                cwd=str(Path().cwd() / "tests" / "cli_assets"))
    assert ret.success
    assert ret.stderr == ''


def test_explainerdashboard_cli_build_explainer(script_runner):
    ret = script_runner.run('explainerdashboard', ' build explainer.yaml', 
                cwd=str(Path().cwd() / "tests" / "cli_assets"))
    assert ret.success
    assert ret.stderr == ''


def test_explainerdashboard_cli_build_explainer_dashboard(script_runner):
    ret = script_runner.run('explainerdashboard', ' build explainer.yaml dashboard.yaml', 
                cwd=str(Path().cwd() / "tests" / "cli_assets"))
    assert ret.success
    assert ret.stderr == ''