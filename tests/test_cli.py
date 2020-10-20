from pathlib import Path

def test_explainerdashboard_cli_help(script_runner):
    ret = script_runner.run('explainerdashboard', '--help', 
                cwd=str(Path().cwd() / "tests" / "cli_assets"))
    assert ret.success
    assert ret.stderr == ''


def test_explainerdashboard_cli(script_runner):
    ret = script_runner.run('explainerdashboard', '--no-dashboard --no-browser', 
                cwd=str(Path().cwd() / "tests" / "cli_assets"))
    assert ret.success
    assert ret.stderr == ''


def test_explainerdashboard_cli_buildonly(script_runner):
    ret = script_runner.run('explainerdashboard', '--build-only', 
                cwd=str(Path().cwd() / "tests" / "cli_assets"))
    assert ret.success
    assert ret.stderr == ''