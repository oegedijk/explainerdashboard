import pytest

pytestmark = pytest.mark.cli

def test_explainerhub_cli_help(explainer_hub_dump_folder, script_runner):
    ret = script_runner.run(['explainerhub', ' --help'], cwd=str(explainer_hub_dump_folder))
    assert ret.success
    assert ret.stderr == ''

