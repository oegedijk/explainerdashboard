from explainerdashboard import ExplainerDashboard
    
def test_yaml(custom_dashboard):
    yaml = custom_dashboard.to_yaml()
    assert isinstance(yaml, str)

def test_yaml_dic(custom_dashboard):
    yaml_dict = custom_dashboard.to_yaml(return_dict=True)
    assert isinstance(yaml_dict, dict)
    assert ("dashboard" in yaml_dict)

def test_load_config_joblib(dashboard_dumps_folder):
    loaded_dashboard = ExplainerDashboard.from_config(
        dashboard_dumps_folder / "explainer.joblib",
        dashboard_dumps_folder / "dashboard.yaml")
    assert isinstance(loaded_dashboard, ExplainerDashboard)

def test_load_config_yaml(dashboard_dumps_folder):
    loaded_dashboard = ExplainerDashboard.from_config(
        dashboard_dumps_folder / "dashboard.yaml")
    assert isinstance(loaded_dashboard, ExplainerDashboard)

def test_load_config_explainer(precalculated_rf_classifier_explainer, dashboard_dumps_folder):
    loaded_dashboard = ExplainerDashboard.from_config(
        precalculated_rf_classifier_explainer, dashboard_dumps_folder / "dashboard.yaml")
    assert isinstance(loaded_dashboard, ExplainerDashboard)
        