from pathlib import Path

from explainerdashboard import ExplainerDashboard, ExplainerHub


def test_hub_users(explainer_hub):
    assert len(explainer_hub.users) > 0
    assert "db2" in explainer_hub.dashboards_with_users
    explainer_hub.add_user("user3", "password")
    explainer_hub.add_user_to_dashboard("db2", "user3")
    assert "user3" in explainer_hub.dashboard_users["db2"]
    explainer_hub.add_user("user4", "password", add_to_users_file=True)
    explainer_hub.add_user_to_dashboard("db2", "user4", add_to_users_file=True)
    assert "user4" in explainer_hub.dashboard_users["db2"]
    assert "user4" in explainer_hub.get_dashboard_users("db2")


def test_load_from_config(explainer_hub, tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("tmp_hub")
    explainer_hub.to_yaml(tmp_path / "hub.yaml")
    explainer_hub2 = ExplainerHub.from_config(tmp_path / "hub.yaml")
    assert isinstance(explainer_hub2, ExplainerHub)


def test_hub_to_html(explainer_hub):
    html = explainer_hub.to_html()
    assert isinstance(html, str)


def test_hub_save_html(explainer_hub, tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("tmp_hub")
    explainer_hub.save_html(tmp_path / "hub.html", save_dashboards=True)
    with open(tmp_path / "hub.html") as html:
        assert isinstance(html.read(), str)


def test_hub_to_zip(explainer_hub, tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("tmp_hub")
    explainer_hub.to_zip(tmp_path / "hub.zip")
    assert (tmp_path / "hub.zip").exists()


def test_add_dashboard_route_after_first_request_adds_dashboard(
    precalculated_rf_classifier_explainer,
    precalculated_rf_regression_explainer,
    tmp_path,
):
    db1 = ExplainerDashboard(precalculated_rf_classifier_explainer, name="db1")
    db2 = ExplainerDashboard(precalculated_rf_regression_explainer, name="db2")

    explainer_path = tmp_path / "db2.joblib"
    db2_yaml = tmp_path / "db2.yaml"
    db2.explainer.dump(explainer_path)
    db2.to_yaml(db2_yaml, explainerfile=str(explainer_path))

    hub = ExplainerHub(
        [db1],
        users_file=str(Path.cwd() / "tests" / "test_assets" / "users.yaml"),
        add_dashboard_route=True,
        add_dashboard_pattern=str(tmp_path / "{}.yaml"),
    )
    hub.app.config["TESTING"] = True
    client = hub.app.test_client()

    # Simulate the app already serving requests before dynamic add.
    response = client.get("/")
    assert response.status_code in (200, 302)

    response = client.get("/add_dashboard/db2")
    assert response.status_code == 302
    assert "db2" in hub.dashboard_names
