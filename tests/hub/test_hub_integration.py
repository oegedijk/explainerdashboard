import pytest

pytestmark = pytest.mark.selenium

@pytest.fixture(scope="session")
def explainer_hub_client(explainer_hub):
    explainer_hub.app.config["TESTING"] = True
    explainer_hub.app.config["WTF_CSRF_CHECK_DEFAULT"] = False
    client = explainer_hub.app.test_client()
    _ctx = explainer_hub.app.test_request_context()
    _ctx.push()

    yield client

    if _ctx is not None:
        _ctx.pop()

def test_explainer_hub_client(explainer_hub_client):
    with explainer_hub_client:
        data = {"username": "user", "password": "password", "next": "/"}       
        response = explainer_hub_client.post("/login/", data=data)
        assert (response.status_code == 200)

