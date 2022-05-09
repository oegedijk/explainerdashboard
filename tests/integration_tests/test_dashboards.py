
import pytest

from explainerdashboard.dashboards import ExplainerDashboard

pytestmark = pytest.mark.selenium

def test_classification_dashboard(dash_duo, precalculated_rf_classifier_explainer):
    db = ExplainerDashboard(precalculated_rf_classifier_explainer, title="testing", responsive=False)
    html = db.to_html()
    assert html.startswith('\n<!DOCTYPE html>\n<html'), "failed to generate dashboard to_html"
    
    dash_duo.start_server(db.app)
    dash_duo.wait_for_text_to_equal("h1", "testing", timeout=30)
    assert dash_duo.get_logs() == [], "browser console should contain no error"


def test_regression_dashboard(dash_duo, precalculated_rf_regression_explainer):
    db = ExplainerDashboard(precalculated_rf_regression_explainer, title="testing", responsive=False)
    html = db.to_html()
    assert html.startswith('\n<!DOCTYPE html>\n<html'), "failed to generate dashboard to_html"

    dash_duo.start_server(db.app)
    dash_duo.wait_for_text_to_equal("h1", "testing", timeout=20)
    assert dash_duo.get_logs() == [], "browser console should contain no error"

def test_simple_classification_dashboard(dash_duo, precalculated_rf_classifier_explainer):
    db = ExplainerDashboard(precalculated_rf_classifier_explainer, title="testing", responsive=False, simple=True)
    html = db.to_html()
    assert html.startswith('\n<!DOCTYPE html>\n<html'), "failed to generate dashboard to_html"

    dash_duo.start_server(db.app)
    dash_duo.wait_for_text_to_equal("h1", "testing", timeout=20)
    assert dash_duo.get_logs() == [], "browser console should contain no error"


def test_simple_regression_dashboard(dash_duo, precalculated_rf_regression_explainer):
    db = ExplainerDashboard(precalculated_rf_regression_explainer, title="testing", responsive=False, simple=True)
    html = db.to_html()
    assert html.startswith('\n<!DOCTYPE html>\n<html'), "failed to generate dashboard to_html"

    dash_duo.start_server(db.app)
    dash_duo.wait_for_text_to_equal("h1", "testing", timeout=20)
    assert dash_duo.get_logs() == [], "browser console should contain no error"


def test_multiclass_dashboard(dash_duo, precalculated_rf_multiclass_explainer):
    db = ExplainerDashboard(precalculated_rf_multiclass_explainer, title="testing", responsive=False)
    html = db.to_html()
    assert html.startswith('\n<!DOCTYPE html>\n<html'), "failed to generate dashboard to_html"

    dash_duo.start_server(db.app)
    dash_duo.wait_for_text_to_equal("h1", "testing", timeout=30)
    assert dash_duo.get_logs() == [], "browser console should contain no error"


def test_xgboost_classification_dashboard(dash_duo, precalculated_xgb_classifier_explainer):
    db = ExplainerDashboard(precalculated_xgb_classifier_explainer, title="testing", responsive=False)
    html = db.to_html()
    assert html.startswith('\n<!DOCTYPE html>\n<html'), "failed to generate dashboard to_html"

    dash_duo.start_server(db.app)
    dash_duo.wait_for_text_to_equal("h1", "testing", timeout=30)
    assert dash_duo.get_logs() == [], "browser console should contain no error"


def test_xgboost_regression_dashboard(dash_duo, precalculated_xgb_regression_explainer):
    db = ExplainerDashboard(precalculated_xgb_regression_explainer, title="testing", responsive=False)
    html = db.to_html()
    assert html.startswith('\n<!DOCTYPE html>\n<html'), "failed to generate dashboard to_html"

    dash_duo.start_server(db.app)
    dash_duo.wait_for_text_to_equal("h1", "testing", timeout=30)
    assert dash_duo.get_logs() == [], "browser console should contain no error"


def test_xgboost_multiclass_dashboard(dash_duo, precalculated_xgb_multiclass_explainer):
    db = ExplainerDashboard(precalculated_xgb_multiclass_explainer, title="testing", responsive=False)
    html = db.to_html()
    assert html.startswith('\n<!DOCTYPE html>\n<html'), "failed to generate dashboard to_html"

    dash_duo.start_server(db.app)
    dash_duo.wait_for_text_to_equal("h1", "testing", timeout=30)
    assert dash_duo.get_logs() == [], "browser console should contain no error"


def test_classification_dashboard_no_y(dash_duo,  precalculated_rf_classifier_explainer_no_y):
    db = ExplainerDashboard(precalculated_rf_classifier_explainer_no_y, title="testing", responsive=False)
    html = db.to_html()
    assert html.startswith('\n<!DOCTYPE html>\n<html'), "failed to generate dashboard to_html"

    dash_duo.start_server(db.app)
    dash_duo.wait_for_text_to_equal("h1", "testing", timeout=30)
    assert dash_duo.get_logs() == [], "browser console should contain no error"


def test_regression_dashboard_no_y(dash_duo, precalculated_rf_regression_explainer_no_y):
    db = ExplainerDashboard(precalculated_rf_regression_explainer_no_y, title="testing", responsive=False)
    html = db.to_html()
    assert html.startswith('\n<!DOCTYPE html>\n<html'), "failed to generate dashboard to_html"

    dash_duo.start_server(db.app)
    dash_duo.wait_for_text_to_equal("h1", "testing", timeout=30)
    assert dash_duo.get_logs() == [], "browser console should contain no error"


def test_multiclass_dashboard_no_y(dash_duo, precalculated_rf_multiclass_explainer_no_y):
    db = ExplainerDashboard(precalculated_rf_multiclass_explainer_no_y, title="testing", responsive=False)
    html = db.to_html()
    assert html.startswith('\n<!DOCTYPE html>\n<html'), "failed to generate dashboard to_html"

    dash_duo.start_server(db.app)
    dash_duo.wait_for_text_to_equal("h1", "testing", timeout=30)
    assert dash_duo.get_logs() == [], "browser console should contain no error"


def test_catboost_classification_dashboard(dash_duo, precalculated_catboost_classifier_explainer):
    db = ExplainerDashboard(precalculated_catboost_classifier_explainer, title="testing", responsive=False)
    html = db.to_html()
    assert html.startswith('\n<!DOCTYPE html>\n<html'), "failed to generate dashboard to_html"

    dash_duo.start_server(db.app)
    dash_duo.wait_for_text_to_equal("h1", "testing", timeout=30)
    assert dash_duo.get_logs() == [], "browser console should contain no error"


def test_catboost_regression_dashboard(dash_duo, precalculated_catboost_regression_explainer):
    db = ExplainerDashboard(precalculated_catboost_regression_explainer, title="testing", responsive=False)
    html = db.to_html()
    assert html.startswith('\n<!DOCTYPE html>\n<html'), "failed to generate dashboard to_html"

    dash_duo.start_server(db.app)
    dash_duo.wait_for_text_to_equal("h1", "testing", timeout=30)
    assert dash_duo.get_logs() == [], "browser console should contain no error"