from dash import html

from explainerdashboard import ExplainerDashboard
from explainerdashboard.dashboard_methods import ExplainerComponent


class CustomTab(ExplainerComponent):
    def __init__(self, explainer, title="Custom Tab", name=None):
        super().__init__(explainer, title, name)

    def layout(self):
        return html.Div([html.H3("Custom content")])


def test_save_html_includes_custom_tab(rf_classifier_explainer):
    dashboard = ExplainerDashboard(
        rf_classifier_explainer, tabs=[CustomTab, "importances"]
    )
    html_out = dashboard.save_html()
    assert "Custom Tab" in html_out
