import pytest

from explainerdashboard import ExplainerDashboard
from explainerdashboard.custom import *

pytestmark = pytest.mark.selenium

class CustomDashboard(ExplainerComponent):
    def __init__(self, explainer, title="Custom Dashboard", name=None):
        super().__init__(explainer)
        self.confusion = ConfusionMatrixComponent(explainer, self.name+'0',
                            hide_selector=True, hide_percentage=True, cutoff=0.75)
        self.contrib = ShapContributionsGraphComponent(explainer, self.name+'1',
                            hide_selector=True, hide_depth=True, hide_sort=True)
        
    def layout(self):
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("Custom Demonstration:"),
                    html.H3("How to build your own layout by re-using ExplainerComponents.")
                ])
            ]),
            dbc.Row([
                dbc.Col([
                    self.confusion.layout(),
                ]),
                dbc.Col([
                    self.contrib.layout(),
                ])
            ])
        ])

def test_custom_classification_dashboard(dash_duo, precalculated_rf_classifier_explainer):
    custom_instance = CustomDashboard(precalculated_rf_classifier_explainer, name='custom')
    db = ExplainerDashboard(precalculated_rf_classifier_explainer, [custom_instance, CustomDashboard], title="testing", responsive=False)
    dash_duo.start_server(db.app)
    dash_duo.wait_for_text_to_equal("h1", "testing", timeout=30)
    assert dash_duo.get_logs() == [], "browser console should contain no error"
