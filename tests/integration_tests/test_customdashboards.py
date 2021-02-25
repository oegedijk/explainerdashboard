import dash
from sklearn.ensemble import RandomForestClassifier

from explainerdashboard import *
from explainerdashboard.datasets import *
from explainerdashboard.custom import *

def get_classification_explainer():
    X_train, y_train, X_test, y_test = titanic_survive()
    model = RandomForestClassifier(n_estimators=50, max_depth=10).fit(X_train, y_train)
    explainer = ClassifierExplainer(
                            model, X_test, y_test, 
                            cats=['Sex', 'Deck', 'Embarked'],
                            labels=['Not survived', 'Survived'])
    explainer.calculate_properties()
    return explainer


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

def test_classification_dashboard(dash_duo):
    explainer = get_classification_explainer()
    custom_instance = CustomDashboard(explainer, name='custom')
    db = ExplainerDashboard(explainer, [custom_instance, CustomDashboard], title="testing", responsive=False)
    dash_duo.start_server(db.app)
    dash_duo.wait_for_text_to_equal("h1", "testing", timeout=30)
    assert dash_duo.get_logs() == [], "browser console should contain no error"
