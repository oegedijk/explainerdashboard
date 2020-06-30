__all__ = [
    'ClassifierModelStatsComposite',
    'RegressionModelStatsComposite',
    'IndividualPredictionsComposite',
    'ShapDependenceComposite',
    'ShapInteractionsComposite',
    'DecisionTreesComposite',
]

import dash_bootstrap_components as dbc
import dash_html_components as html

from.dashboard_methods import *
from .classifier_components import *
from .regression_components import *
from .overview_components import *
from .connectors import *
from .shap_components import *
from .decisiontree_components import *


class ClassifierModelStatsComposite(ExplainerComponent):
    def __init__(self, explainer, title="Classification Stats", 
                    header_mode="none", name=None,
                    bin_size=0.1, quantiles=10, cutoff=0.5):
        super().__init__(explainer, title, header_mode, name)

        self.precision = PrecisionComponent(explainer)
        self.confusionmatrix = ConfusionMatrixComponent(explainer)
        self.liftcurve = LiftCurveComponent(explainer)
        self.classification = ClassificationComponent(explainer)
        self.rocauc = RocAucComponent(explainer)
        self.prauc = PrAucComponent(explainer)

        self.cutoffconnector = CutoffConnector(explainer,
            cutoff_components=[self.precision, self.confusionmatrix, 
                self.liftcurve, self.classification, self.rocauc, self.prauc])

        self.register_components(
            self.precision, self.confusionmatrix, self.liftcurve,
            self.classification, self.rocauc, self.prauc,
            self.cutoffconnector)

    def _layout(self):
        return dbc.Container([
            dbc.Row([dbc.Col([html.H2('Model Performance:')])]),
            dbc.Row([
                dbc.Col([
                    self.cutoffconnector.layout(),
                ])
            ]),
            dbc.Row([
                dbc.Col([
                    self.precision.layout()
                ], md=6, align="start"),
                dbc.Col([
                    self.confusionmatrix.layout()
                ], md=6, align="start"),              
            ]),
            dbc.Row([
                dbc.Col([
                    self.liftcurve.layout()         
                ], md=6, align="start"),
                dbc.Col([
                    self.classification.layout()
                ], md=6, align="start"),
            ]),
            dbc.Row([    
                dbc.Col([
                    self.rocauc.layout()
                ], md=6),
                dbc.Col([
                    self.prauc.layout()
                ], md=6),
            ]),
        ], fluid=True)


class RegressionModelStatsComposite(ExplainerComponent):
    def __init__(self, explainer, title="Regression Stats", 
                    header_mode="none", name=None,
                    logs=False, pred_or_actual="vs_pred", ratio=False,
                    col=None):
        super().__init__(explainer, title, header_mode, name)
     
        assert pred_or_actual in ['vs_actual', 'vs_pred'], \
            "pred_or_actual should be 'vs_actual' or 'vs_pred'!"

        self.preds_vs_actual = PredictedVsActualComponent(explainer, logs=logs)
        self.modelsummary = RegressionModelSummaryComponent(explainer)
        self.residuals = ResidualsComponent(explainer, 
                            pred_or_actual=pred_or_actual, ratio=ratio)
        self.residuals_vs_col = ResidualsVsColComponent(explainer, 
                                    col=col, ratio=ratio)
        self.register_components([self.preds_vs_actual, self.modelsummary,
                    self.residuals, self.residuals_vs_col])

    def _layout(self):
        return dbc.Container([
            dbc.Row([dbc.Col([html.H2('Model Performance:')])]),
            dbc.Row([
                dbc.Col([
                    self.preds_vs_actual.layout()
                ], md=6),
                dbc.Col([
                    self.modelsummary.layout()     
                ], md=6),
            ], align="start"),
            dbc.Row([
                dbc.Col([
                    self.residuals.layout()
                ], md=6),
                dbc.Col([
                    self.residuals_vs_col.layout()
                ], md=6),
            ])
        ], fluid=True)


class IndividualPredictionsComposite(ExplainerComponent):
    def __init__(self, explainer, title="Individual Contributions",
                        header_mode="none", name=None):
        super().__init__(explainer, title, header_mode, name)

        self.index = ClassifierRandomIndexComponent(explainer)
        self.summary = PredictionSummaryComponent(explainer)
        self.contributions = ShapContributionsGraphComponent(explainer)
        self.pdp = PdpComponent(explainer)
        self.contributions_list = ShapContributionsTableComponent(explainer)

        self.index_connector = IndexConnector(self.index, 
                [self.summary, self.contributions, self.pdp, self.contributions_list])

        self.register_components(self.index, self.summary, self.contributions, self.pdp, self.contributions_list, self.index_connector)

    def _layout(self):
        return html.Div([
            dbc.Container([
                dbc.Row([
                    dbc.Col([
                        self.index.layout()
                    ]),
                    dbc.Col([
                        self.summary.layout()
                    ])
                ]),
                dbc.Row([
                    dbc.Col([
                        self.contributions.layout()
                    ]),
                    dbc.Col([
                        self.pdp.layout()
                    ]),
                ]),
                dbc.Row([
                    dbc.Col([
                        self.contributions_list.layout()
                    ]),
                    dbc.Col([
                        html.Div([]),
                    ]),
                ])
            ], fluid=True),
        ])


class ShapDependenceComposite(ExplainerComponent):
    def __init__(self, explainer, title='Shap Dependence',
                    header_mode="none", name=None,
                    depth=None, cats=True):
        super().__init__(explainer, title, header_mode, name)
        
        self.shap_summary = ShapSummaryComponent(self.explainer, depth=depth, cats=cats)
        self.shap_dependence = ShapDependenceComponent(
                                    self.explainer, hide_cats=True, cats=cats)
        self.connector = ShapSummaryDependenceConnector(self.shap_summary, self.shap_dependence)
        self.register_components(self.shap_summary, self.shap_dependence, self.connector)

    def _layout(self):
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    self.shap_summary.layout()
                ], md=6),
                dbc.Col([
                    self.shap_dependence.layout()
                ], md=6),
                ]),
            ],  fluid=True)


class ShapInteractionsComposite(ExplainerComponent):
    def __init__(self, explainer, title='Shap Interactions',
                    header_mode="none", name=None,
                    depth=None, cats=True):
        super().__init__(explainer, title, header_mode, name)

        self.interaction_summary = InteractionSummaryComponent(
                explainer, depth=depth, cats=cats)
        self.interaction_dependence = InteractionDependenceComponent(
                explainer, cats=cats)
        self.connector = InteractionSummaryDependenceConnector(
            self.interaction_summary, self.interaction_dependence)
        self.register_components(
            self.interaction_summary, self.interaction_dependence, self.connector)
        
    def _layout(self):
        return html.Div([
            dbc.Row([
                dbc.Col([
                    self.interaction_summary.layout()
                ], width=6),
                dbc.Col([
                    self.interaction_dependence.layout()
                ], width=6),
            ])
        ])


class DecisionTreesComposite(ExplainerComponent):
    def __init__(self, explainer, title="Decision Trees",
                    header_mode="none", name=None):
        super().__init__(explainer, title, header_mode, name)
        
        self.index = ClassifierRandomIndexComponent(explainer)
        self.trees = DecisionTreesComponent(explainer)
        self.decisionpath_table = DecisionPathTableComponent(explainer)
        self.decisionpath_graph = DecisionPathGraphComponent(explainer)

        self.index_connector = IndexConnector(self.index, 
            [self.trees, self.decisionpath_table, self.decisionpath_graph])
        self.highlight_connector = HighlightConnector(self.trees, 
            [self.decisionpath_table, self.decisionpath_graph])

        self.register_components(self.index, self.trees, 
                self.decisionpath_table, self.decisionpath_graph, 
                self.index_connector, self.highlight_connector)

    def _layout(self):
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    self.index.layout(),
                ])
            ]),
            dbc.Row([
                dbc.Col([
                    self.trees.layout(),
                ])
            ]),
            dbc.Row([
                dbc.Col([
                    self.decisionpath_table.layout()
                ])
            ]),
            dbc.Row([
                dbc.Col([
                    self.decisionpath_graph.layout()
                ])
            ]),
        ])