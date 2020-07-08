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
        """Composite of multiple classifier related components: 
            - precision graph
            - confusion matrix
            - lift curve
            - classification graph
            - roc auc graph
            - pr auc graph

        Args:
            explainer (Explainer): explainer object constructed with either
                        ClassifierExplainer() or RegressionExplainer()
            title (str, optional): Title of tab or page. Defaults to 
                        "Decision Trees".
            header_mode (str, optional): {"standalone", "hidden" or "none"}. 
                        Defaults to "none".
            name (str, optional): unique name to add to Component elements. 
                        If None then random uuid is generated to make sure 
                        it's unique. Defaults to None.
            bin_size (float, optional): bin_size for precision plot. Defaults to 0.1.
            quantiles (int, optional): number of quantiles for precision plot. Defaults to 10.
            cutoff (float, optional): initial cutoff. Defaults to 0.5.
        """
        super().__init__(explainer, title, header_mode, name)

        self.precision = PrecisionComponent(explainer)
        self.confusionmatrix = ConfusionMatrixComponent(explainer)
        self.liftcurve = LiftCurveComponent(explainer)
        self.classification = ClassificationComponent(explainer)
        self.rocauc = RocAucComponent(explainer)
        self.prauc = PrAucComponent(explainer)

        self.cutoffpercentile = CutoffPercentileComponent(explainer)
        self.cutoffconnector = CutoffConnector(self.cutoffpercentile,
                [self.precision, self.confusionmatrix, self.liftcurve, 
                 self.classification, self.rocauc, self.prauc])

        self.register_components(
            self.precision, self.confusionmatrix, self.liftcurve,
            self.classification, self.rocauc, self.prauc, self.cutoffpercentile,
            self.cutoffconnector)

    def _layout(self):
        return dbc.Container([
            dbc.Row([dbc.Col([html.H2('Model Performance:')])]),
            dbc.Row([
                dbc.Col([
                    self.cutoffpercentile.layout(),
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
        """Composite for displaying multiple regression related graphs:

        - predictions vs actual plot
        - residual plot
        - residuals vs feature

        Args:
            explainer (Explainer): explainer object constructed with either
                        ClassifierExplainer() or RegressionExplainer()
            title (str, optional): Title of tab or page. Defaults to 
                        "Regression Stats".
            header_mode (str, optional): {"standalone", "hidden" or "none"}. 
                        Defaults to "none".
            name (str, optional): unique name to add to Component elements. 
                        If None then random uuid is generated to make sure 
                        it's unique. Defaults to None.
            logs (bool, optional): Use log axis. Defaults to False.
            pred_or_actual (str, optional): plot residuals vs predictions 
                        or vs y (actual). Defaults to "vs_pred".
            ratio (bool, optional): Use residual ratios. Defaults to False.
            col ({str, int}, optional): Feature to use for residuals plot. Defaults to None.
        """
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
    def __init__(self, explainer, title="Individual Predictions",
                        header_mode="none", name=None):
        """Composite for a number of component that deal with individual predictions:

        - random index selector
        - prediction summary
        - shap contributions graph
        - shap contribution table
        - pdp graph

        Args:
            explainer (Explainer): explainer object constructed with either
                        ClassifierExplainer() or RegressionExplainer()
            title (str, optional): Title of tab or page. Defaults to 
                        "Individual Predictions".
            header_mode (str, optional): {"standalone", "hidden" or "none"}. 
                        Defaults to "none".
            name (str, optional): unique name to add to Component elements. 
                        If None then random uuid is generated to make sure 
                        it's unique. Defaults to None.
        """
        super().__init__(explainer, title, header_mode, name)

        if self.explainer.is_classifier:
            self.index = ClassifierRandomIndexComponent(explainer)
        elif self.explainer.is_regression:
            self.index = RegressionRandomIndexComponent(explainer)
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
    def __init__(self, explainer, title='Feature Dependence',
                    header_mode="none", name=None,
                    depth=None, cats=True):
        """Composite of ShapSummary and ShapDependence component

        Args:
            explainer (Explainer): explainer object constructed with either
                        ClassifierExplainer() or RegressionExplainer()
            title (str, optional): Title of tab or page. Defaults to 
                        "Feature Dependence".
            header_mode (str, optional): {"standalone", "hidden" or "none"}. 
                        Defaults to "none".
            name (str, optional): unique name to add to Component elements. 
                        If None then random uuid is generated to make sure 
                        it's unique. Defaults to None.
            depth (int, optional): Number of features to display. Defaults to None.
            cats (bool, optional): Group categorical features. Defaults to True.
        """
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
    def __init__(self, explainer, title='Feature Interactions',
                    header_mode="none", name=None,
                    depth=None, cats=True):
        """Composite of InteractionSummaryComponent and InteractionDependenceComponent

        Args:
            explainer (Explainer): explainer object constructed with either
                        ClassifierExplainer() or RegressionExplainer()
            title (str, optional): Title of tab or page. Defaults to 
                        "Feature Interactions".
            header_mode (str, optional): {"standalone", "hidden" or "none"}. 
                        Defaults to "none".
            name (str, optional): unique name to add to Component elements. 
                        If None then random uuid is generated to make sure 
                        it's unique. Defaults to None.
            depth (int, optional): Initial number of features to display. Defaults to None.
            cats (bool, optional): Initally group cats. Defaults to True.
        """
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
        """Composite of decision tree related components:
        
        - individual decision trees barchart
        - decision path table
        - deciion path graph

        Args:
            explainer (Explainer): explainer object constructed with either
                        ClassifierExplainer() or RegressionExplainer()
            title (str, optional): Title of tab or page. Defaults to 
                        "Decision Trees".
            header_mode (str, optional): {"standalone", "hidden" or "none"}. 
                        Defaults to "none".
            name (str, optional): unique name to add to Component elements. 
                        If None then random uuid is generated to make sure 
                        it's unique. Defaults to None.
        """
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