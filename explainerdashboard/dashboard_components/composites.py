__all__ = [
    'ClassifierModelStatsComposite',
    'RegressionModelStatsComposite',
    'IndividualPredictionsComposite',
    'ShapDependenceComposite',
    'ShapInteractionsComposite',
    'DecisionTreesComposite',
    'WhatIfComposite',
]

import dash_bootstrap_components as dbc
import dash_html_components as html

from ..dashboard_methods import *
from .classifier_components import *
from .regression_components import *
from .overview_components import *
from .connectors import *
from .shap_components import *
from .decisiontree_components import *


class ClassifierModelStatsComposite(ExplainerComponent):
    def __init__(self, explainer, title="Classification Stats", name=None,
                    hide_title=False, hide_selector=True, pos_label=None,
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
            name (str, optional): unique name to add to Component elements. 
                        If None then random uuid is generated to make sure 
                        it's unique. Defaults to None.
            hide_title (bool, optional): hide title. Defaults to False.          
            hide_selector (bool, optional): hide all pos label selectors. Defaults to True.
            pos_label ({int, str}, optional): initial pos label. Defaults to explainer.pos_label
            bin_size (float, optional): bin_size for precision plot. Defaults to 0.1.
            quantiles (int, optional): number of quantiles for precision plot. Defaults to 10.
            cutoff (float, optional): initial cutoff. Defaults to 0.5.
        """
        super().__init__(explainer, title, name)

        self.summary = ClassifierModelSummaryComponent(explainer, hide_selector=hide_selector, pos_label=pos_label)
        self.precision = PrecisionComponent(explainer, hide_selector=hide_selector, pos_label=pos_label)
        self.confusionmatrix = ConfusionMatrixComponent(explainer, hide_selector=hide_selector, pos_label=pos_label)
        self.cumulative_precision = CumulativePrecisionComponent(explainer, hide_selector=hide_selector, pos_label=pos_label)
        self.liftcurve = LiftCurveComponent(explainer, hide_selector=hide_selector, pos_label=pos_label)
        self.classification = ClassificationComponent(explainer, hide_selector=hide_selector, pos_label=pos_label)
        self.rocauc = RocAucComponent(explainer, hide_selector=hide_selector, pos_label=pos_label)
        self.prauc = PrAucComponent(explainer, hide_selector=hide_selector, pos_label=pos_label)

        self.cutoffpercentile = CutoffPercentileComponent(explainer, hide_selector=hide_selector, pos_label=pos_label)
        self.cutoffconnector = CutoffConnector(self.cutoffpercentile,
                [self.summary, self.precision, self.confusionmatrix, self.liftcurve, 
                 self.cumulative_precision, self.classification, self.rocauc, self.prauc])

        self.register_components()

    def layout(self):
        return html.Div([
            dbc.Row([
                make_hideable(
                    dbc.Col([
                     html.H2('Model Performance:')]), hide=self.hide_title),
            ]),
            dbc.Row([
                dbc.Col([
                    self.cutoffpercentile.layout(),
                ])
            ]),
            dbc.Row([
                dbc.Col([
                    self.summary.layout()
                ], md=6, align="start"),
                dbc.Col([
                    self.confusionmatrix.layout()
                ], md=6, align="start"),              
            ]),
            dbc.Row([    
                dbc.Col([
                    self.precision.layout()
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
            dbc.Row([
                dbc.Col([
                    self.liftcurve.layout()         
                ], md=6, align="start"),
                dbc.Col([
                    self.cumulative_precision.layout()
                ], md=6, align="start"), 
            ]), 
        ])


class RegressionModelStatsComposite(ExplainerComponent):
    def __init__(self, explainer, title="Regression Stats", name=None,
                    hide_title=False,
                    logs=False, pred_or_actual="vs_pred", residuals='difference',
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
            name (str, optional): unique name to add to Component elements. 
                        If None then random uuid is generated to make sure 
                        it's unique. Defaults to None.
            hide_title (bool, optional): hide title. Defaults to False.
            logs (bool, optional): Use log axis. Defaults to False.
            pred_or_actual (str, optional): plot residuals vs predictions 
                        or vs y (actual). Defaults to "vs_pred".
            residuals (str, {'difference', 'ratio', 'log-ratio'} optional): 
                    How to calcualte residuals. Defaults to 'difference'.
            col ({str, int}, optional): Feature to use for residuals plot. Defaults to None.
        """
        super().__init__(explainer, title, name)
     
        assert pred_or_actual in ['vs_actual', 'vs_pred'], \
            "pred_or_actual should be 'vs_actual' or 'vs_pred'!"

        self.preds_vs_actual = PredictedVsActualComponent(explainer, logs=logs)
        self.modelsummary = RegressionModelSummaryComponent(explainer)
        self.residuals = ResidualsComponent(explainer, 
                            pred_or_actual=pred_or_actual, residuals=residuals)
        self.residuals_vs_col = ResidualsVsColComponent(explainer, 
                                    col=col, residuals=residuals)
        self.y_vs_col = ActualVsColComponent(explainer, col=col)
        self.preds_vs_col = PredsVsColComponent(explainer, col=col)
        self.register_components()

    def layout(self):
        return html.Div([
            dbc.Row([
                make_hideable(
                    dbc.Col([
                        html.H2('Model Performance:')]), hide=self.hide_title)
            ]),
            dbc.Row([
                dbc.Col([
                    self.modelsummary.layout()     
                ], md=6),
                dbc.Col([
                    self.preds_vs_actual.layout()
                ], md=6),
            ], align="start"),
            dbc.Row([
                dbc.Col([
                    self.residuals.layout()
                ], md=6),
                dbc.Col([
                    self.residuals_vs_col.layout()
                ], md=6),
            ]),
            dbc.Row([
                dbc.Col([
                    self.y_vs_col.layout()
                ], md=6),
                dbc.Col([
                    self.preds_vs_col.layout()
                ], md=6),
            ])
        ])


class IndividualPredictionsComposite(ExplainerComponent):
    def __init__(self, explainer, title="Individual Predictions", name=None,
                        hide_title=False, hide_selector=True):
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
            name (str, optional): unique name to add to Component elements. 
                        If None then random uuid is generated to make sure 
                        it's unique. Defaults to None.
            hide_title (bool, optional): hide title. Defaults to False.
            hide_selector(bool, optional): hide all pos label selectors. Defaults to True.
        """
        super().__init__(explainer, title, name)

        if self.explainer.is_classifier:
            self.index = ClassifierRandomIndexComponent(
                            explainer, hide_selector=hide_selector)
        elif self.explainer.is_regression:
            self.index = RegressionRandomIndexComponent(explainer)
        self.summary = PredictionSummaryComponent(
                            explainer, hide_selector=hide_selector)
        self.contributions = ShapContributionsGraphComponent(
                                explainer, hide_selector=hide_selector)
        self.pdp = PdpComponent(
                        explainer, hide_selector=hide_selector)
        self.contributions_list = ShapContributionsTableComponent(
                                        explainer, hide_selector=hide_selector)

        self.index_connector = IndexConnector(self.index, 
                [self.summary, self.contributions, self.pdp, self.contributions_list])

        self.register_components()

    def layout(self):
        return dbc.Container([
                dbc.Row([
                    dbc.Col([
                        self.index.layout()
                    ], md=6),
                    dbc.Col([
                        self.summary.layout()
                    ], md=6)
                ]),
                dbc.Row([
                    dbc.Col([
                        self.contributions.layout()
                    ], md=6),
                    dbc.Col([
                        self.pdp.layout()
                    ], md=6),
                ]),
                dbc.Row([
                    dbc.Col([
                        self.contributions_list.layout()
                    ], md=6),
                    dbc.Col([
                        html.Div([]),
                    ], md=6),
                ])
        ], fluid=True)


class WhatIfComposite(ExplainerComponent):
    def __init__(self, explainer, title="What if...", name=None,
                        hide_title=False, hide_selector=True):
        """Composite for the whatif component:

        Args:
            explainer (Explainer): explainer object constructed with either
                        ClassifierExplainer() or RegressionExplainer()
            title (str, optional): Title of tab or page. Defaults to 
                        "Individual Predictions".
            name (str, optional): unique name to add to Component elements. 
                        If None then random uuid is generated to make sure 
                        it's unique. Defaults to None.
            hide_title (bool, optional): hide title. Defaults to False.
            hide_selector(bool, optional): hide all pos label selectors. Defaults to True.
        """
        super().__init__(explainer, title, name)

        if self.explainer.is_classifier:
            self.index = ClassifierRandomIndexComponent(
                            explainer, hide_selector=hide_selector)
        elif self.explainer.is_regression:
            self.index = RegressionRandomIndexComponent(explainer)

        self.whatif = WhatIfComponent(explainer, 
                            hide_title=True, hide_index=True, hide_selector=hide_selector)

        self.index_connector = IndexConnector(self.index, [self.whatif])

        self.register_components()

    def layout(self):
        return dbc.Container([
                dbc.Row([
                    make_hideable(
                        dbc.Col([
                            html.H1("What if...")
                        ]), hide=self.hide_title),
                ]),
                dbc.Row([
                    dbc.Col([
                        self.index.layout()
                    ]),
                ]),
                dbc.Row([
                    dbc.Col([
                        self.whatif.layout()
                    ]),
                ]),
        ], fluid=True)


class ShapDependenceComposite(ExplainerComponent):
    def __init__(self, explainer, title='Feature Dependence', name=None,
                    hide_selector=True,
                    depth=None, cats=True):
        """Composite of ShapSummary and ShapDependence component

        Args:
            explainer (Explainer): explainer object constructed with either
                        ClassifierExplainer() or RegressionExplainer()
            title (str, optional): Title of tab or page. Defaults to 
                        "Feature Dependence".
            name (str, optional): unique name to add to Component elements. 
                        If None then random uuid is generated to make sure 
                        it's unique. Defaults to None.
            hide_selector (bool, optional): hide all pos label selectors. Defaults to True.
            depth (int, optional): Number of features to display. Defaults to None.
            cats (bool, optional): Group categorical features. Defaults to True.
        """
        super().__init__(explainer, title, name)
        
        self.shap_summary = ShapSummaryComponent(
                    self.explainer, 
                    hide_selector=hide_selector, 
                    depth=depth, cats=cats)
        self.shap_dependence = ShapDependenceComponent(
                    self.explainer, 
                    hide_selector=hide_selector, hide_cats=True, 
                    cats=cats)
        self.connector = ShapSummaryDependenceConnector(
                    self.shap_summary, self.shap_dependence)
        self.register_components()

    def layout(self):
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
    def __init__(self, explainer, title='Feature Interactions', name=None,
                    hide_selector=True,
                    depth=None, cats=True):
        """Composite of InteractionSummaryComponent and InteractionDependenceComponent

        Args:
            explainer (Explainer): explainer object constructed with either
                        ClassifierExplainer() or RegressionExplainer()
            title (str, optional): Title of tab or page. Defaults to 
                        "Feature Interactions".
            name (str, optional): unique name to add to Component elements. 
                        If None then random uuid is generated to make sure 
                        it's unique. Defaults to None.
            hide_selector (bool, optional): hide all pos label selectors. Defaults to True.
            depth (int, optional): Initial number of features to display. Defaults to None.
            cats (bool, optional): Initally group cats. Defaults to True.
        """
        super().__init__(explainer, title, name)

        self.interaction_summary = InteractionSummaryComponent(
                explainer, hide_selector=hide_selector, depth=depth, cats=cats)
        self.interaction_dependence = InteractionDependenceComponent(
                explainer, hide_selector=hide_selector, cats=cats, hide_cats=True)
        self.connector = InteractionSummaryDependenceConnector(
            self.interaction_summary, self.interaction_dependence)
        self.register_components()
        
    def layout(self):
        return dbc.Container([
                dbc.Row([
                    dbc.Col([
                        self.interaction_summary.layout()
                    ], md=6),
                    dbc.Col([
                        self.interaction_dependence.layout()
                    ], md=6),
                ]) 
        ], fluid=True)


class DecisionTreesComposite(ExplainerComponent):
    def __init__(self, explainer, title="Decision Trees", name=None,
                    hide_selector=True):
        """Composite of decision tree related components:
        
        - individual decision trees barchart
        - decision path table
        - deciion path graph

        Args:
            explainer (Explainer): explainer object constructed with either
                        RandomForestClassifierExplainer() or RandomForestRegressionExplainer()
            title (str, optional): Title of tab or page. Defaults to 
                        "Decision Trees".
            name (str, optional): unique name to add to Component elements. 
                        If None then random uuid is generated to make sure 
                        it's unique. Defaults to None.
            hide_selector (bool, optional): hide all pos label selectors. Defaults to True.
        """
        super().__init__(explainer, title, name)
        
        self.trees = DecisionTreesComponent(explainer, hide_selector=hide_selector)
        self.decisionpath_table = DecisionPathTableComponent(explainer, hide_selector=hide_selector)

        if explainer.is_classifier:
            self.index = ClassifierRandomIndexComponent(explainer, hide_selector=hide_selector)
        elif explainer.is_regression:
            self.index = RegressionRandomIndexComponent(explainer)

        self.decisionpath_graph = DecisionPathGraphComponent(explainer, hide_selector=hide_selector)

        self.index_connector = IndexConnector(self.index, 
            [self.trees, self.decisionpath_table, self.decisionpath_graph])
        self.highlight_connector = HighlightConnector(self.trees, 
            [self.decisionpath_table, self.decisionpath_graph])

        self.register_components()
        
    def layout(self):
        return html.Div([
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
