__all__ = [
    'ImportancesComposite',
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

from ..explainers import RandomForestExplainer, XGBExplainer
from ..dashboard_methods import *
from .classifier_components import *
from .regression_components import *
from .overview_components import *
from .connectors import *
from .shap_components import *
from .decisiontree_components import *


class ImportancesComposite(ExplainerComponent):
    def __init__(self, explainer, title="Feature Importances", name=None,
                    hide_selector=True, **kwargs):
        """Overview tab of feature importances

        Can show both permutation importances and mean absolute shap values.

        Args:
            explainer (Explainer): explainer object constructed with either
                        ClassifierExplainer() or RegressionExplainer()
            title (str, optional): Title of tab or page. Defaults to 
                        "Feature Importances".
            name (str, optional): unique name to add to Component elements. 
                        If None then random uuid is generated to make sure 
                        it's unique. Defaults to None.
            hide_selector (bool, optional): hide the post label selector. 
                Defaults to True.
        """
        super().__init__(explainer, title, name)

        self.importances = ImportancesComponent(
                explainer, hide_selector=hide_selector, **kwargs)
        self.register_components()

    def layout(self):
        return html.Div([
            dbc.Row([
                dbc.Col([
                    self.importances.layout(),
                ]),
            ], style=dict(margin=25))
        ])


class ClassifierModelStatsComposite(ExplainerComponent):
    def __init__(self, explainer, title="Classification Stats", name=None,
                    hide_title=False, hide_selector=True, pos_label=None,
                    bin_size=0.1, quantiles=10, cutoff=0.5, **kwargs):
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

        self.summary = ClassifierModelSummaryComponent(explainer,  
                hide_selector=hide_selector, pos_label=pos_label, **kwargs)
        self.precision = PrecisionComponent(explainer, 
                hide_selector=hide_selector, pos_label=pos_label, **kwargs)
        self.confusionmatrix = ConfusionMatrixComponent(explainer, 
                hide_selector=hide_selector, pos_label=pos_label, **kwargs)
        self.cumulative_precision = CumulativePrecisionComponent(explainer, 
                hide_selector=hide_selector, pos_label=pos_label, **kwargs)
        self.liftcurve = LiftCurveComponent(explainer, 
                hide_selector=hide_selector, pos_label=pos_label, **kwargs)
        self.classification = ClassificationComponent(explainer, 
                hide_selector=hide_selector, pos_label=pos_label, **kwargs)
        self.rocauc = RocAucComponent(explainer, 
                hide_selector=hide_selector, pos_label=pos_label, **kwargs)
        self.prauc = PrAucComponent(explainer, 
                hide_selector=hide_selector, pos_label=pos_label, **kwargs)

        self.cutoffpercentile = CutoffPercentileComponent(explainer, 
                hide_selector=hide_selector, pos_label=pos_label, **kwargs)
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
            ], style=dict(marginBottom=25)),
            dbc.CardDeck([
                self.summary.layout(),
                self.confusionmatrix.layout(),
            ], style=dict(marginBottom=25)),
            dbc.CardDeck([
                self.precision.layout(),
                self.classification.layout()
            ], style=dict(marginBottom=25)),
            dbc.CardDeck([
                self.rocauc.layout(),
                self.prauc.layout()
            ], style=dict(marginBottom=25)),
            dbc.CardDeck([
                self.liftcurve.layout(),
                self.cumulative_precision.layout()
            ], style=dict(marginBottom=25)),
        ])


class RegressionModelStatsComposite(ExplainerComponent):
    def __init__(self, explainer, title="Regression Stats", name=None,
                    hide_title=False,
                    logs=False, pred_or_actual="vs_pred", residuals='difference',
                    col=None, **kwargs):
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

        self.modelsummary = RegressionModelSummaryComponent(explainer, **kwargs)
        self.preds_vs_actual = PredictedVsActualComponent(explainer, 
                    logs=logs, **kwargs)
        self.residuals = ResidualsComponent(explainer, 
                    pred_or_actual=pred_or_actual, residuals=residuals, **kwargs)
        self.reg_vs_col = RegressionVsColComponent(explainer, 
                    logs=logs, **kwargs)

        self.register_components()

    def layout(self):
        return html.Div([
            dbc.Row([
                make_hideable(
                    dbc.Col([
                        html.H2('Model Performance:')]), hide=self.hide_title)
            ]),
            dbc.CardDeck([
                self.modelsummary.layout(),
                self.preds_vs_actual.layout(),
            ], style=dict(marginBottom=25)),
            dbc.CardDeck([
                self.residuals.layout(),
                self.reg_vs_col.layout(),
            ], style=dict(marginBottom=25))
        ])


class IndividualPredictionsComposite(ExplainerComponent):
    def __init__(self, explainer, title="Individual Predictions", name=None,
                        hide_title=False, hide_selector=True, **kwargs):
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
            self.index = ClassifierRandomIndexComponent(explainer, 
                    hide_selector=hide_selector, **kwargs)
            self.summary = ClassifierPredictionSummaryComponent(explainer, 
                    hide_selector=hide_selector, **kwargs)
        elif self.explainer.is_regression:
            self.index = RegressionRandomIndexComponent(explainer, 
                    hide_selector=hide_selector, **kwargs)
            self.summary = RegressionPredictionSummaryComponent(explainer, 
                    hide_selector=hide_selector, **kwargs)

        self.contributions = ShapContributionsGraphComponent(explainer, 
                        hide_selector=hide_selector, **kwargs)
        self.pdp = PdpComponent(explainer, 
                        hide_selector=hide_selector, **kwargs)
        self.contributions_list = ShapContributionsTableComponent(explainer, 
                        hide_selector=hide_selector,  **kwargs)

        self.index_connector = IndexConnector(self.index, 
                [self.summary, self.contributions, self.pdp, self.contributions_list])

        self.register_components()

    def layout(self):
        return dbc.Container([
                dbc.CardDeck([
                    self.index.layout(),
                    self.summary.layout()
                ], style=dict(marginBottom=25, marginTop=25)),
                dbc.CardDeck([
                    self.contributions.layout(),
                    self.pdp.layout(),
                ], style=dict(marginBottom=25, marginTop=25)),
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
                        hide_title=False, hide_selector=True, **kwargs):
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
            self.index = ClassifierRandomIndexComponent(explainer, 
                        hide_selector=hide_selector, **kwargs)
        elif self.explainer.is_regression:
            self.index = RegressionRandomIndexComponent(explainer, **kwargs)

        self.input = FeatureInputComponent(explainer, 
                        hide_selector=hide_selector, 
                        **update_params(kwargs, hide_index=True))
        
        self.contrib = ShapContributionsGraphComponent(explainer, 
                        feature_input_component=self.input,
                        hide_selector=hide_selector, **kwargs)
        self.pdp = PdpComponent(explainer, feature_input_component=self.input,
                        hide_selector=hide_selector, **kwargs)

        self.index_connector = IndexConnector(self.index, [self.input])

        self.register_components()

    def layout(self):
        return dbc.Container([
                dbc.Row([
                    make_hideable(
                        dbc.Col([
                            html.H1(self.title)
                        ]), hide=self.hide_title),
                ]),
                dbc.Row([
                    dbc.Col([
                        self.index.layout()
                    ]),
                ], style=dict(marginBottom=15, marginTop=15)),
                dbc.Row([
                    dbc.Col([
                        self.input.layout()
                    ]),
                ], style=dict(marginBottom=15, marginTop=15)),
                dbc.CardDeck([self.contrib.layout(), self.pdp.layout()])
        ], fluid=True)


class ShapDependenceComposite(ExplainerComponent):
    def __init__(self, explainer, title='Feature Dependence', name=None,
                    hide_selector=True,
                    depth=None, cats=True, **kwargs):
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
                    self.explainer, **update_params(kwargs,
                    hide_selector=hide_selector, 
                    depth=depth, cats=cats))
        self.shap_dependence = ShapDependenceComponent(
                    self.explainer, 
                    hide_selector=hide_selector, cats=cats,
                    **update_params(kwargs, hide_cats=True)
                    )
        self.connector = ShapSummaryDependenceConnector(
                    self.shap_summary, self.shap_dependence)
        self.register_components()

    def layout(self):
        return dbc.Container([
            dbc.CardDeck([
                self.shap_summary.layout(),
                self.shap_dependence.layout()
            ], style=dict(marginTop=25)),
        ], fluid=True)

class ShapInteractionsComposite(ExplainerComponent):
    def __init__(self, explainer, title='Feature Interactions', name=None,
                    hide_selector=True,
                    depth=None, cats=True, **kwargs):
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

        self.interaction_summary = InteractionSummaryComponent(explainer, 
                hide_selector=hide_selector, depth=depth, cats=cats, **kwargs)
        self.interaction_dependence = InteractionDependenceComponent(explainer, 
                hide_selector=hide_selector, cats=cats, **update_params(kwargs, hide_cats=True))
        self.connector = InteractionSummaryDependenceConnector(
            self.interaction_summary, self.interaction_dependence)
        self.register_components()
        
    def layout(self):
        return dbc.Container([
                dbc.CardDeck([
                    self.interaction_summary.layout(),
                    self.interaction_dependence.layout()
                ], style=dict(marginTop=25))
        ], fluid=True)


class DecisionTreesComposite(ExplainerComponent):
    def __init__(self, explainer, title="Decision Trees", name=None,
                    hide_selector=True, **kwargs):
        """Composite of decision tree related components:
        
        - index selector
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
        
        self.trees = DecisionTreesComponent(explainer, 
                    hide_selector=hide_selector, **kwargs)
        self.decisionpath_table = DecisionPathTableComponent(explainer, 
                    hide_selector=hide_selector, **kwargs)

        if explainer.is_classifier:
            self.index = ClassifierRandomIndexComponent(explainer, 
                    hide_selector=hide_selector, **kwargs)
        elif explainer.is_regression:
            self.index = RegressionRandomIndexComponent(explainer,
                    **kwargs)

        self.decisionpath_graph = DecisionPathGraphComponent(explainer, 
                    hide_selector=hide_selector, **kwargs)

        self.index_connector = IndexConnector(self.index, 
            [self.trees, self.decisionpath_table, self.decisionpath_graph])
        self.highlight_connector = HighlightConnector(self.trees, 
            [self.decisionpath_table, self.decisionpath_graph])

        self.register_components()
        
    def layout(self):
        if isinstance(self.explainer, XGBExplainer):
            return html.Div([
                dbc.Row([
                    dbc.Col([
                        self.index.layout(),
                    ])
                ], style=dict(margin=25)),
                dbc.Row([
                    dbc.Col([
                        self.trees.layout(),
                    ], md=8),
                    dbc.Col([
                        self.decisionpath_table.layout()
                    ], md=4)
                ], style=dict(margin=25)),
                dbc.Row([
                    dbc.Col([
                        self.decisionpath_graph.layout()
                    ])
                ], style=dict(margin=25)),
            ])
        elif isinstance(self.explainer, RandomForestExplainer):
            return html.Div([
                dbc.Row([
                    dbc.Col([
                        self.index.layout(),
                    ])
                ], style=dict(margin=25)),
                dbc.Row([
                    dbc.Col([
                        self.trees.layout(),
                    ]),
                ], style=dict(margin=25)),
                dbc.Row([
                    dbc.Col([
                        self.decisionpath_table.layout() 
                    ]),
                ], style=dict(margin=25)),
                dbc.Row([
                    dbc.Col([
                        self.decisionpath_graph.layout()
                    ])
                ], style=dict(margin=25)),
            ])
        else:
            raise ValueError("explainer is neither a RandomForestExplainer nor an XGBExplainer!")
