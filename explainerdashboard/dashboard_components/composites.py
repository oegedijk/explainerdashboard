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
                    hide_importances=False,
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
            hide_importances (bool, optional): hide the ImportancesComponent
            hide_selector (bool, optional): hide the post label selector. 
                Defaults to True.
        """
        super().__init__(explainer, title, name)

        self.importances = ImportancesComponent(
                explainer, name=self.name+"0", hide_selector=hide_selector, **kwargs)
        self.register_components()

    def layout(self):
        return html.Div([
            dbc.Row([
                make_hideable(
                    dbc.Col([
                        self.importances.layout(),
                    ]), hide=self.hide_importances),
            ], style=dict(margin=25))
        ])


class ClassifierModelStatsComposite(ExplainerComponent):
    def __init__(self, explainer, title="Classification Stats", name=None,
                    hide_title=True, hide_selector=True, 
                    hide_globalcutoff=False,
                    hide_modelsummary=False, hide_confusionmatrix=False,
                    hide_precision=False, hide_classification=False,
                    hide_rocauc=False, hide_prauc=False,
                    hide_liftcurve=False, hide_cumprecision=False,
                    pos_label=None,
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
            hide_title (bool, optional): hide title. Defaults to True.          
            hide_selector (bool, optional): hide all pos label selectors. Defaults to True.
            hide_globalcutoff (bool, optional): hide CutoffPercentileComponent
            hide_modelsummary (bool, optional): hide ClassifierModelSummaryComponent
            hide_confusionmatrix (bool, optional): hide ConfusionMatrixComponent
            hide_precision (bool, optional): hide PrecisionComponent
            hide_classification (bool, optional): hide ClassificationComponent
            hide_rocauc (bool, optional): hide RocAucComponent
            hide_prauc (bool, optional): hide PrAucComponent
            hide_liftcurve (bool, optional): hide LiftCurveComponent
            hide_cumprecision (bool, optional): hide CumulativePrecisionComponent
            pos_label ({int, str}, optional): initial pos label. Defaults to explainer.pos_label
            bin_size (float, optional): bin_size for precision plot. Defaults to 0.1.
            quantiles (int, optional): number of quantiles for precision plot. Defaults to 10.
            cutoff (float, optional): initial cutoff. Defaults to 0.5.
        """
        super().__init__(explainer, title, name)

        self.summary = ClassifierModelSummaryComponent(explainer, name=self.name+"0", 
                hide_selector=hide_selector, pos_label=pos_label, **kwargs)
        self.precision = PrecisionComponent(explainer, name=self.name+"1",
                hide_selector=hide_selector, pos_label=pos_label, **kwargs)
        self.confusionmatrix = ConfusionMatrixComponent(explainer, name=self.name+"2",
                hide_selector=hide_selector, pos_label=pos_label, **kwargs)
        self.cumulative_precision = CumulativePrecisionComponent(explainer, name=self.name+"3",
                hide_selector=hide_selector, pos_label=pos_label, **kwargs)
        self.liftcurve = LiftCurveComponent(explainer, name=self.name+"4",
                hide_selector=hide_selector, pos_label=pos_label, **kwargs)
        self.classification = ClassificationComponent(explainer, name=self.name+"5",
                hide_selector=hide_selector, pos_label=pos_label, **kwargs)
        self.rocauc = RocAucComponent(explainer, name=self.name+"6",
                hide_selector=hide_selector, pos_label=pos_label, **kwargs)
        self.prauc = PrAucComponent(explainer, name=self.name+"7",
                hide_selector=hide_selector, pos_label=pos_label, **kwargs)

        self.cutoffpercentile = CutoffPercentileComponent(explainer, name=self.name+"8",
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
                make_hideable(
                    dbc.Col([
                        self.cutoffpercentile.layout(),
                    ]), hide=self.hide_globalcutoff),
            ], style=dict(marginBottom=25)),
            dbc.CardDeck([
                make_hideable(self.summary.layout(), hide=self.hide_modelsummary),
                make_hideable(self.confusionmatrix.layout(), hide=self.hide_confusionmatrix),
            ], style=dict(marginBottom=25)),
            dbc.CardDeck([
                make_hideable(self.precision.layout(), hide=self.hide_precision),
                make_hideable(self.classification.layout(), hide=self.hide_classification)
            ], style=dict(marginBottom=25)),
            dbc.CardDeck([
                make_hideable(self.rocauc.layout(), hide=self.hide_rocauc),
                make_hideable(self.prauc.layout(), hide=self.hide_prauc),
            ], style=dict(marginBottom=25)),
            dbc.CardDeck([
                make_hideable(self.liftcurve.layout(), self.hide_liftcurve),
                make_hideable(self.cumulative_precision.layout(), self.hide_cumprecision),
            ], style=dict(marginBottom=25)),
        ])


class RegressionModelStatsComposite(ExplainerComponent):
    def __init__(self, explainer, title="Regression Stats", name=None,
                    hide_title=True, hide_modelsummary=False,
                    hide_predsvsactual=False, hide_residuals=False, 
                    hide_regvscol=False,
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
            hide_title (bool, optional): hide title. Defaults to True.
            hide_modelsummary (bool, optional): hide RegressionModelSummaryComponent
            hide_predsvsactual (bool, optional): hide PredictedVsActualComponent
            hide_residuals (bool, optional): hide ResidualsComponent
            hide_regvscol (bool, optional): hide RegressionVsColComponent
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

        self.modelsummary = RegressionModelSummaryComponent(explainer, 
                                name=self.name+"0",**kwargs)
        self.preds_vs_actual = PredictedVsActualComponent(explainer, name=self.name+"0",
                    logs=logs, **kwargs)
        self.residuals = ResidualsComponent(explainer, name=self.name+"1",
                    pred_or_actual=pred_or_actual, residuals=residuals, **kwargs)
        self.reg_vs_col = RegressionVsColComponent(explainer, name=self.name+"2",
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
                make_hideable(self.modelsummary.layout(), hide=self.hide_modelsummary),
                make_hideable(self.preds_vs_actual.layout(), hide=self.hide_predsvsactual),
            ], style=dict(margin=25)),
            dbc.CardDeck([
                make_hideable(self.residuals.layout(), hide=self.hide_residuals),
                make_hideable(self.reg_vs_col.layout(), hide=self.hide_regvscol),
            ], style=dict(margin=25))
        ])


class IndividualPredictionsComposite(ExplainerComponent):
    def __init__(self, explainer, title="Individual Predictions", name=None,
                        hide_predindexselector=False, hide_predictionsummary=False,
                        hide_contributiongraph=False, hide_pdp=False,
                        hide_contributiontable=False,
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
            hide_predindexselector (bool, optional): hide ClassifierRandomIndexComponent 
                or RegressionRandomIndexComponent
            hide_predictionsummary (bool, optional): hide ClassifierPredictionSummaryComponent
                or RegressionPredictionSummaryComponent
            hide_contributiongraph (bool, optional): hide ShapContributionsGraphComponent
            hide_pdp (bool, optional): hide PdpComponent
            hide_contributiontable (bool, optional): hide ShapContributionsTableComponent
            hide_title (bool, optional): hide title. Defaults to False.
            hide_selector(bool, optional): hide all pos label selectors. Defaults to True.
        """
        super().__init__(explainer, title, name)

        if self.explainer.is_classifier:
            self.index = ClassifierRandomIndexComponent(explainer, name=self.name+"0",
                    hide_selector=hide_selector, **kwargs)
            self.summary = ClassifierPredictionSummaryComponent(explainer, name=self.name+"1",
                    hide_selector=hide_selector, **kwargs)
        elif self.explainer.is_regression:
            self.index = RegressionRandomIndexComponent(explainer, name=self.name+"0",
                    hide_selector=hide_selector, **kwargs)
            self.summary = RegressionPredictionSummaryComponent(explainer, name=self.name+"1",
                    hide_selector=hide_selector, **kwargs)

        self.contributions = ShapContributionsGraphComponent(explainer, name=self.name+"2",
                        hide_selector=hide_selector, **kwargs)
        self.pdp = PdpComponent(explainer, name=self.name+"3",
                        hide_selector=hide_selector, **kwargs)
        self.contributions_list = ShapContributionsTableComponent(explainer, name=self.name+"4",
                        hide_selector=hide_selector,  **kwargs)

        self.index_connector = IndexConnector(self.index, 
                [self.summary, self.contributions, self.pdp, self.contributions_list])

        self.register_components()

    def layout(self):
        return dbc.Container([
                dbc.CardDeck([
                    make_hideable(self.index.layout(), hide=self.hide_predindexselector),
                    make_hideable(self.summary.layout(), hide=self.hide_predictionsummary),
                ], style=dict(marginBottom=25, marginTop=25)),
                dbc.CardDeck([
                    make_hideable(self.contributions.layout(), hide=self.hide_contributiongraph),
                    make_hideable(self.pdp.layout(), hide=self.hide_pdp),
                ], style=dict(marginBottom=25, marginTop=25)),
                dbc.Row([
                    dbc.Col([
                        make_hideable(self.contributions_list.layout(), hide=self.hide_contributiontable),
                    ], md=6),
                    dbc.Col([
                        html.Div([]),
                    ], md=6),
                ])
        ], fluid=True)


class WhatIfComposite(ExplainerComponent):
    def __init__(self, explainer, title="What if...", name=None,
                        hide_whatifindexselector=False, hide_inputeditor=False,
                        hide_whatifprediction=False, hide_whatifcontributiongraph=False, 
                        hide_whatifpdp=False, hide_whatifcontributiontable=False,
                        hide_title=True, hide_selector=True, 
                        n_input_cols=4, sort='importance', **kwargs):
        """Composite for the whatif component:

        Args:
            explainer (Explainer): explainer object constructed with either
                        ClassifierExplainer() or RegressionExplainer()
            title (str, optional): Title of tab or page. Defaults to 
                        "Individual Predictions".
            name (str, optional): unique name to add to Component elements. 
                        If None then random uuid is generated to make sure 
                        it's unique. Defaults to None.
            hide_title (bool, optional): hide title. Defaults to True.
            hide_selector(bool, optional): hide all pos label selectors. Defaults to True.
            hide_whatifindexselector (bool, optional): hide ClassifierRandomIndexComponent
                or RegressionRandomIndexComponent
            hide_inputeditor (bool, optional): hide FeatureInputComponent
            hide_whatifprediction (bool, optional): hide PredictionSummaryComponent
            hide_whatifcontributiongraph (bool, optional): hide ShapContributionsGraphComponent
            hide_whatifcontributiontable (bool, optional): hide ShapContributionsTableComponent
            hide_whatifpdp (bool, optional): hide PdpComponent
            n_input_cols (int, optional): number of columns to divide the feature inputs into.
                Defaults to 4. 
            sort ({'abs', 'high-to-low', 'low-to-high', 'importance'}, optional): sorting of shap values. 
                        Defaults to 'importance'.
        """
        super().__init__(explainer, title, name)
        
        if 'hide_whatifcontribution' in kwargs:
            print("Warning: hide_whatifcontribution will be deprecated, use hide_whatifcontributiongraph instead!")
            self.hide_whatifcontributiongraph = kwargs['hide_whatifcontribution']

        self.input = FeatureInputComponent(explainer, name=self.name+"0",
                        hide_selector=hide_selector, n_input_cols=self.n_input_cols,
                        **update_params(kwargs, hide_index=True))
        
        if self.explainer.is_classifier:
            self.index = ClassifierRandomIndexComponent(explainer, name=self.name+"1",
                        hide_selector=hide_selector, **kwargs)
            self.prediction = ClassifierPredictionSummaryComponent(explainer, name=self.name+"2",
                        feature_input_component=self.input,
                        hide_star_explanation=True,
                        hide_selector=hide_selector, **kwargs)
        elif self.explainer.is_regression:
            self.index = RegressionRandomIndexComponent(explainer, name=self.name+"1", **kwargs)
            self.prediction = RegressionPredictionSummaryComponent(explainer, name=self.name+"2",
                        feature_input_component=self.input, **kwargs)
        
        
        self.contribgraph = ShapContributionsGraphComponent(explainer, name=self.name+"3",
                        feature_input_component=self.input,
                        hide_selector=hide_selector, sort=sort, **kwargs)
        self.contribtable = ShapContributionsTableComponent(explainer, name=self.name+"4",
                        feature_input_component=self.input,
                        hide_selector=hide_selector, sort=sort, **kwargs)
        
        self.pdp = PdpComponent(explainer, name=self.name+"5",
                        feature_input_component=self.input,
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
                    make_hideable(
                        dbc.Col([
                            self.index.layout(), 
                        ], md=7), hide=self.hide_whatifindexselector),
                    make_hideable(
                        dbc.Col([
                            self.prediction.layout(),
                        ], md=5), hide=self.hide_whatifprediction),
                ], style=dict(marginBottom=15, marginTop=15)),
                dbc.CardDeck([
                    make_hideable(self.input.layout(), hide=self.hide_inputeditor),
                ], style=dict(marginBottom=15, marginTop=15)),
                dbc.CardDeck([
                    make_hideable(self.contribgraph.layout(), hide=self.hide_whatifcontributiongraph),
                    make_hideable(self.pdp.layout(), hide=self.hide_whatifpdp),
                ], style=dict(marginBottom=15, marginTop=15)),
                dbc.Row([
                    make_hideable(
                        dbc.Col([
                            self.contribtable.layout()
                        ], md=6), hide=self.hide_whatifcontributiontable),
                    dbc.Col([], md=6),
                ])
        ], fluid=True)


class ShapDependenceComposite(ExplainerComponent):
    def __init__(self, explainer, title='Feature Dependence', name=None,
                    hide_selector=True, 
                    hide_shapsummary=False, hide_shapdependence=False,
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
            hide_shapsummary (bool, optional): hide ShapSummaryComponent
            hide_shapdependence (bool, optional): ShapDependenceComponent
            depth (int, optional): Number of features to display. Defaults to None.
            cats (bool, optional): Group categorical features. Defaults to True.
        """
        super().__init__(explainer, title, name)
        
        self.shap_summary = ShapSummaryComponent(
                    self.explainer, name=self.name+"0",
                    **update_params(kwargs, hide_selector=hide_selector, depth=depth, cats=cats))
        self.shap_dependence = ShapDependenceComponent(
                    self.explainer, name=self.name+"1",
                    hide_selector=hide_selector, cats=cats,
                    **update_params(kwargs, hide_cats=True)
                    )
        self.connector = ShapSummaryDependenceConnector(
                    self.shap_summary, self.shap_dependence)
        self.register_components()

    def layout(self):
        return dbc.Container([
            dbc.CardDeck([
                make_hideable(self.shap_summary.layout(), hide=self.hide_shapsummary),
                make_hideable(self.shap_dependence.layout(), hide=self.hide_shapdependence),
            ], style=dict(marginTop=25)),
        ], fluid=True)

class ShapInteractionsComposite(ExplainerComponent):
    def __init__(self, explainer, title='Feature Interactions', name=None,
                    hide_selector=True,
                    hide_interactionsummary=False, hide_interactiondependence=False,
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
            hide_interactionsummary (bool, optional): hide InteractionSummaryComponent
            hide_interactiondependence (bool, optional): hide InteractionDependenceComponent
            depth (int, optional): Initial number of features to display. Defaults to None.
            cats (bool, optional): Initally group cats. Defaults to True.
        """
        super().__init__(explainer, title, name)

        self.interaction_summary = InteractionSummaryComponent(explainer, name=self.name+"0",
                hide_selector=hide_selector, depth=depth, cats=cats, **kwargs)
        self.interaction_dependence = InteractionDependenceComponent(explainer, name=self.name+"1",
                hide_selector=hide_selector, cats=cats, **update_params(kwargs, hide_cats=True))
        self.connector = InteractionSummaryDependenceConnector(
            self.interaction_summary, self.interaction_dependence)
        self.register_components()
        
    def layout(self):
        return dbc.Container([
                dbc.CardDeck([
                    make_hideable(self.interaction_summary.layout(), hide=self.hide_interactionsummary),
                    make_hideable(self.interaction_dependence.layout(), hide=self.hide_interactiondependence),
                ], style=dict(marginTop=25))
        ], fluid=True)


class DecisionTreesComposite(ExplainerComponent):
    def __init__(self, explainer, title="Decision Trees", name=None,
                    hide_treeindexselector=False, hide_treesgraph=False,
                    hide_treepathtable=False, hide_treepathgraph=False,
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
            hide_treeindexselector (bool, optional): hide ClassifierRandomIndexComponent
                or RegressionRandomIndexComponent
            hide_treesgraph (bool, optional): hide DecisionTreesComponent
            hide_treepathtable (bool, optional): hide DecisionPathTableComponent
            hide_treepathgraph (bool, optional): DecisionPathGraphComponent
            hide_selector (bool, optional): hide all pos label selectors. Defaults to True.
        """
        super().__init__(explainer, title, name)
        
        self.trees = DecisionTreesComponent(explainer, name=self.name+"0",
                    hide_selector=hide_selector, **kwargs)
        self.decisionpath_table = DecisionPathTableComponent(explainer, name=self.name+"1",
                    hide_selector=hide_selector, **kwargs)

        if explainer.is_classifier:
            self.index = ClassifierRandomIndexComponent(explainer, name=self.name+"2",
                    hide_selector=hide_selector, **kwargs)
        elif explainer.is_regression:
            self.index = RegressionRandomIndexComponent(explainer, name=self.name+"2",
                    **kwargs)

        self.decisionpath_graph = DecisionPathGraphComponent(explainer, name=self.name+"3",
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
                    make_hideable(
                        dbc.Col([
                            self.index.layout(), 
                        ]), hide=self.hide_treeindexselector),
                ], style=dict(margin=25)),
                dbc.Row([
                    make_hideable(
                        dbc.Col([
                            self.trees.layout(), 
                        ], md=8), hide=self.hide_treesgraph),
                    make_hideable(
                        dbc.Col([
                            self.decisionpath_table.layout(), 
                        ], md=4), hide=self.hide_treepathtable),
                ], style=dict(margin=25)),
                dbc.Row([
                    make_hideable(
                        dbc.Col([
                            self.decisionpath_graph.layout()
                        ]), hide=self.hide_treepathgraph),
                ], style=dict(margin=25)),
            ])
        elif isinstance(self.explainer, RandomForestExplainer):
            return html.Div([
                dbc.Row([
                    make_hideable(
                        dbc.Col([
                            self.index.layout(), 
                        ]), hide=self.hide_treeindexselector),
                ], style=dict(margin=25)),
                dbc.Row([
                    make_hideable(
                        dbc.Col([
                            self.trees.layout(), 
                        ]), hide=self.hide_treesgraph),
                ], style=dict(margin=25)),
                dbc.Row([
                    make_hideable(
                        dbc.Col([
                            self.decisionpath_table.layout(), 
                        ]), hide=self.hide_treepathtable),
                ], style=dict(margin=25)),
                dbc.Row([
                    make_hideable(
                        dbc.Col([
                            self.decisionpath_graph.layout()
                        ]), hide=self.hide_treepathgraph),
                ], style=dict(margin=25)),
            ])
        else:
            raise ValueError("explainer is neither a RandomForestExplainer nor an XGBExplainer! "
                            "Pass decision_trees=False to disable the decision tree tab.")
