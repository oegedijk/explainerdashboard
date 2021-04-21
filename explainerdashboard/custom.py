import dash
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc

from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from .dashboard_components import *
from .dashboard_tabs import *
from .dashboards import ExplainerTabsLayout, ExplainerPageLayout


class SimplifiedClassifierDashboard(ExplainerComponent):
    def __init__(self, explainer, title="Simple Classification Stats", name=None,
                 classifier_custom_component='pr_auc',
                 hide_title=True, hide_selector=True,
                 hide_globalcutoff=False,
                 hide_confusionmatrix=False, hide_classifier_custom_component=False,
                 hide_shapsummary=False, hide_shapdependence=False,
                 hide_predindexselector=False, hide_predictionsummary=False,
                 hide_contributiongraph=False,
                 pos_label=None, index_check=True,
                 bin_size=0.1, quantiles=10, cutoff=0.5, depth=None, **kwargs):
        """Composite of multiple classifier related components, on a single tab: 
            - confusion matrix
            - one other model quality indicator: choose from pr auc graph, precision graph, lift curve, classification graph, or roc auc graph
            - shap importance
            - shap dependence
            - shap contribution graph

        Args:
            explainer (Explainer): explainer object constructed with either
                        ClassifierExplainer() or RegressionExplainer()
            title (str, optional): Title of tab or page. Defaults to 
                        "Decision Trees".
            name (str, optional): unique name to add to Component elements. 
                        If None then random uuid is generated to make sure 
                        it's unique. Defaults to None.
            classifier_custom_component (str, optional): custom classifier quality indicator supported by the ClassifierExplainer object. 
                        Valid values are: pr_auc, precision_graph, lift_curve, class_graph, roc_auc.
            hide_title (bool, optional): hide title. Defaults to True.          
            hide_selector (bool, optional): hide all pos label selectors. Defaults to True.
            hide_globalcutoff (bool, optional): hide CutoffPercentileComponent
            hide_confusionmatrix (bool, optional): hide ConfusionMatrixComponent
            hide_classifier_custom_component (bool, optional): hide the chosen classifier_custom_component
            hide_shapsummary (bool, optional): hide ShapSummaryComponent
            hide_shapdependence (bool, optional): hide ShapDependenceComponent
            hide_predindexselector (bool, optional): hide ClassifierRandomIndexComponent 
                or RegressionRandomIndexComponent
            hide_predictionsummary (bool, optional): hide ClassifierPredictionSummaryComponent
                or RegressionPredictionSummaryComponent
            hide_contributiongraph (bool, optional): hide ShapContributionsGraphComponent
            pos_label ({int, str}, optional): initial pos label. Defaults to explainer.pos_label
            index_check (bool, optional): only pass valid indexes from random index 
                selector to feature input. Defaults to True.
            bin_size (float, optional): bin_size for precision plot. Defaults to 0.1.
            quantiles (int, optional): number of quantiles for precision plot. Defaults to 10.
            cutoff (float, optional): initial cutoff. Defaults to 0.5.
            depth (int, optional): Number of features to display. Defaults to None.
        """
        super().__init__(explainer, title, name)

        assert classifier_custom_component.lower() in ['pr_auc', 'precision_graph', 'lift_curve', 'class_graph',
                                                       'roc_auc'], 'ERROR: classifier_custom_component only accept one of the following values: [\'pr_auc\', \'precision_graph\', \'lift_curve\', \'class_graph\', \'roc_auc\']. Please check your input!'

        self.confusionmatrix = ConfusionMatrixComponent(explainer, name=self.name+"0",
                                                        hide_selector=hide_selector, pos_label=pos_label, **kwargs)
        if classifier_custom_component.lower() == 'pr_auc':
            self.classifier_custom_component = PrAucComponent(explainer, name=self.name+"1",
                                                              hide_selector=hide_selector, pos_label=pos_label, **kwargs)
        elif classifier_custom_component.lower() == 'precision_graph':
            self.classifier_custom_component = PrecisionComponent(explainer, name=self.name+"1",
                                                                  hide_selector=hide_selector, pos_label=pos_label, **kwargs)
        elif classifier_custom_component.lower() == 'lift_curve':
            self.classifier_custom_component = LiftCurveComponent(explainer, name=self.name+"1",
                                                                  hide_selector=hide_selector, pos_label=pos_label, **kwargs)
        elif classifier_custom_component.lower() == 'class_graph':
            self.classifier_custom_component = ClassificationComponent(explainer, name=self.name+"1",
                                                                       hide_selector=hide_selector, pos_label=pos_label, **kwargs)
        else:  # must be roc_auc
            self.classifier_custom_component = RocAucComponent(explainer, name=self.name+"1",
                                                               hide_selector=hide_selector, pos_label=pos_label, **kwargs)

        self.shap_summary = ShapSummaryComponent(
            explainer, name=self.name+"2",
            **update_params(kwargs, hide_selector=hide_selector, depth=depth))
        self.shap_dependence = ShapDependenceComponent(
            explainer, name=self.name+"3",
            hide_selector=hide_selector, **kwargs)
        self.index = ClassifierRandomIndexComponent(explainer, name=self.name+"4",
                                                    hide_selector=hide_selector, **kwargs)
        self.summary = ClassifierPredictionSummaryComponent(explainer, name=self.name+"5",
                                                            hide_selector=hide_selector, **kwargs)
        self.contributions = ShapContributionsGraphComponent(explainer, name=self.name+"6",
                                                             hide_selector=hide_selector, **kwargs)

        self.cutoffpercentile = CutoffPercentileComponent(explainer, name=self.name+"7",
                                                          hide_selector=hide_selector, pos_label=pos_label, **kwargs)
        self.cutoffconnector = CutoffConnector(self.cutoffpercentile,
                                               [self.confusionmatrix, self.classifier_custom_component])
        self.connector = ShapSummaryDependenceConnector(
            self.shap_summary, self.shap_dependence)
        self.index_connector = IndexConnector(self.index,
                                              [self.summary, self.contributions],
                                              explainer=explainer if index_check else None)

    def layout(self):
        return html.Div([
            dbc.Row([
                make_hideable(
                    dbc.Col([
                        self.cutoffpercentile.layout(),
                    ]), hide=self.hide_globalcutoff),
            ], style=dict(marginTop=25, marginBottom=25)),
            dbc.CardDeck([
                make_hideable(self.confusionmatrix.layout(),
                              hide=self.hide_confusionmatrix),
                make_hideable(self.classifier_custom_component.layout(),
                              hide=self.hide_classifier_custom_component),
            ], style=dict(marginBottom=25)),
            dbc.CardDeck([
                make_hideable(self.shap_summary.layout(),
                              hide=self.hide_shapsummary),
                make_hideable(self.shap_dependence.layout(),
                              hide=self.hide_shapdependence),
            ], style=dict(marginTop=25)),
            dbc.CardDeck([
                make_hideable(self.index.layout(),
                              hide=self.hide_predindexselector),
                make_hideable(self.summary.layout(),
                              hide=self.hide_predictionsummary),
            ], style=dict(marginBottom=25, marginTop=25)),
            dbc.CardDeck([
                make_hideable(self.contributions.layout(),
                              hide=self.hide_contributiongraph),
            ], style=dict(marginBottom=25, marginTop=25))
        ])
