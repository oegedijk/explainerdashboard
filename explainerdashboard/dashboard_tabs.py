__all__ = [
    'ImportancesTab',
    'ModelSummaryTab',
    'ContributionsTab',
    'ShapDependenceTab',
    'ShapInteractionsTab',
    'DecisionTreesTab',
]

import dash
import dash_bootstrap_components as dbc
import dash_html_components as html

from .dashboard_components import *


class ImportancesTab(ExplainerComponent):
    def __init__(self, explainer, title="Feature Importances",
                    header_mode="none", name=None,
                    tab_id="importances",
                    importance_type="shap", depth=None, cats=True):
        super().__init__(explainer, title, header_mode, name)

        self.importances = ImportancesComponent(explainer, 
                importance_type=importance_type, depth=depth, cats=cats)

        self.register_components(self.importances)

    def _layout(self):
        return html.Div([
            self.importances.layout(),
        ])


class ModelSummaryTab(ExplainerComponent):
    def __init__(self, explainer, title="Model Performance",
                    header_mode="none", name=None,
                    tab_id="modelsummary",
                    bin_size=0.1, quantiles=10, cutoff=0.5, 
                    logs=False, pred_or_actual="vs_pred", ratio=False, col=None,
                    importance_type="shap", depth=None, cats=True):
        super().__init__(explainer, title, header_mode, name)
        
        if self.explainer.is_classifier:
            self.model_stats = ClassifierModelStatsComposite(explainer, 
                bin_size=bin_size, quantiles=quantiles, cutoff=cutoff) 
        elif explainer.is_regression:
            self.model_stats = RegressionModelStatsComposite(explainer,
                logs=logs, pred_or_actual=pred_or_actual, ratio=ratio)

        self.register_components(self.model_stats)

    def _layout(self):
        return html.Div([
            self.model_stats.layout()
        ])


class ContributionsTab(ExplainerComponent):
    def __init__(self, explainer, title="Individual Predictions",
                        header_mode="none", name=None):
        super().__init__(explainer, title, header_mode, name)
        self.tab_id = "contributions"
        self.contribs = IndividualPredictionsComposite(explainer)
        self.register_components(self.contribs)
    
    def _layout(self):
        return html.Div([
            self.contribs.layout()
        ])


class ShapDependenceTab(ExplainerComponent):
    def __init__(self, explainer, title='Feature Dependence',
                    header_mode="none", name=None,
                    tab_id="shap_dependence", 
                    depth=None, cats=True):
        super().__init__(explainer, title, header_mode, name)

        self.shap_overview = ShapDependenceComposite(
            explainer, depth=depth, cats=cats)
        self.register_components(self.shap_overview)

    def _layout(self):
        return html.Div([
            self.shap_overview.layout()
        ])


class ShapInteractionsTab(ExplainerComponent):
    def __init__(self, explainer, title='Feature Interactions',
                    header_mode="none", name=None,
                    tab_id="shap_interactions", 
                    depth=None, cats=True):
        super().__init__(explainer, title, header_mode, name)
        self.interaction_overview = ShapInteractionsComposite(
                    explainer, depth=depth, cats=cats)
        self.register_components(self.interaction_overview)

    def _layout(self):
        return html.Div([
            self.interaction_overview.layout()
        ])


class DecisionTreesTab(ExplainerComponent):
    def __init__(self, explainer, title="Decision Trees",
                    header_mode="none", name=None):
        super().__init__(explainer, title, header_mode, name)
    
        self.trees = DecisionTreesComposite(explainer)
        self.register_components(self.trees)

    def _layout(self):
        return html.Div([
            self.trees.layout()
        ])