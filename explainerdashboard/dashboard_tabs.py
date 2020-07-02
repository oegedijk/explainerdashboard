__all__ = [
    'ImportancesTab',
    'ModelSummaryTab',
    'ContributionsTab',
    'ShapDependenceTab',
    'ShapInteractionsTab',
    'DecisionTreesTab',
]

import dash_html_components as html

from .dashboard_components import *


class ImportancesTab(ExplainerComponent):
    def __init__(self, explainer, title="Feature Importances",
                    header_mode="none", name=None,
                    importance_type="shap", depth=None, cats=True):
        """Overview tab of feature importances

        Can show both permutation importances and mean absolute shap values.

        Args:
            explainer (Explainer): explainer object constructed with either
                        ClassifierExplainer() or RegressionExplainer()
            title (str, optional): Title of tab or page. Defaults to 
                        "Feature Importances".
            header_mode (str, optional): {"dashboard", "standalone", "hidden" or "none"}. 
                        Defaults to "none".
            name (str, optional): unique name to add to Component elements. 
                        If None then random uuid is generated to make sure 
                        it's unique. Defaults to None.
            importance_type (str, {'permutation', 'shap'} optional): 
                        Type of importance to describe. Defaults to "shap".
            depth (int, optional): Number of features to display by default. Defaults to None.
            cats (bool, optional): Group categoricals together. Defaults to True.
        """
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
                    bin_size=0.1, quantiles=10, cutoff=0.5, 
                    logs=False, pred_or_actual="vs_pred", ratio=False, col=None):
        """Tab shows a summary of model performance.

        Args:
            explainer (Explainer): explainer object constructed with either
                        ClassifierExplainer() or RegressionExplainer()
            title (str, optional): Title of tab or page. Defaults to 
                        "Model Performance".
            header_mode (str, optional): {"dashboard", "standalone", "hidden" or "none"}. 
                        Defaults to "none".
            name (str, optional): unique name to add to Component elements. 
                        If None then random uuid is generated to make sure 
                        it's unique. Defaults to None.
            bin_size (float, optional): precision plot bin_size. Defaults to 0.1.
            quantiles (int, optional): precision plot number of quantiles. Defaults to 10.
            cutoff (float, optional): cutoff for classifier plots. Defaults to 0.5.
            logs (bool, optional): use logs for regression plots. Defaults to False.
            pred_or_actual (str, optional): show residuals vs prediction or vs actual. Defaults to "vs_pred".
            ratio (bool, optional): show residual ratio. Defaults to False.
            col ([type], optional): Feature to show residuals against. Defaults to None.

        """
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
        """Tab showing individual predictions, the SHAP contributions that
        add up to this predictions, in both graph and table form, and a pdp plot.

        Args:
            explainer (Explainer): explainer object constructed with either
                        ClassifierExplainer() or RegressionExplainer()
            title (str, optional): Title of tab or page. Defaults to 
                        "Individual Predictions".
            header_mode (str, optional): {"dashboard", "standalone", "hidden" or "none"}. 
                        Defaults to "none".
            name (str, optional): unique name to add to Component elements. 
                        If None then random uuid is generated to make sure 
                        it's unique. Defaults to None.
        """
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
        """Tab showing both a summary of feature importance (aggregate or detailed).
        for each feature, and a shap dependence graph.

        Args:
            explainer (Explainer): explainer object constructed with either
                        ClassifierExplainer() or RegressionExplainer()
            title (str, optional): Title of tab or page. Defaults to 
                        'Feature Dependence'.
            header_mode (str, optional): {"dashboard", "standalone", "hidden" or "none"}. 
                        Defaults to "none".
            name (str, optional): unique name to add to Component elements. 
                        If None then random uuid is generated to make sure 
                        it's unique. Defaults to None.
        """
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
                    depth=None, cats=True):
        """[summary]

        Args:
            explainer (Explainer): explainer object constructed with either
                        ClassifierExplainer() or RegressionExplainer()
            title (str, optional): Title of tab or page. Defaults to 
                        'Feature Interactions'.
            header_mode (str, optional): {"dashboard", "standalone", "hidden" or "none"}. 
                        Defaults to "none".
            name (str, optional): unique name to add to Component elements. 
                        If None then random uuid is generated to make sure 
                        it's unique. Defaults to None.
            depth (int, optional): default number of feature to display. Defaults to None.
            cats (bool, optional): default grouping of cats. Defaults to True.
        """
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
        """Tab showing individual decision trees

        Args:
            explainer (Explainer): explainer object constructed with either
                        ClassifierExplainer() or RegressionExplainer()
            title (str, optional): Title of tab or page. Defaults to 
                        'Decision Trees'.
            header_mode ({"dashboard", "standalone", "hidden" or "none"}, optional): 
                        Defaults to "none".
            name (str, optional): unique name to add to Component elements. 
                        If None then random uuid is generated to make sure 
                        it's unique. Defaults to None.
        """
        super().__init__(explainer, title, header_mode, name)

    
        self.trees = DecisionTreesComposite(explainer)
        self.register_components(self.trees)

    def _layout(self):
        return html.Div([
            self.trees.layout()
        ])