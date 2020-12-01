__all__ = [
    'ImportancesTab',
    'ModelSummaryTab',
    'ContributionsTab',
    'WhatIfTab',
    'ShapDependenceTab',
    'ShapInteractionsTab',
    'DecisionTreesTab',
]

import dash_html_components as html

from .dashboard_components import *


class ImportancesTab(ExplainerComponent):
    def __init__(self, explainer, title="Feature Importances", name=None,
                    hide_type=False, hide_depth=False, hide_cats=False,
                    hide_title=False, hide_selector=False,
                    pos_label=None, importance_type="shap", depth=None, 
                    cats=True, disable_permutations=False, **kwargs):
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
            hide_selector(bool, optional) Hide pos label selector. Defaults to False.
            importance_type (str, {'permutation', 'shap'} optional): 
                        Type of importance to describe. Defaults to "shap".
            depth (int, optional): Number of features to display by default. Defaults to None.
            cats (bool, optional): Group categoricals together. Defaults to True.
        """
        super().__init__(explainer, title, name)

        self.importances = ImportancesComponent(
                explainer, 
                hide_selector=hide_selector,
                importance_type=importance_type, 
                depth=depth, 
                cats=cats, hide_cats=hide_cats)

        self.register_components(self.importances)

    def layout(self):
        return html.Div([
            self.importances.layout(),
        ])


class ModelSummaryTab(ExplainerComponent):
    def __init__(self, explainer, title="Model Performance", name=None,
                    bin_size=0.1, quantiles=10, cutoff=0.5, 
                    logs=False, pred_or_actual="vs_pred", residuals='difference', 
                    col=None, **kwargs):
        """Tab shows a summary of model performance.

        Args:
            explainer (Explainer): explainer object constructed with either
                        ClassifierExplainer() or RegressionExplainer()
            title (str, optional): Title of tab or page. Defaults to 
                        "Model Performance".
            name (str, optional): unique name to add to Component elements. 
                        If None then random uuid is generated to make sure 
                        it's unique. Defaults to None.
            bin_size (float, optional): precision plot bin_size. Defaults to 0.1.
            quantiles (int, optional): precision plot number of quantiles. Defaults to 10.
            cutoff (float, optional): cutoff for classifier plots. Defaults to 0.5.
            logs (bool, optional): use logs for regression plots. Defaults to False.
            pred_or_actual (str, optional): show residuals vs prediction or vs actual. Defaults to "vs_pred".
            residuals (str, {'difference', 'ratio', 'log-ratio'} optional): 
                    How to calcualte residuals. Defaults to 'difference'.
            col ([type], optional): Feature to show residuals against. Defaults to None.

        """
        super().__init__(explainer, title, name)
        
        if self.explainer.is_classifier:
            self.model_stats = ClassifierModelStatsComposite(explainer, 
                bin_size=bin_size, quantiles=quantiles, cutoff=cutoff, **kwargs) 
        elif explainer.is_regression:
            self.model_stats = RegressionModelStatsComposite(explainer,
                logs=logs, pred_or_actual=pred_or_actual, residuals=residuals, **kwargs)

        self.register_components(self.model_stats)

    def layout(self):
        return html.Div([
            self.model_stats.layout()
        ])


class ContributionsTab(ExplainerComponent):
    def __init__(self, explainer, title="Individual Predictions", name=None, 
                     **kwargs):
        """Tab showing individual predictions, the SHAP contributions that
        add up to this predictions, in both graph and table form, and a pdp plot.

        Args:
            explainer (Explainer): explainer object constructed with either
                ClassifierExplainer() or RegressionExplainer()
            title (str, optional): Title of tab or page. Defaults to 
                "Individual Predictions".
            name (str, optional): unique name to add to Component elements. 
                If None then random uuid is generated to make sure  it's unique. 
                Defaults to None.
            higher_is_better (bool, optional): in contributions plot, up is green
                and down is red. (set to False to flip)
        """
        super().__init__(explainer, title, name)
        self.tab_id = "contributions"
        self.contribs = IndividualPredictionsComposite(explainer, 
                            #higher_is_better=higher_is_better, 
                            **kwargs)
        self.register_components(self.contribs)
    
    def layout(self):
        return html.Div([
            self.contribs.layout()
        ])

class WhatIfTab(ExplainerComponent):
    def __init__(self, explainer, title="What if...", name=None,
                    **kwargs):
        """Tab showing individual predictions and allowing edits 
            to the features...

        Args:
            explainer (Explainer): explainer object constructed with either
                        ClassifierExplainer() or RegressionExplainer()
            title (str, optional): Title of tab or page. Defaults to 
                        "Individual Predictions".
            name (str, optional): unique name to add to Component elements. 
                        If None then random uuid is generated to make sure 
                        it's unique. Defaults to None.
        """
        super().__init__(explainer, title, name)
        self.tab_id = "whatif"
        self.whatif = WhatIfComposite(explainer, **kwargs)
        self.register_components(self.whatif)
    
    def layout(self):
        return html.Div([
            self.whatif.layout()
        ])


class ShapDependenceTab(ExplainerComponent):
    def __init__(self, explainer, title='Feature Dependence', name=None,
                    tab_id="shap_dependence", 
                    depth=None, cats=True, **kwargs):
        """Tab showing both a summary of feature importance (aggregate or detailed).
        for each feature, and a shap dependence graph.

        Args:
            explainer (Explainer): explainer object constructed with either
                        ClassifierExplainer() or RegressionExplainer()
            title (str, optional): Title of tab or page. Defaults to 
                        'Feature Dependence'.
            name (str, optional): unique name to add to Component elements. 
                        If None then random uuid is generated to make sure 
                        it's unique. Defaults to None.
        """
        super().__init__(explainer, title, name)

        self.shap_overview = ShapDependenceComposite(
            explainer, depth=depth, cats=cats, **kwargs)
        self.register_components(self.shap_overview)

    def layout(self):
        return html.Div([
            self.shap_overview.layout()
        ])


class ShapInteractionsTab(ExplainerComponent):
    def __init__(self, explainer, title='Feature Interactions', name=None,
                    depth=None, cats=True, **kwargs):
        """[summary]

        Args:
            explainer (Explainer): explainer object constructed with either
                        ClassifierExplainer() or RegressionExplainer()
            title (str, optional): Title of tab or page. Defaults to 
                        'Feature Interactions'.
            name (str, optional): unique name to add to Component elements. 
                        If None then random uuid is generated to make sure 
                        it's unique. Defaults to None.
            depth (int, optional): default number of feature to display. Defaults to None.
            cats (bool, optional): default grouping of cats. Defaults to True.
        """
        super().__init__(explainer, title, name)
        self.interaction_overview = ShapInteractionsComposite(
                    explainer, depth=depth, cats=cats, **kwargs)
        self.register_components(self.interaction_overview)

    def layout(self):
        return html.Div([
            self.interaction_overview.layout()
        ])


class DecisionTreesTab(ExplainerComponent):
    def __init__(self, explainer, title="Decision Trees", name=None,
                        **kwargs):
        """Tab showing individual decision trees

        Args:
            explainer (Explainer): explainer object constructed with either
                        ClassifierExplainer() or RegressionExplainer()
            title (str, optional): Title of tab or page. Defaults to 
                        'Decision Trees'.
            name (str, optional): unique name to add to Component elements. 
                        If None then random uuid is generated to make sure 
                        it's unique. Defaults to None.
        """
        super().__init__(explainer, title, name)

        self.trees = DecisionTreesComposite(explainer, **kwargs)
        self.register_components(self.trees)

    def layout(self):
        return html.Div([
            self.trees.layout()
        ])