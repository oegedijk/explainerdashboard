# -*- coding: utf-8 -*-

__all__ = ['ExplainerDashboard', 
            'RandomForestDashboard', 
            'ClassifierDashboard',
            'RandomForestClassifierDashboard']

import inspect 
import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html

import plotly.io as pio

from .dashboard_tabs.model_summary_tab import *
from .dashboard_tabs.contributions_tab import *
from .dashboard_tabs.shap_dependence_tab import *
from .dashboard_tabs.shap_interactions_tab import *
from .dashboard_tabs.shadow_trees_tab import *

pio.templates.default = "none" 

# Stolen from https://www.fast.ai/2019/08/06/delegation/
def delegates(to=None, keep=False):
    "Decorator: replace `**kwargs` in signature with params from `to`"
    def _f(f):
        from_f = f.__init__ if to is None else f
        sig = inspect.signature(from_f)
        sigd = dict(sig.parameters)
        k = sigd.pop('kwargs')
        if to is None:
            for base_cls in f.__bases__:
                to_f = base_cls.__init__
                s2 = {k:v for k,v in inspect.signature(to_f).parameters.items()
                    if v.default != inspect.Parameter.empty and k not in sigd}
                sigd.update(s2)
        else:
            to_f = to
            s2 = {k:v for k,v in inspect.signature(to_f).parameters.items()
                if v.default != inspect.Parameter.empty and k not in sigd}
            sigd.update(s2)
        if keep: sigd['kwargs'] = k
        from_f.__signature__ = sig.replace(parameters=sigd.values())
        return f
    return _f


class ExplainerDashboard:
    def __init__(self, explainer, title='Model Explainer', *,    
                    contributions=True,
                    shap_dependence=True,
                    shap_interaction=True):
        self.explainer=explainer
        self.title = title

        self.contributions = contributions
        self.shap_dependence = shap_dependence
        self.shap_interaction = shap_interaction

        # calculate properties before starting dashboard:
        if shap_dependence or contributions:
            _ = explainer.shap_values, explainer.preds 
        if shap_interaction:
            _ = explainer.shap_interaction_values

        self.app = dash.Dash(__name__)
        self.app.config['suppress_callback_exceptions']=True
        self.app.css.config.serve_locally = True
        self.app.scripts.config.serve_locally = True
        self.app.title = title

        self.tabs = []
        self.insert_tab_layouts(self.tabs)

        self.app.layout = dbc.Container([
            dbc.Row([html.H1(title)]),
            dcc.Tabs(self.tabs, id="tabs")
        ],  fluid=True)

        self.register_callbacks()

    def insert_tab_layouts(self, tabs):
        if self.contributions:
            tabs.append(dcc.Tab(children=contributions_tab(self.explainer), 
                            label='Individual Contributions', 
                            id='contributions_tab'))
        if self.shap_dependence:
            tabs.append(dcc.Tab(children=shap_dependence_tab(self.explainer), 
                            label='Dependence Plots', 
                            id='dependence_tab'))
        if self.shap_interaction:
            tabs.append(dcc.Tab(children=shap_interactions_tab(self.explainer), 
                            label='Interactions graphs', 
                            id='interactions_tab'))
        
    def register_callbacks(self):
        if self.contributions: 
            contributions_tab_register_callbacks(self.explainer, self.app)
        if self.shap_dependence: 
            shap_dependence_tab_register_callbacks(self.explainer, self.app)
        if self.shap_interaction: 
            shap_interactions_tab_register_callbacks(self.explainer, self.app)
        
    def run(self, port=8050):
        print(f"Running {self.title} on http://localhost:{port}")
        self.app.run_server(port=port)


@delegates()
class RandomForestDashboard(ExplainerDashboard):
    """
    Adds a shadow_trees tab to ExplainerDashboard where you can inspect
    individal DecisionTrees in the RandomForest.
    """
    def __init__(self, explainer, title='Model Explainer', *, 
                   shadow_trees=True, **kwargs):
        self.shadow_trees = shadow_trees
        if shadow_trees:
            _ = self.shadow_trees 
        super().__init__(explainer, title, **kwargs)        

    def insert_tab_layouts(self, tabs):
        super().insert_tab_layouts(tabs)
        if self.shadow_trees:
            tabs.append(dcc.Tab(children=shadow_trees_tab(self.explainer), 
                            label='Individual Trees', 
                            id='trees_tab'))
        
    def register_callbacks(self):
        super().register_callbacks()
        if self.shadow_trees: 
            shadow_trees_tab_register_callbacks(self.explainer, self.app)


@delegates()
class ClassifierDashboard(ExplainerDashboard):
    """
    Adds a Model Overview tab to ExplainerDashboard with different plots
    for classifier performance (confusion matrix, roc_auc plot, etc)
    """
    def __init__(self, explainer, title='Model Explainer',  
                    classifier_summary=True, **kwargs):
        self.classifier_summary = classifier_summary
        if classifier_summary:
            _ = explainer.pred_probas, explainer.permutation_importances
            if explainer.cats is not None:
                _ = explainer.permutation_importances_cats

        super().__init__(explainer, title, **kwargs)

    def insert_tab_layouts(self, tabs):
        if self.classifier_summary:
            tabs.append(dcc.Tab(children=model_summary_tab(self.explainer),
                            label='Model Overview', 
                            id='model_tab'))
        super().insert_tab_layouts(tabs)
        
    def register_callbacks(self):
        if self.classifier_summary: 
            model_summary_tab_register_callbacks(self.explainer, self.app)
        super().register_callbacks()



@delegates()
class RandomForestClassifierDashboard(ClassifierDashboard, 
                        RandomForestDashboard):
    """
    Adds both a Classifier Model Overview tab and a Shadow Trees tab.
    """
    def __init__(self, explainer, title='Model Explainer',  **kwargs):
        super().__init__(explainer, title, **kwargs)


