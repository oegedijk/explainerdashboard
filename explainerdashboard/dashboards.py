# -*- coding: utf-8 -*-

__all__ = ['ExplainerDashboard', 
            'RandomForestDashboard']

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
# then extended to deal with multiple inheritance
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
    """Constructs a dashboard out of an ExplainerBunch object. You can indicate
    which tabs to include, and pass kwargs to individual tabs.
    """
    def __init__(self, explainer, title='Model Explainer', *,  
                    model_summary=True,  
                    contributions=True,
                    shap_dependence=True,
                    shap_interaction=False,
                    **kwargs):
        """Constructs an ExplainerDashboard.
        
        :param explainer: an ExplainerBunch object
        :param title: Title of the dashboard, defaults to 'Model Explainer'
        :type title: str, optional
        :param model_summary: display model_summary tab or not, defaults to True
        :type model_summary: bool, optional
        :param contributions: display individual contributions tab or not, defaults to True
        :type contributions: bool, optional
        :param shap_dependence: display shap dependence tab or not, defaults to True
        :type shap_dependence: bool, optional
        :param shap_interaction: display tab interaction tab or not. Note: shap interaction values can take a long time to compute, so therefore this tab defaults to False, defaults to False
        :type shap_interaction: bool, optional
        """
        self.explainer=explainer
        self.title = title

        self.model_summary = model_summary
        self.contributions = contributions
        self.shap_dependence = shap_dependence
        self.shap_interaction = shap_interaction

        self.kwargs=kwargs

        # calculate properties before starting dashboard:
        
        if shap_dependence or contributions or model_summary:
            _ = explainer.shap_values, explainer.preds 
            if explainer.cats is not None:
                _ = explainer.shap_values_cats
            if explainer.is_classifier:
                _ = explainer.pred_probas
        if model_summary:
            _ = explainer.permutation_importances
            if explainer.cats is not None:
                _ = explainer.permutation_importances_cats
        if shap_interaction:
            _ = explainer.shap_interaction_values
            if explainer.cats is not None:
                _ = explainer.shap_interaction_values_cats

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
        """Inserts the layouts for the appropriate tabs. These can be extended
        in inherited ExplainerDashboard classes to easily add new tabs.
        
        :param tabs: a list of tab lauout that will be appended to
        :type tabs: list
        """
        if self.model_summary:
            tabs.append(dcc.Tab(
                    children=model_summary_tab(self.explainer, **self.kwargs),
                    label='Model Overview', 
                    id='model_tab'))
        if self.contributions:
            tabs.append(dcc.Tab(
                    children=contributions_tab(self.explainer, **self.kwargs), 
                    label='Individual Contributions', 
                    id='contributions_tab'))
        if self.shap_dependence:
            tabs.append(dcc.Tab(
                    children=shap_dependence_tab(self.explainer, **self.kwargs), 
                    label='Dependence Plots', 
                    id='dependence_tab'))
        if self.shap_interaction:
            tabs.append(dcc.Tab(
                    children=shap_interactions_tab(self.explainer, **self.kwargs), 
                    label='Interactions graphs', 
                    id='interactions_tab'))
        
    def register_callbacks(self):
        """Registers the appropriate callbacks for the different tabs to the
        Dash app. Can be easily extended by inheriting classes to add more tabs.
        """
        if self.model_summary: 
            model_summary_tab_register_callbacks(self.explainer, self.app, **self.kwargs)
        if self.contributions: 
            contributions_tab_register_callbacks(self.explainer, self.app, **self.kwargs)
        if self.shap_dependence: 
            shap_dependence_tab_register_callbacks(self.explainer, self.app, **self.kwargs)
        if self.shap_interaction: 
            shap_interactions_tab_register_callbacks(self.explainer, self.app, **self.kwargs)
        
    def run(self, port=8050):
        """Starts the dashboard using the built-in Flask server on localhost:port
        
        :param port: the port to run the dashboard on, defaults to 8050
        :type port: int, optional
        """
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
        """The RandomForestDashboard comes with an additional potential tab
        called shadow_trees where individual DecisionTrees within the RandomForest
        can be viewed. 
        
        :param shadow_trees: [description], defaults to True
        :type shadow_trees: bool, optional
        """
        self.shadow_trees = shadow_trees
        if shadow_trees:
            _ = self.shadow_trees 
        super().__init__(explainer, title, **kwargs)        

    def insert_tab_layouts(self, tabs):
        super().insert_tab_layouts(tabs)
        if self.shadow_trees:
            tabs.append(dcc.Tab(
                children=shadow_trees_tab(self.explainer, **self.kwargs), 
                label='Individual Trees', 
                id='trees_tab'))
        
    def register_callbacks(self):
        super().register_callbacks()
        if self.shadow_trees: 
            shadow_trees_tab_register_callbacks(self.explainer, self.app, **self.kwargs)



