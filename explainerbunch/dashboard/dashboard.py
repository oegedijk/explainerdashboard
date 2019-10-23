# -*- coding: utf-8 -*-

__all__ = ['ExplainerDashboard']

import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html

import plotly.io as pio

from .model_summary_tab import *
from .contributions_tab import *
from .shap_dependence_tab import *
from .shap_interactions_tab import *
from .shadow_trees_tab import *

pio.templates.default = "none" 

class ExplainerDashboard:
    def __init__(self, explainer, title='Model Explainer',
                    include_model_summary=True,
                    include_contributions=True,
                    include_shap_dependence=True,
                    include_shap_interaction=True,
                    include_shadow_trees=True):
        """
        ExplainerDashboard
        """
        assert (include_model_summary or
                    include_contributions or
                    include_shap_dependence or
                    include_shap_interaction or
                    include_shadow_trees), 'You need to at least include 1 tab'

        self.explainer=explainer
        self.title = title

        tabs = []
        if include_model_summary:
            tabs.append(dcc.Tab(children=model_summary_tab(self.explainer),
                            label='Model Overview', 
                            id='model_tab'))

        if include_contributions:
            tabs.append(dcc.Tab(children=contributions_tab(self.explainer), 
                            label='Individual Contributions', 
                            id='contributions_tab'))

        if include_shap_dependence:
            tabs.append(dcc.Tab(children=shap_dependence_tab(self.explainer), 
                            label='Dependence Plots', 
                            id='dependence_tab'))

        if include_shap_interaction:
            tabs.append(dcc.Tab(children=shap_interactions_tab(self.explainer), 
                            label='Interactions graphs', 
                            id='interactions_tab'))

        if include_shadow_trees:
            tabs.append(dcc.Tab(children=shadow_trees_tab(self.explainer), 
                            label='Individual Trees', 
                            id='trees_tab'))

        self.app = dash.Dash(__name__.split('.')[0], static_folder='assets')

        self.app.config['suppress_callback_exceptions']=True
        self.app.css.config.serve_locally = True
        self.app.scripts.config.serve_locally = True
        self.app.title = title

        self.app.layout = dbc.Container([
            dbc.Row([html.H1(title)]),
            dcc.Tabs(tabs, id="tabs")
        ],  fluid=True)

        if include_model_summary: 
            model_summary_tab_register_callbacks(self.explainer, self.app)

        if include_contributions: 
            contributions_tab_register_callbacks(self.explainer, self.app)

        if include_shap_dependence: 
            shap_dependence_tab_register_callbacks(self.explainer, self.app)

        if include_shap_interaction: 
            shap_interactions_tab_register_callbacks(self.explainer, self.app)

        if include_shadow_trees: 
            shadow_trees_tab_register_callbacks(self.explainer, self.app)

    def run(self, port=8050):
        print(f"Running {self.title} on http://localhost:{port}")
        self.app.run_server(port=port)
 
