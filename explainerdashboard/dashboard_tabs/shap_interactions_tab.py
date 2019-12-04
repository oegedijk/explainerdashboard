__all__ = ['ShapInteractionsTab']

import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_table

from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from .dashboard_methods import *


class ShapInteractionsTab:
    def __init__(self, explainer, standalone=False, tab_id="shap_interactions", title='Shap Interactions',
                 n_features=10, **kwargs):
        self.explainer = explainer
        self.standalone = standalone
        self.tab_id = tab_id
        self.title = title

        self.n_features = n_features
        self.kwargs = kwargs
        
    def layout(self):
        if self.standalone:
            return shap_interactions_layout(self.explainer, title=self.title, standalone=self.standalone, 
                                            n_features=self.n_features)
        else:
            return shap_interactions_layout(self.explainer,  
                                            n_features=self.n_features)
    
    def register_callbacks(self, app):
        shap_interactions_callbacks(self.explainer, app, standalone=self.standalone)


def shap_interactions_layout(explainer, 
            title=None, standalone=False, hide_selector=False,
            n_features=10, **kwargs):
    """return layout for shap interactions tab.
    
    :param explainer: ExplainerBunch
    :type explainer: ExplainerBunch 
    :param title: title displayed on top of page, defaults to None
    :type title: str
    :param standalone: when standalone layout, include a a label_store, defaults to False
    :type standalone: bool
    :param hide_selector: if model is a classifier, optionally hide the positive label selector, defaults to False
    :type hide_selector: bool
    :param n_features: default number of features to display, defaults to 10
    :type n_features: int, optional
    :rtype: dbc.Container
    """
    cats_display = 'none' if explainer.cats is None else 'inline-block'
    return dbc.Container([
    #dbc.Row([dbc.Col([html.H3('Shap Interaction Values')])]),
    title_and_label_selector(explainer, title, standalone, hide_selector),
    dbc.Row([
        dbc.Col([
            html.H3('Shap Interaction Summary'),
            dbc.Row([
                dbc.Col([
                    dbc.Label("Feature"),
                    dcc.Dropdown(id='interaction-col', 
                        options=[{'label': col, 'value':col} 
                                    for col in explainer.mean_abs_shap_df()\
                                                                .Feature.tolist()],
                        value=explainer.mean_abs_shap_df().Feature[0])],
                    width=6), 
                dbc.Col([
                    dbc.Label("Depth:"),
                    dcc.Dropdown(id='interaction-scatter-depth',
                        options = [{'label': str(i+1), 'value':i+1} 
                                        for i in range(len(explainer.columns)-1)],
                        value=min(n_features, len(explainer.columns)-1))],
                    width=3), 
                dbc.Col([
                    dbc.Label("Grouping:"),
                    dbc.FormGroup(
                    [
                        dbc.RadioButton(
                            id='interaction-group-categoricals', 
                            className="form-check-input"),
                        dbc.Label("Group Cats",
                                html_for='interaction-group-categoricals',
                                className="form-check-label"),
                    ], check=True)],
                    width=3),
                ], form=True),
            dbc.Label('(Click on a dot to display interaction graph)'),
            dcc.Loading(id="loading-interaction-shap-scatter", 
                         children=[dcc.Graph(id='interaction-shap-scatter-graph')])
        ], width=6),
        dbc.Col([
            html.H3('Shap Interaction Plots'),
            dbc.Row([
                dbc.Col([
                    dbc.Label("Interaction Feature"),
                    dcc.Dropdown(id='interaction-interact-col', 
                        options=[{'label': col, 'value':col} 
                                    for col in explainer.mean_abs_shap_df()\
                                                                .Feature.tolist()],
                        value=explainer.mean_abs_shap_df().Feature[0])],
                    width=8), 
                dbc.Col([
                    dbc.Label("Highlight index:"),
                    dbc.Input(id='interaction-highlight-index', 
                        placeholder="Highlight index...", debounce=True)],
                    width=4), 
                ], form=True),
            
            dcc.Loading(id="loading-interaction-graph", 
                         children=[dcc.Graph(id='interaction-graph')]),
            dcc.Loading(id="loading-reverse-interaction-graph", 
                         children=[dcc.Graph(id='reverse-interaction-graph')]),
        ], width=6)
    ]), 
    ],  fluid=True)


def shap_interactions_callbacks(explainer, app, 
             standalone=False, n_features=10, **kwargs):
    if standalone:
        label_selector_register_callback(explainer, app)

    @app.callback(
        [Output('interaction-col', 'options'),
         Output('interaction-scatter-depth', 'options')],
        [Input('interaction-group-categoricals', 'checked')])
    def update_col_options(cats):
        col_options = [{'label': col, 'value':col} 
                                for col in explainer.mean_abs_shap_df(cats=cats)\
                                                            .Feature.tolist()] 
        depth_options = [{'label': str(i+1), 'value':i+1} for i in range(len(col_options))]
        return (col_options, depth_options )

    @app.callback(
        [Output('interaction-shap-scatter-graph', 'figure'),
         Output('interaction-interact-col', 'options')],
        [Input('interaction-col', 'value'),
         Input('interaction-scatter-depth', 'value'),
         Input('label-store', 'data')],
        [State('interaction-group-categoricals', 'checked')])
    def update_interaction_scatter_graph(col,  depth, pos_label, cats):
        if col is not None:
            if depth is None: depth = n_features
            plot = plot = explainer.plot_shap_interaction_summary(col, topx=depth, cats=cats)
            interact_cols = explainer.shap_top_interactions(col, cats=cats)
            interact_col_options = [{'label': col, 'value':col} for col in interact_cols]
            return plot, interact_col_options
        raise PreventUpdate


    @app.callback(
        [Output('interaction-highlight-index', 'value'),
         Output('interaction-interact-col', 'value')],
        [Input('interaction-shap-scatter-graph', 'clickData')])
    def display_scatter_click_data(clickData):
        if clickData is not None:
            #return str(clickData)
            idx = clickData['points'][0]['pointIndex']
            col = clickData['points'][0]['text'].split('=')[0]
            return (idx, col)
        raise PreventUpdate


    @app.callback(
        [Output('interaction-graph', 'figure'),
         Output('reverse-interaction-graph', 'figure')],
        [Input('interaction-interact-col', 'value'),
         Input('interaction-highlight-index', 'value'),
         Input('label-store', 'data')],
        [State('interaction-col', 'value'),
         State('interaction-group-categoricals', 'checked')])
    def update_dependence_graph(interact_col, index, pos_label, col, cats):
        if interact_col is not None:
            return (explainer.plot_shap_interaction_dependence(
                        col, interact_col, highlight_idx=index, cats=cats),
                    explainer.plot_shap_interaction_dependence(
                        interact_col, col, highlight_idx=index, cats=cats))
        raise PreventUpdate