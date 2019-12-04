__all__ = ['ShapDependenceTab']

import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_table

from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from .dashboard_methods import *

class ShapDependenceTab:
    def __init__(self, explainer, standalone=False, tab_id="shap_dependence", title='Shap Dependence',
                 n_features=10, **kwargs):
        self.explainer = explainer
        self.standalone = standalone
        self.tab_id = tab_id
        self.title = title

        self.n_features = n_features
        self.kwargs = kwargs
        
    def layout(self):
        if self.standalone:
            return shap_dependence_layout(self.explainer, title=self.title, standalone=self.standalone, 
                                            n_features=self.n_features)
        else:
            return shap_dependence_layout(self.explainer,  
                                            n_features=self.n_features)
    
    def register_callbacks(self, app):
        shap_dependence_callbacks(self.explainer, app, standalone=self.standalone)


def shap_dependence_layout(explainer, 
            title=None, standalone=False, hide_selector=False,
            n_features=10, **kwargs):
    """return layout for shap dependence tab.
    
    :param explainer: ExplainerBunch
    :type explainer: ExplainerBunch 
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
    title_and_label_selector(explainer, title, standalone, hide_selector),
    dbc.Row([
        dbc.Col([
            html.H3('Shap Summary'),
            dbc.Row([
                dbc.Col([
                    dbc.Label("Depth:"),
                    dcc.Dropdown(id='dependence-scatter-depth',
                        options = [{'label': str(i+1), 'value':i+1} 
                                        for i in range(len(explainer.columns)-1)],
                        value=min(n_features, len(explainer.columns)-1))],
                    width=3), 
                dbc.Col([
                    dbc.Label("Grouping:"),
                    dbc.FormGroup(
                    [
                        dbc.RadioButton(
                            id='dependence-group-categoricals', 
                            className="form-check-input"),
                        dbc.Label("Group Cats",
                                html_for='dependence-group-categoricals',
                                className="form-check-label"),
                    ], check=True)],
                    width=3),
                ], form=True, justify="between"),

            dbc.Label('(Click on a dot to display dependece graph)'),
            dcc.Loading(id="loading-dependence-shap-scatter", 
                    children=[dcc.Graph(id='dependence-shap-scatter-graph')])
        ]),
        dbc.Col([
            html.H3('Shap Dependence Plot'),
            dbc.Row([
                dbc.Col([
                    html.Label('Plot dependence for column:'),
                    dcc.Dropdown(id='dependence-col', 
                        options=[{'label': col, 'value':col} 
                                    for col in explainer.mean_abs_shap_df()\
                                                                .Feature.tolist()],
                        value=explainer.mean_abs_shap_df().Feature[0])],
                    width=5), 
                dbc.Col([
                     html.Label('Color observation by column:'),
                    dcc.Dropdown(id='dependence-color-col', 
                        options=[{'label': col, 'value':col} 
                                    for col in explainer.mean_abs_shap_df()\
                                                                .Feature.tolist()],
                        value=explainer.mean_abs_shap_df().Feature.tolist()[0])],
                    width=5), 
                dbc.Col([
                    html.Label('Highlight:'),
                    dbc.Input(id='dependence-highlight-index', 
                            placeholder="Highlight index...",
                            debounce=True)]
                    , width=2) 
                ], form=True),
            
            dcc.Loading(id="loading-dependence-graph", 
                         children=[dcc.Graph(id='dependence-graph')]),
        ], width=6),
        ]),
    ],  fluid=True)


def shap_dependence_callbacks(explainer, app, 
             standalone=False, **kwargs):

    if standalone:
        label_selector_register_callback(explainer, app)

    @app.callback(
        [Output('dependence-shap-scatter-graph', 'figure'),
        Output('dependence-col', 'options'),
        Output('dependence-scatter-depth', 'options')],
        [Input('dependence-group-categoricals', 'checked'),
        Input('dependence-scatter-depth', 'value'),
        Input('label-store', 'data')])
    def update_dependence_shap_scatter_graph(cats, depth, pos_label):
        ctx = dash.callback_context
        if ctx.triggered:
            if depth is None: depth = 10
            plot = explainer.plot_shap_summary(topx=depth, cats=cats)
            trigger = ctx.triggered[0]['prop_id'].split('.')[0]

            if trigger=='dependence-scatter-depth':
                return (plot, dash.no_update, dash.no_update)
            else:
                col_options = [{'label': col, 'value':col} 
                            for col in explainer.mean_abs_shap_df(cats=cats)\
                                                        .Feature.tolist()]
                depth_options = [{'label': str(i+1), 'value':i+1} 
                            for i in range(len(col_options))]
                return (plot, col_options, depth_options)
        raise PreventUpdate


    @app.callback(
        [Output('dependence-highlight-index', 'value'),
         Output('dependence-col', 'value')],
        [Input('dependence-shap-scatter-graph', 'clickData')],
        [State('dependence-group-categoricals', 'checked')])
    def display_scatter_click_data(clickData, cats):
        if clickData is not None:
            idx = clickData['points'][0]['pointIndex']
            col = clickData['points'][0]['text'].split('=')[0]                             
            return (idx, col)
        raise PreventUpdate

    @app.callback(
        [Output('dependence-color-col', 'options'),
         Output('dependence-color-col', 'value')],
        [Input('dependence-col', 'value')],
        [State('dependence-group-categoricals', 'checked')])
    def set_color_col_dropdown(col, cats):
        sorted_interact_cols = explainer.shap_top_interactions(col, cats=cats)
        options = [{'label': col, 'value':col} 
                                    for col in sorted_interact_cols]
        value =   sorted_interact_cols[1]                                
        return (options, value)


    @app.callback(
        Output('dependence-graph', 'figure'),
        [Input('dependence-color-col', 'value'),
         Input('dependence-highlight-index', 'value'),
         Input('label-store', 'data')],
        [State('dependence-col', 'value'),
         State('dependence-group-categoricals', 'checked')])
    def update_dependence_graph(color_col, idx, pos_label, col, cats):
        if color_col is not None:
            return explainer.plot_shap_dependence(
                        col, color_col, highlight_idx=idx, cats=cats)
        raise PreventUpdate