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
                 n_features=10, cats=True, **kwargs):
        self.explainer = explainer
        self.standalone = standalone
        self.tab_id = tab_id
        self.title = title

        self.n_features = n_features
        self.cats = cats

        self.kwargs = kwargs
        if self.standalone:
            self.label_selector = TitleAndLabelSelector(explainer, title=title)
        
    def layout(self):
        return dbc.Container([
            self.label_selector.layout() if self.standalone else None,
            # need to add dummy to make callbacks on tab change work:
            html.Div(id='tabs') if self.standalone else None, 
            shap_interactions_layout(self.explainer, n_features=self.n_features, cats=self.cats, **self.kwargs)
        ], fluid=True)
    
    def register_callbacks(self, app):
        if self.standalone:
            self.label_selector.register_callbacks(app)
        shap_interactions_callbacks(self.explainer, app)


def shap_interactions_layout(explainer, 
            n_features=10, cats=True, **kwargs):
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
    dbc.Row([
        dbc.Col([
            html.H3('Shap Interaction Summary'),
            dbc.Row([
                dbc.Col([
                    dbc.Label("Feature"),
                    dcc.Dropdown(id='interaction-col', 
                        options=[{'label': col, 'value': col} 
                                    for col in explainer.columns_ranked(cats)],
                        value=explainer.columns_ranked(cats)[0])],
                    width=4), 
                dbc.Col([
                    dbc.Label("Depth:"),
                    dcc.Dropdown(id='interaction-summary-depth',
                        options = [{'label': str(i+1), 'value':i+1} 
                                        for i in range(len(explainer.columns_ranked(cats))-1)],
                        value=min(n_features, len(explainer.columns_ranked(cats))-1))],
                    width=2), 
                dbc.Col([
                    dbc.FormGroup(
                        [
                            dbc.Label("Summary Type"),
                            dbc.RadioItems(
                                options=[
                                    {"label": "Aggregate", "value": "aggregate"},
                                    {"label": "Detailed", "value": "detailed"},
                                ],
                                value="aggregate",
                                id="interaction-summary-type",
                                inline=True,
                            ),
                        ]
                    )
                ], width=3),
                dbc.Col([
                    dbc.Label("Grouping:"),
                    dbc.FormGroup(
                    [
                        dbc.RadioButton(
                            id='interaction-group-categoricals', 
                            className="form-check-input",
                            checked=cats),
                        dbc.Label("Group Cats",
                                html_for='interaction-group-categoricals',
                                className="form-check-label"),
                    ], check=True)],
                    width=3),
                ], form=True),
            dbc.Label('(Click on a dot to display interaction graph)'),
            dcc.Loading(id="loading-interaction-summary-scatter", 
                         children=[dcc.Graph(id='interaction-shap-summary-graph')])
        ], width=6),
        dbc.Col([
            html.H3('Shap Interaction Plots'),
            dbc.Row([
                dbc.Col([
                    dbc.Label("Interaction Feature"),
                    dcc.Dropdown(id='interaction-interact-col', 
                        options=[{'label': col, 'value':col} 
                                    for col in explainer.columns_ranked(cats)],
                        value=explainer.shap_top_interactions(explainer.columns_ranked(cats)[0], cats=cats)[1]
                    ),
                ], width=8), 
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


def shap_interactions_callbacks(explainer, app, standalone=False, n_features=10, **kwargs):

    @app.callback(
        [Output('interaction-col', 'options'),
         Output('interaction-col', 'value'),
         Output('interaction-summary-depth', 'options'),
         Output('interaction-summary-depth', 'value')],
        [Input('interaction-group-categoricals', 'checked')],
        [State('interaction-col', 'value'),
         State('tabs', 'value')])
    def update_col_options(cats, col, tab):
        cols = explainer.columns_ranked(cats)
        col_options = [{'label': col, 'value': col} for col in cols] 
        if col not in cols:
            col = explainer.inverse_cats(col)
        depth_options = [{'label': str(i+1), 'value': i+1} for i in range(len(cols))]
        depth = len(cols)-1
        return (col_options, col, depth_options, depth)

    @app.callback(
        [Output('interaction-shap-summary-graph', 'figure'),
         Output('interaction-interact-col', 'options')],
        [Input('interaction-summary-type', 'value'),
         Input('interaction-col', 'value'),
         Input('interaction-summary-depth', 'value'),
         Input('label-store', 'data')],
        [State('interaction-group-categoricals', 'checked')])
    def update_interaction_scatter_graph(summary_type, col, depth, pos_label, cats):
        if col is not None:
            if depth is None: 
                depth = len(explainer.columns_ranked(cats))-1 
            if summary_type=='aggregate':
                plot = explainer.plot_interactions(col, topx=depth, cats=cats)
            elif summary_type=='detailed':
                plot = explainer.plot_shap_interaction_summary(col, topx=depth, cats=cats)

            interact_cols = explainer.shap_top_interactions(col, cats=cats)
            interact_col_options = [{'label': col, 'value':col} for col in interact_cols]
            return plot, interact_col_options
        return None, None

    @app.callback(
        [Output('interaction-highlight-index', 'value'),
         Output('interaction-interact-col', 'value')],
        [Input('interaction-shap-summary-graph', 'clickData')])
    def display_scatter_click_data(clickdata):
        if clickdata is not None and clickdata['points'][0] is not None:
            if isinstance(clickdata['points'][0]['y'], float): # detailed
                idx = clickdata['points'][0]['pointIndex']
                col = clickdata['points'][0]['text'].split('=')[0]                             
                return (idx, col)
            elif  isinstance(clickdata['points'][0]['y'], str): # aggregate
                col = clickdata['points'][0]['y']
                return (dash.no_update, col) 
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