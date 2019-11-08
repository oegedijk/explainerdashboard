__all__ = ['shap_interactions_tab', 'shap_interactions_tab_register_callbacks']

import dash
import dash_core_components as dcc
import dash_daq as daq
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_table

from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate


def shap_interactions_tab(explainer, n_features=10, **kwargs):
    """return layout for shap interactions tab.
    
    :param explainer: ExplainerBunch
    :type explainer: ExplainerBunch 
    :param n_features: default number of features to display, defaults to 10
    :type n_features: int, optional
    :rtype: dbc.Container
    """
    cats_display = 'none' if explainer.cats is None else 'inline-block'
    return dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H3('Shap Interaction Values'),
            html.Label('Display shap interaction values for column:'),
            html.Div([
                html.Div([
                    dcc.Dropdown(id='interaction-col', 
                        options=[{'label': col, 'value':col} 
                                    for col in explainer.mean_abs_shap_df()\
                                                                .Feature.tolist()],
                        value=explainer.mean_abs_shap_df().Feature[0])
                ], style = {'width': '50%', 'display': 'inline-block'}),  
                html.Div([
                    html.Label('no of columns:'),
                    dcc.Dropdown(id='interaction-scatter-depth',
                        options = [{'label': str(i+1), 'value':i+1} 
                                        for i in range(len(explainer.columns)-1)],
                        value=min(n_features, len(explainer.columns)-1)),
                ], style = dict(width='30%', display='inline-block')),  
                html.Div([
                    daq.ToggleSwitch(
                        id='interaction-group-categoricals',
                        label='Group Categoricals')
                ], style = {'width': '20%', 'display': cats_display}),
            ], style = {'width':'100%'}),
            html.Label('(Click on a dot to display interaction graph)'),
            dcc.Loading(id="loading-interaction-shap-scatter", 
                         children=[dcc.Graph(id='interaction-shap-scatter-graph')])
        ], width=6),
        dbc.Col([
            html.Div([
                html.Div([
                    html.Label('Show interaction with column:'),
                    dcc.Dropdown(id='interaction-interact-col', 
                        options=[{'label': col, 'value':col} 
                                    for col in explainer.mean_abs_shap_df()\
                                                                .Feature.tolist()],
                        value=explainer.mean_abs_shap_df().Feature[0]),
                ], style = {'width': '80%', 'display': 'inline-block'}),
                html.Div([
                    html.Label('Highlight:'),
                    dbc.Input(id='interaction-highlight-index', 
                        placeholder="Highlight index...", debounce=True),
                ], style = {'width': '20%', 'display': 'inline-block'}),
            ]),
            
            html.Label('Shap interaction values:'),
            dcc.Loading(id="loading-interaction-graph", 
                         children=[dcc.Graph(id='interaction-graph')]),
        ], width=6)
    ]), 
    ],  fluid=True)


def shap_interactions_tab_register_callbacks(explainer, app, **kwargs):
    @app.callback(
        [Output('interaction-shap-scatter-graph', 'figure'),
        Output('interaction-interact-col', 'options'),
        Output('interaction-scatter-depth', 'options'),
        Output('interaction-col', 'options'),],
        [Input('interaction-col', 'value'),
        Input('interaction-group-categoricals', 'value'),
        Input('interaction-scatter-depth', 'value')])
    def update_interaction_scatter_graph(col, cats, depth):
        ctx = dash.callback_context
        if ctx.triggered:
            if depth is None: depth = 10
            plot = plot = explainer.plot_shap_interaction_summary(col, topx=depth, cats=cats)
            trigger = ctx.triggered[0]['prop_id'].split('.')[0]

            if trigger=='interaction-scatter-depth':
                return (plot, dash.no_update, dash.no_update, dash.no_update)
            elif trigger=='interaction-group-categoricals':
                interact_cols = explainer.shap_top_interactions(col, cats=cats)
                col_options = [{'label': col, 'value':col} 
                                        for col in explainer.mean_abs_shap_df(cats=cats)\
                                                                    .Feature.tolist()] 
                interact_col_options = [{'label': col, 'value':col} for col in interact_cols]
                depth_options = [{'label': str(i+1), 'value':i+1} for i in range(len(col_options))]
                return (plot, interact_col_options, depth_options, col_options)
            elif trigger=='interaction-col':
                interact_cols = explainer.shap_top_interactions(col, cats=cats)
                interact_col_options = [{'label': col, 'value':col} for col in interact_cols]
                return (plot, interact_col_options, dash.no_update, dash.no_update)
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
        Output('interaction-graph', 'figure'),
        [Input('interaction-col', 'value'),
        Input('interaction-interact-col', 'value'),
        Input('interaction-highlight-index', 'value')],
        [State('interaction-group-categoricals', 'value')])
    def update_dependence_graph(col, interact_col, idx, cats):
        if interact_col is not None:
            return explainer.plot_shap_interaction_dependence(
                    col, interact_col, highlight_idx=idx, cats=cats)
        raise PreventUpdate