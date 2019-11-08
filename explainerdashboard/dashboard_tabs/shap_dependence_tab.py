__all__ = ['shap_dependence_tab', 'shap_dependence_tab_register_callbacks']

import dash
import dash_core_components as dcc
import dash_daq as daq
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_table

from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate


def shap_dependence_tab(explainer, n_features=10, **kwargs):
    """return layout for shap dependence tab.
    
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
            html.H3('Individual Shap Values'),
            html.Div([
                html.Div([
                    html.Label('(Click on a dot to display dependence graph)'),
                ], style = dict(width='48%', display='inline-block')),
                html.Div([
                    html.Label('no of columns:'),
                    dcc.Dropdown(id='dependence-scatter-depth',
                        options = [{'label': str(i+1), 'value':i+1} 
                                        for i in range(len(explainer.columns))],
                        value=min(n_features, len(explainer.columns))),
                ], style = dict(width='28%', display='inline-block')),
                html.Div([
                    daq.ToggleSwitch(
                        id='dependence-group-categoricals',
                        label='Group Categoricals'),
                ], style = dict(width='18%', display=cats_display)),
            ], style = {'width':'100%'}), 
            dcc.Loading(id="loading-dependence-shap-scatter", 
                    children=[dcc.Graph(id='dependence-shap-scatter-graph')])
        ]),
        dbc.Col([
            html.Div([
                html.Div([
                    html.Label('Plot dependence for column:'),
                    dcc.Dropdown(id='dependence-col', 
                        options=[{'label': col, 'value':col} 
                                    for col in explainer.mean_abs_shap_df()\
                                                                .Feature.tolist()],
                        value=explainer.mean_abs_shap_df().Feature[0]),
                ], style = dict(width='40%', display='inline-block')),
                html.Div([
                    html.Label('Color observation by column:'),
                    dcc.Dropdown(id='dependence-color-col', 
                        options=[{'label': col, 'value':col} 
                                    for col in explainer.mean_abs_shap_df()\
                                                                .Feature.tolist()],
                        value=explainer.mean_abs_shap_df().Feature.tolist()[0])
                ], style = dict(width='40%', display='inline-block')),
                html.Div([
                    html.Label('Highlight:'),
                    dbc.Input(id='dependence-highlight-index', 
                            placeholder="Highlight index...",
                            debounce=True),
                ], style = dict(width='20%', display='inline-block')) 
            ]), 
            dcc.Graph(id='dependence-graph')
        ])
    ]), 
    ] ,  fluid=True)


def shap_dependence_tab_register_callbacks(explainer, app, **kwargs):
    @app.callback(
        [Output('dependence-shap-scatter-graph', 'figure'),
        Output('dependence-col', 'options'),
        Output('dependence-scatter-depth', 'options')],
        [Input('dependence-group-categoricals', 'value'),
        Input('dependence-scatter-depth', 'value')])
    def update_dependence_shap_scatter_graph(cats, depth):
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
        [State('dependence-group-categoricals', 'value')])
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
        [State('dependence-group-categoricals', 'value')])
    def set_color_col_dropdown(col, cats):
        sorted_interact_cols = explainer.shap_top_interactions(col, cats=cats)
        options = [{'label': col, 'value':col} 
                                    for col in sorted_interact_cols]
        value =   sorted_interact_cols[1]                                
        return (options, value)


    @app.callback(
        Output('dependence-graph', 'figure'),
        [Input('dependence-color-col', 'value'),
         Input('dependence-highlight-index', 'value')],
        [State('dependence-col', 'value'),
         State('dependence-group-categoricals', 'value')])
    def update_dependence_graph(color_col, idx, col, cats):
        if color_col is not None:
            return explainer.plot_shap_dependence(
                        col, color_col, highlight_idx=idx, cats=cats)
        raise PreventUpdate