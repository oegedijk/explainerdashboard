__all__ = ['model_summary_tab', 'model_summary_tab_register_callbacks']

import dash
import dash_core_components as dcc
import dash_daq as daq
import dash_bootstrap_components as dbc
import dash_html_components as html

from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate


def model_summary_tab(explainer, bin_size=0.1, quantiles=10, 
                        cutoff=0.5, n_features=15, **kwargs):
    """returns layout for model summary tab
    
    :param explainer: ExplainerBunch object for which to build layout for
    :type explainer: ExplainerBUnch
    :param bin_size: default bin_size for precision_plot (classifier only), defaults to 0.1
    :type bin_size: float, optional
    :param quantiles: default number of quantiles for precision_plot (classifier only), defaults to 10
    :type quantiles: int, optional
    :param cutoff: cutoff used for confusion_matrix, roc_auc and pr_auc plots (classifier only), defaults to 0.5
    :type cutoff: float, optional
    :param n_features: default number of features to display, defaults to 15
    :type n_features: int, optional
    :rtype: dbc.Container
    """
    # hide cats toggle if not cats defined:
    cats_display = 'none' if explainer.cats is None else 'inline-block'

    if explainer.is_classifier:
        model_stats_rows = [
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.Label('Bin size:'),
                        dcc.Slider(id='precision-binsize', 
                                    min = 0.01, max = 0.5, step=0.01, value=bin_size,
                                    marks={0.01: '0.01', 0.05: '0.05', 0.10: '0.10',
                                        0.20: '0.20', 0.25: '0.25' , 0.33: '0.33', 
                                        0.5: '0.5'}, 
                                    included=False,
                                    tooltip = {'always_visible' : True})
                    ], style={'margin': 20,}),
                ], width=4),
                dbc.Col([
                    html.Div([
                            dcc.RadioItems(
                            id='binsize-or-quantiles',
                            options=[
                                {'label': 'Bin Size', 
                                'value': 'bin_size'},
                                {'label': 'Quantiles', 
                                'value': 'quantiles'}
                            ],
                            value='bin_size',
                            labelStyle={'display': 'inline-block'}),
                        ], style={'align': 'center', }),
                ]),
                dbc.Col([
                    html.Div([
                        html.Label('Quantiles:'),
                        dcc.Slider(id='precision-quantiles', 
                                    min = 1, max = 20, step=1, value=quantiles,
                                    marks={1: '1', 5: '5', 10: '10', 15: '15', 20:'20'}, 
                                    included=False,
                                    tooltip = {'always_visible' : True})
                    ], style={'margin': 20, }),
                ], width=4)
            ], align="center",),
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id='precision-graph'),
                    html.Div([
                        html.Label('Cutoff:'),
                        dcc.Slider(id='precision-cutoff', 
                                    min = 0.01, max = 0.99, step=0.01, value=cutoff,
                                    marks={0.01: '0.01', 0.25: '0.25', 0.50: '0.50',
                                            0.75: '0.75', 0.99: '0.99'}, 
                                    included=False,
                                    tooltip = {'always_visible' : True})
                    ], style={'margin': 20}),
                ])
            ]),
            dbc.Row([
                dbc.Col([
                    dcc.Loading(id="loading-confusionmatrix-graph", 
                                children=[dcc.Graph(id='confusionmatrix-graph')]),
                    dcc.RadioItems(
                        id='confusionmatrix-normalize',
                        options=[{'label': o, 'value': o} 
                                    for o in ['Counts', 'Normalized']],
                        value='Counts',
                        labelStyle={'display': 'inline-block'}
                    ),
                ]),
                dbc.Col([
                    dcc.Loading(id="loading-roc-auc-graph", 
                                children=[dcc.Graph(id='roc-auc-graph')]),
                ]),
                dbc.Col([
                    dcc.Loading(id="loading-pr-auc-graph", 
                                children=[dcc.Graph(id='pr-auc-graph')]),
                ]),
            ]),
        ] 
    else:
        model_stats_rows = [dbc.Row([])]

    return dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H2('Model overview:'),
        ]),
    ]),
    
    *model_stats_rows,

    dbc.Row([
        dbc.Col([
            html.Label('Model importances:'),
            html.Div([
                html.Div([
                    dcc.RadioItems(
                    id='permutation-or-shap',
                    options=[
                        {'label': 'Permutation Importances', 
                        'value': 'permutation'},
                        {'label': 'SHAP values', 
                        'value': 'shap'}
                    ],
                    value='shap',
                    labelStyle={'display': 'inline-block'}),
                ], style={'width':'30%', 'display': 'inline-block'}),
                html.Div([
                    html.Div('Select max number of importances to display:'),
                    dcc.Dropdown(id='importance-tablesize',
                                options = [{'label': str(i+1), 'value':i+1} 
                                            for i in range(len(explainer.columns))],
                                value=min(n_features, len(explainer.columns))),

                ], style={'width':'30%', 'display': 'inline-block'}),
                html.Div([
                    daq.ToggleSwitch(
                        id='group-categoricals',
                        label='Group Categoricals',
                    ),
                ], style={'width':'30%', 'display': cats_display}),
            ]),
            
            html.Div([
                dcc.Graph(id='importances-graph'),
            ])
        ])
    ]),  
    ], fluid=True)


def model_summary_tab_register_callbacks(explainer, app, **kwargs):
    if explainer.is_classifier:
        @app.callback(
            Output('precision-graph', 'figure'),
            [Input('precision-binsize', 'value'),
            Input('precision-quantiles', 'value'),
            Input('binsize-or-quantiles', 'value'),
            Input('precision-cutoff', 'value')],
        )
        def update_precision_graph(bin_size, quantiles, bins, cutoff):
            if bins=='bin_size':
                return explainer.plot_precision(bin_size=bin_size, cutoff=cutoff)
            elif bins=='quantiles':
                return explainer.plot_precision(quantiles=quantiles, cutoff=cutoff)
            raise PreventUpdate


        @app.callback(
            [Output('confusionmatrix-graph', 'figure'),
            Output('roc-auc-graph', 'figure'),
            Output('pr-auc-graph', 'figure')],
            [Input('precision-cutoff', 'value'),
            Input('confusionmatrix-normalize', 'value')],
        )
        def update_precision_graph(cutoff, normalized):
            confmat_plot = explainer.plot_confusion_matrix(
                                cutoff=cutoff, normalized=normalized=='Normalized')
            roc_auc_plot = explainer.plot_roc_auc(cutoff=cutoff)
            pr_auc_plot = explainer.plot_pr_auc(cutoff=cutoff)
            return (confmat_plot, roc_auc_plot, pr_auc_plot)


    @app.callback(  
        Output('importances-graph', 'figure'),
        [Input('importance-tablesize', 'value'),
        Input('group-categoricals', 'value'),
        Input('permutation-or-shap', 'value')]
    )
    def update_importances(tablesize, cats, permutation_shap): 
        return explainer.plot_importances(
                    type=permutation_shap, topx=tablesize, cats=cats)