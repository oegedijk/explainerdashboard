__all__ = ['model_summary_tab', 'model_summary_tab_register_callbacks']

import dash
import dash_core_components as dcc
import dash_daq as daq
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_table

from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate


def model_summary_tab(self):
    return dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H2('Model overview:'),
            html.Div([
                html.Label('Bin size:'),
                dcc.Slider(id='precision-binsize', 
                            min = 0.01, max = 0.5, step=0.01, value=0.05,
                            marks={0.01: '0.01', 0.05: '0.05', 0.10: '0.10',
                                0.20: '0.20', 0.25: '0.25' , 0.33: '0.33', 
                                0.5: '0.5'}, 
                            included=False,
                            tooltip = {'always_visible' : True})
            ], style={'margin': 20}),
            dcc.Graph(id='precision-graph'),
            html.Div([
                html.Label('Cutoff:'),
                dcc.Slider(id='precision-cutoff', 
                            min = 0.01, max = 0.99, step=0.01, value=0.5,
                            marks={0.01: '0.01', 0.25: '0.25', 0.50: '0.50',
                                    0.75: '0.75', 0.99: '0.99'}, 
                            included=False,
                            tooltip = {'always_visible' : True})
            ], style={'margin': 20}),
        ]),  
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
                                            for i in range(len(self.columns))],
                                value=min(15, len(self.columns))),

                ], style={'width':'30%', 'display': 'inline-block'}),
                html.Div([
                    daq.ToggleSwitch(
                        id='group-categoricals',
                        label='Group Categoricals',
                    ),
                ], style={'width':'30%', 'display': 'inline-block'}),
            ]),
            
            html.Div([
                dcc.Graph(id='importances-graph'),
            ])
        ])
    ]),  
    ], fluid=True)


def model_summary_tab_register_callbacks(self, app):
    @app.callback(
        Output('precision-graph', 'figure'),
        [Input('precision-binsize', 'value'),
        Input('precision-cutoff', 'value')],
    )
    def update_precision_graph(bin_size, cutoff):
        return self.plot_precision(bin_size, cutoff)


    @app.callback(
        [Output('confusionmatrix-graph', 'figure'),
        Output('roc-auc-graph', 'figure'),
        Output('pr-auc-graph', 'figure')],
        [Input('precision-cutoff', 'value'),
        Input('confusionmatrix-normalize', 'value')],
    )
    def update_precision_graph(cutoff, normalized):
        confmat_plot = self.plot_confusion_matrix(
                            cutoff=cutoff, normalized=normalized=='Normalized')
        roc_auc_plot = self.plot_roc_auc(cutoff=cutoff)
        pr_auc_plot = self.plot_pr_auc(cutoff=cutoff)
        return (confmat_plot, roc_auc_plot, pr_auc_plot)


    @app.callback(  
        Output('importances-graph', 'figure'),
        [Input('importance-tablesize', 'value'),
        Input('group-categoricals', 'value'),
        Input('permutation-or-shap', 'value')]
    )
    def update_importances(tablesize, cats, permutation_shap): 
        return self.plot_importances(
                    type=permutation_shap, topx=tablesize, cats=cats)