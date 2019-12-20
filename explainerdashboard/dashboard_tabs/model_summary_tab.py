__all__ = ['ModelSummaryTab']

import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html

from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from .dashboard_methods import *

class ModelSummaryTab:
    def __init__(self, explainer, standalone=False, tab_id="model_summary", title='Model Summary',
                 bin_size=0.1, quantiles=10, cutoff=0.5, 
                 round=2, logs=False, vs_actual=False, ratio=False,
                 n_features=15, **kwargs):
        self.explainer = explainer
        self.standalone = standalone
        
        self.tab_id = tab_id
        self.title = title

        if self.standalone:
            self.label_selector = TitleAndLabelSelector(explainer, title=title)

        if self.explainer.is_classifier:
            self.model_stats = ClassifierModelStats(explainer, bin_size, quantiles, cutoff) 
        elif explainer.is_regression:
            self.model_stats = RegressionModelStats(explainer, round, logs, vs_actual, ratio)
        else:
            self.model_stats =  EmptyLayout()

        self.importances = ImportancesStats(explainer, n_features)

    def layout(self):
        return dbc.Container([
            self.label_selector.layout() if self.standalone else None,
            self.model_stats.layout(),
            self.importances.layout()
        ], fluid=True)

    
    def register_callbacks(self, app, **kwargs):
        if self.standalone:
            self.label_selector.register_callbacks(app)
        self.model_stats.register_callbacks(app)
        self.importances.register_callbacks(app)


class ImportancesStats:
    def __init__(self, explainer, n_features=15):
        self.explainer = explainer
        self.n_features = n_features
    
    def layout(self):
        cats_display = 'none' if self.explainer.cats is None else None
        return dbc.Container([
            dbc.Row([dbc.Col([html.H2('Feature Importances:')])]),
            dbc.Row([
                dbc.Col([
                    dbc.FormGroup(
                            [
                                dbc.Label("Importances type:"),
                                dbc.RadioItems(
                                    options=[
                                        {'label': 'Permutation Importances', 
                                        'value': 'permutation'},
                                        {'label': 'SHAP values', 
                                        'value': 'shap'}
                                    ],
                                    value='shap',
                                    id='permutation-or-shap',
                                    inline=True,
                                ),
                            ]
                        )
                ]),
                dbc.Col([
                    html.Label('Select max number of importances to display:'),
                    dcc.Dropdown(id='importance-tablesize',
                                        options = [{'label': str(i+1), 'value':i+1} 
                                                    for i in range(len(self.explainer.columns))],
                                        value=min(self.n_features, len(self.explainer.columns)))
                ]),
                dbc.Col([
                    html.Div([
                        dbc.Label("Grouping:"),
                        dbc.FormGroup(
                        [
                            dbc.RadioButton(
                                id='group-categoricals', 
                                className="form-check-input"),
                            dbc.Label("Group Cats",
                                    html_for='group-categoricals',
                                    className="form-check-label"),
                        ], check=True),
                    ], style=cats_display)
                ]),
                    
            ], form=True, justify="between"),

            dbc.Row([
                dbc.Col([
                    dcc.Loading(id="loading-importances-graph", 
                            children=[dcc.Graph(id='importances-graph')])
                ]),
            ]), 
            ], fluid=True)
        
    def register_callbacks(self, app, **kwargs):
        @app.callback(  
            Output('importances-graph', 'figure'),
            [Input('importance-tablesize', 'value'),
             Input('group-categoricals', 'checked'),
             Input('permutation-or-shap', 'value'),
             Input('label-store', 'data')]
        )
        def update_importances(tablesize, cats, permutation_shap, pos_label): 
            return self.explainer.plot_importances(
                        type=permutation_shap, topx=tablesize, cats=cats)

class ClassifierModelStats:
    def __init__(self, explainer, bin_size=0.1, quantiles=10, cutoff=0.5):
        self.explainer = explainer
        self.bin_size, self.quantiles, self.cutoff = bin_size, quantiles, cutoff

    def layout(self):
        return dbc.Container([
            dbc.Row([dbc.Col([html.H2('Model Performance:')])]),
            dbc.Row([
                dbc.ButtonGroup(
                    [
                        dbc.Button("precision plot", id="precision-plot-button", ), 
                        dbc.Button("lift curve", id="lift-curve-button")
                    ],
                    size="lg",
                    className="mr-1",
                ),
            ], justify="center"),
            html.Div([
                dbc.Row([
                    dbc.Col([
                        dbc.FormGroup([
                                    dbc.RadioButton(
                                        id="lift-curve-percentage", className="form-check-input"
                                    ),
                                    dbc.Label(
                                        "Display percentages",
                                        html_for="lift-curve-percentage",
                                        className="form-check-label",
                                    ),
                                ], check=True),
                    ], width=2),
                ], justify="end",),
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            dcc.Loading(id="loading-lift-curve", 
                                        children=[dcc.Graph(id='lift-curve-graph')]),
                        ], style={'margin': 20}),
                    ]),
                ]),
            ], id='lift-curve-div',  style={}), # style={'display': 'none'}),
            html.Div([
                dbc.Row([
                    dbc.Col([
                            dbc.Label('Bin size:', html_for='precision-binsize'),
                            dcc.Slider(id='precision-binsize', 
                                        min = 0.01, max = 0.5, step=0.01, value=self.bin_size,
                                        marks={0.01: '0.01', 0.05: '0.05', 0.10: '0.10',
                                            0.20: '0.20', 0.25: '0.25' , 0.33: '0.33', 
                                            0.5: '0.5'}, 
                                        included=False,
                                        tooltip = {'always_visible' : True})
                        ], width=4),
                    dbc.Col([
                            dbc.Label('Binning Method:', html_for='binsize-or-quantiles'),
                            dbc.RadioItems(
                                id='binsize-or-quantiles',
                                options=[
                                    {'label': 'Bin Size', 
                                    'value': 'bin_size'},
                                    {'label': 'Quantiles', 
                                    'value': 'quantiles'}
                                ],
                                value='bin_size',
                                inline=True)
                        ], width=2),
                    dbc.Col([
                        dbc.FormGroup([
                                    dbc.RadioButton(
                                        id="precision-multiclass", className="form-check-input"
                                    ),
                                    dbc.Label(
                                        "Display all classes",
                                        html_for="precision-multiclass",
                                        className="form-check-label",
                                    ),
                                ], check=True),
                    ], width=2),
                    dbc.Col([
                        dbc.Label('Quantiles:', html_for='precision-quantiles'),
                        dcc.Slider(id='precision-quantiles', 
                                    min = 1, max = 20, step=1, value=self.quantiles,
                                    marks={1: '1', 5: '5', 10: '10', 15: '15', 20:'20'}, 
                                    included=False,
                                    tooltip = {'always_visible' : True})
                    ], width=4),
                ], form=True, align="center"),
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            dcc.Loading(id="loading-precision-graph", 
                                    children=[dcc.Graph(id='precision-graph')]),
                        ], style={'margin': 20}),
                    ], width=12),
                ], align="center"),                
            ], id='precision-plot-div', style={}), #{'display': 'none'}),
            
            dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.Label('Cutoff:'),
                            dcc.Slider(id='precision-cutoff', 
                                        min = 0.01, max = 0.99, step=0.01, value=self.cutoff,
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
                    dbc.FormGroup(
                        [
                            dbc.Label("Confusion Matrix Display:"),
                            dbc.RadioItems(
                                options=[{'label': o, 'value': o} 
                                    for o in ['Counts', 'Normalized']],
                                value='Counts',
                                id='confusionmatrix-normalize',
                                inline=True,
                            ),  
                        ]
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
        ], fluid=True)

    def register_callbacks(self, app, **kwargs):
        @app.callback(
            [Output('precision-plot-div', 'style'),
             Output('lift-curve-div', 'style')],
            [Input('precision-plot-button', 'n_clicks'),
             Input('lift-curve-button', 'n_clicks')]
        )
        def update_output_div(precision, lift):
            ctx = dash.callback_context
            if ctx.triggered:
                trigger = ctx.triggered[0]['prop_id'].split('.')[0]
                if trigger=='precision-plot-button':
                    return {}, {'display': 'none'}
                elif trigger=='lift-curve-button':
                    return {'display': 'none'}, {}, 
            raise PreventUpdate

        @app.callback(
            Output('precision-graph', 'figure'),
            [Input('precision-binsize', 'value'),
             Input('precision-quantiles', 'value'),
             Input('binsize-or-quantiles', 'value'),
             Input('precision-cutoff', 'value'),
             Input('precision-multiclass', 'checked'),
             Input('label-store', 'data')],
        )
        def update_precision_graph(bin_size, quantiles, bins, cutoff, multiclass, pos_label):
            if bins=='bin_size':
                return self.explainer.plot_precision(
                    bin_size=bin_size, cutoff=cutoff, multiclass=multiclass)
            elif bins=='quantiles':
                return self.explainer.plot_precision(
                    quantiles=quantiles, cutoff=cutoff, multiclass=multiclass)
            raise PreventUpdate


        @app.callback(
             Output('confusionmatrix-graph', 'figure'),
            [Input('precision-cutoff', 'value'),
             Input('confusionmatrix-normalize', 'value'),
             Input('label-store', 'data')],
        )
        def update_precision_graph(cutoff, normalized, pos_label):
            return self.explainer.plot_confusion_matrix(
                                cutoff=cutoff, normalized=normalized=='Normalized')

        @app.callback(
            Output('lift-curve-graph', 'figure'),
            [Input('lift-curve-percentage', 'checked'),
             Input('precision-cutoff', 'value'),
             Input('label-store', 'data')],
        )
        def update_precision_graph(percentage, cutoff, pos_label):
            return self.explainer.plot_lift_curve(cutoff=cutoff, percentage=percentage)
        
        @app.callback(
            Output('roc-auc-graph', 'figure'),
            [Input('precision-cutoff', 'value'),
             Input('label-store', 'data')],
        )
        def update_precision_graph(cutoff, pos_label):
            return self.explainer.plot_roc_auc(cutoff=cutoff)

        @app.callback(
            Output('pr-auc-graph', 'figure'),
            [Input('precision-cutoff', 'value'),
             Input('label-store', 'data')],
        )
        def update_precision_graph(cutoff, pos_label):
            return self.explainer.plot_pr_auc(cutoff=cutoff)

class RegressionModelStats:
    def __init__(self, explainer, round=2, logs=False, vs_actual=False, ratio=False):
        self.explainer = explainer
        self.round, self.logs, self.vs_actual, self. ratio  = round, logs, vs_actual, ratio

    def layout(self):
        return dbc.Container([
            dbc.Row([dbc.Col([html.H2('Model Performance:')])]),
            dbc.Row([
                dbc.Col([
                    dcc.Loading(id="loading-predicted-vs-actual-graph", 
                                children=[dcc.Graph(id='predicted-vs-actual-graph')]),
                ], width=6),
                dbc.Col([
                    dcc.Loading(id="loading-residuals-graph", 
                                children=[dcc.Graph(id='residuals-graph')]),
                ], width=6),
            ])
        ])
        

    def register_callbacks(self, app, **kwargs):
        @app.callback(
            Output('predicted-vs-actual-graph', 'figure'),
            [Input('label-store', 'data')],
        )
        def update_predicted_vs_actual_graph(pos_label):
            return self.explainer.plot_predicted_vs_actual()

        @app.callback(
            Output('residuals-graph', 'figure'),
            [Input('label-store', 'data')],
        )
        def update_residuals_graph( pos_label):
            return self.explainer.plot_residuals()



