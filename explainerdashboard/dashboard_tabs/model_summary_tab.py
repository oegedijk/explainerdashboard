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
                 bin_size=0.1, quantiles=10, cutoff=0.5, n_features=15, **kwargs):
        self.explainer = explainer
        self.standalone = standalone
        
        self.bin_size = bin_size
        self.quantiles = quantiles
        self.cutoff = cutoff
        self.n_features = n_features
        self.kwargs = kwargs
        
        self.tab_id = tab_id
        self.title = title
        
    def layout(self):
        if self.standalone:
            return model_summary_layout(self.explainer, title=self.title, standalone=self.standalone, 
                                     bin_size=self.bin_size, quantiles=self.quantiles, 
                                     cutoff=self.cutoff, n_features=self.n_features)
        else:
            return model_summary_layout(self.explainer,  
                                     bin_size=self.bin_size, quantiles=self.quantiles, 
                                     cutoff=self.cutoff, n_features=self.n_features)
    
    def register_callbacks(self, app):
        return model_summary_callbacks(self.explainer, app, standalone=self.standalone)


def model_summary_layout(explainer, 
            title=None, standalone=False, hide_selector=False,
            bin_size=0.1, quantiles=10, cutoff=0.5, n_features=15, **kwargs):
    """returns layout for model summary tab
    
    :param explainer: ExplainerBunch object for which to build layout for
    :type explainer: ExplainerBUnch
    :type title: str
    :param standalone: when standalone layout, include a a label_store, defaults to False
    :type standalone: bool
    :param hide_selector: if model is a classifier, optionally hide the positive label selector, defaults to False
    :type hide_selector: bool
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
                            dbc.Label('Bin size:', html_for='precision-binsize'),
                            dcc.Slider(id='precision-binsize', 
                                        min = 0.01, max = 0.5, step=0.01, value=bin_size,
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
                                    min = 1, max = 20, step=1, value=quantiles,
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
                    ]),
                ]),
            ], id='precision-plot-div', style={'display': 'none'}),
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
            ], id='lift-curve-div', style={'display': 'none'}),
            dbc.Row([
                    dbc.Col([
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
        ] 
    elif explainer.is_regression:
        model_stats_rows = [
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
        ]

    return dbc.Container([
        title_and_label_selector(explainer, title, standalone, hide_selector),
        dbc.Row([dbc.Col([html.H2('Model overview:')])]),
    
        *model_stats_rows,
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
                                                for i in range(len(explainer.columns))],
                                    value=min(n_features, len(explainer.columns)))
            ]),
            dbc.Col([
                dbc.Label("Grouping:"),
                dbc.FormGroup(
                [
                    dbc.RadioButton(
                        id='group-categoricals', 
                        className="form-check-input"),
                    dbc.Label("Group Cats",
                            html_for='group-categoricals',
                            className="form-check-label"),
                ], check=True)
            ])],
            form=True, justify="between"),

        dcc.Loading(id="loading-importances-graph", 
                        children=[dcc.Graph(id='importances-graph')]), 
        ], fluid=True)


def model_summary_callbacks(explainer, app, standalone=False, **kwargs):
    if explainer.is_classifier:
        if standalone:
            label_selector_register_callback(explainer, app)

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
                return explainer.plot_precision(
                    bin_size=bin_size, cutoff=cutoff, multiclass=multiclass)
            elif bins=='quantiles':
                return explainer.plot_precision(
                    quantiles=quantiles, cutoff=cutoff, multiclass=multiclass)
            raise PreventUpdate


        @app.callback(
             Output('confusionmatrix-graph', 'figure'),
            [Input('precision-cutoff', 'value'),
             Input('confusionmatrix-normalize', 'value'),
             Input('label-store', 'data')],
        )
        def update_precision_graph(cutoff, normalized, pos_label):
            return explainer.plot_confusion_matrix(
                                cutoff=cutoff, normalized=normalized=='Normalized')

        @app.callback(
            Output('lift-curve-graph', 'figure'),
            [Input('lift-curve-percentage', 'checked'),
             Input('precision-cutoff', 'value'),
             Input('label-store', 'data')],
        )
        def update_precision_graph(percentage, cutoff, pos_label):
            return explainer.plot_lift_curve(cutoff=cutoff, percentage=percentage)
        
        @app.callback(
            Output('roc-auc-graph', 'figure'),
            [Input('precision-cutoff', 'value'),
             Input('label-store', 'data')],
        )
        def update_precision_graph(cutoff, pos_label):
            return explainer.plot_roc_auc(cutoff=cutoff)

        @app.callback(
            Output('pr-auc-graph', 'figure'),
            [Input('precision-cutoff', 'value'),
             Input('label-store', 'data')],
        )
        def update_precision_graph(cutoff, pos_label):
            return explainer.plot_pr_auc(cutoff=cutoff)
    
    elif explainer.is_regression:
        
        @app.callback(
            Output('predicted-vs-actual-graph', 'figure'),
            [Input('label-store', 'data')],
        )
        def update_predicted_vs_actual_graph(pos_label):
            return explainer.plot_predicted_vs_actual()

        @app.callback(
            Output('residuals-graph', 'figure'),
            [Input('label-store', 'data')],
        )
        def update_residuals_graph( pos_label):
            return explainer.plot_residuals()



    @app.callback(  
        Output('importances-graph', 'figure'),
        [Input('importance-tablesize', 'value'),
         Input('group-categoricals', 'checked'),
         Input('permutation-or-shap', 'value'),
         Input('label-store', 'data')]
    )
    def update_importances(tablesize, cats, permutation_shap, pos_label): 
        return explainer.plot_importances(
                    type=permutation_shap, topx=tablesize, cats=cats)