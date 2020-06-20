__all__ = [
    'ModelSummaryTab', 
    'ImportancesStats',
    'ClassifierModelStats',
    'RegressionModelStats',
]


import numpy as np

import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html

from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from .dashboard_methods import *

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class ModelSummaryTab:
    def __init__(self, explainer, standalone=False, hide_title=False,
                    tab_id="model_summary", title='Model Summary',
                    bin_size=0.1, quantiles=10, cutoff=0.5, 
                    round=2, logs=False, vs_actual=False, ratio=False,
                    n_features=15, **kwargs):
        self.explainer = explainer
        self.standalone = standalone
        
        self.tab_id = tab_id
        self.title = title

        if self.standalone:
            # If standalone then no 'pos-label-selector' or 'tabs'
            # component has been defined by overarching Dashboard.
            # The callbacks expect these to be there, so we add them in here.
            self.label_selector = TitleAndLabelSelector(
                                    explainer, title=title, 
                                    hidden=hide_title, dummy_tabs=True)
        else:
            # No need to define anything, so just add empty dummy
            self.label_selector = DummyComponent()

        if self.explainer.is_classifier:
            self.model_stats = ClassifierModelStats(explainer, 
                bin_size=bin_size, quantiles=quantiles, cutoff=cutoff) 
        elif explainer.is_regression:
            self.model_stats = RegressionModelStats(explainer,
                round=round, logs=logs, vs_actual=vs_actual, ratio=ratio)
        else:
            self.model_stats =  EmptyLayout()

        self.importances = ImportancesStats(explainer, n_features=n_features)

    def layout(self):
        return dbc.Container([
            self.label_selector.layout(),
            dcc.Store(id='testtest'),
            self.model_stats.layout(),
            self.importances.layout()
        ], fluid=True)

    
    def register_callbacks(self, app, **kwargs):
        self.label_selector.register_callbacks(app)
        self.model_stats.register_callbacks(app)
        self.importances.register_callbacks(app)


class ImportancesStats:
    def __init__(self, explainer, standalone=False, hide_title=False,
                        title="Importances",
                        n_features=None):
        self.explainer = explainer
        self.n_features = n_features
        self.standalone = standalone
        self.hide_title = hide_title
        if self.standalone:
            self.label_selector = TitleAndLabelSelector(
                                    explainer, title=title, 
                                    hidden=hide_title, dummy_tabs=True)
        else:
            self.label_selector = DummyComponent()
    
    def layout(self):
        cats_display = 'none' if self.explainer.cats is None else None
        return dbc.Container([
            self.label_selector.layout(),
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
                                                    for i in range(len(self.explainer.columns_cats))],
                                        value=min(self.n_features, len(self.explainer.columns_cats)) if self.n_features is not None else None)
                ]),
                dbc.Col([
                    html.Div([
                        dbc.Label("Grouping:"),
                        dbc.FormGroup(
                        [
                            dbc.RadioButton(
                                id='group-categoricals', 
                                className="form-check-input",
                                checked=True),
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
        self.label_selector.register_callbacks(app)

        @app.callback(  
            Output('importances-graph', 'figure'),
            [Input('importance-tablesize', 'value'),
             Input('group-categoricals', 'checked'),
             Input('permutation-or-shap', 'value'),
             Input('label-store', 'data')],
            [State('tabs', 'value')]
        )
        def update_importances(tablesize, cats, permutation_shap, pos_label, tab): 
            return self.explainer.plot_importances(
                        kind=permutation_shap, topx=tablesize, 
                        cats=cats, pos_label=pos_label)

class ClassifierModelStats:
    def __init__(self, explainer, title="Classification Stats", 
                    standalone=False, hide_title=False,
                    bin_size=0.1, quantiles=10, cutoff=0.5):
        self.explainer = explainer
        self.standalone = standalone
        self.hide_title = hide_title
        if self.standalone:
            self.label_selector = TitleAndLabelSelector(
                                    explainer, title=title, 
                                    hidden=hide_title, dummy_tabs=True)
        else:
            self.label_selector = DummyComponent()

        self.bin_size, self.quantiles, self.cutoff = bin_size, quantiles, cutoff

    def layout(self):
        return dbc.Container([
            self.label_selector.layout(),
            dbc.Row([dbc.Col([html.H2('Model Performance:')])]),

            dbc.Row([
                dbc.Col([
                    html.Div([
                        dcc.Loading(id="loading-precision-graph", 
                                children=[dcc.Graph(id='precision-graph')]),
                    ], style={'margin': 0}),
                    html.Div([
                        dbc.Label('Bin size:', html_for='precision-binsize'),
                        html.Div([
                            dcc.Slider(id='precision-binsize', 
                                    min = 0.01, max = 0.5, step=0.01, value=self.bin_size,
                                    marks={0.01: '0.01', 0.05: '0.05', 0.10: '0.10',
                                        0.20: '0.20', 0.25: '0.25' , 0.33: '0.33', 
                                        0.5: '0.5'}, 
                                    included=False,
                                    tooltip = {'always_visible' : False})
                        ], style={'margin-bottom': 25}),
                    ], id='bin-size-div', style={'margin': 5}),
                    html.Div([
                        dbc.Label('Quantiles:', html_for='precision-quantiles'),
                        html.Div([
                            dcc.Slider(id='precision-quantiles', 
                                        min = 1, max = 20, step=1, value=self.quantiles,
                                        marks={1: '1', 5: '5', 10: '10', 15: '15', 20:'20'}, 
                                        included=False,
                                        tooltip = {'always_visible' : False}),
                        ], style={'margin-bottom':25}),
                    ], id='quantiles-div', style={'margin': 5}),
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
                        inline=True),
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
                ], md=6, align="start"),
                dbc.Col([
                    dcc.Loading(id="loading-confusionmatrix-graph", 
                                children=[dcc.Graph(id='confusionmatrix-graph')]),
                    dbc.FormGroup([
                                dbc.RadioButton(
                                    id='confusionmatrix-percentage', 
                                    className="form-check-input", 
                                    checked=True
                                ),
                                dbc.Label(
                                    "Display percentages",
                                    html_for="confusionmatrix-percentage",
                                    className="form-check-label",
                                ),
                    ], check=True),
                    dbc.FormGroup([
                                dbc.RadioButton(
                                    id="confusionmatrix-binary", 
                                    className="form-check-input", 
                                    checked=True
                                ),
                                dbc.Label(
                                    "Binary (use cutoff for positive vs not positive)",
                                    html_for="confusionmatrix-binary",
                                    className="form-check-label",
                                ),
                    ], check=True),
                ], md=6, align="start"),              
            ]),
            dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.Label('Cutoff prediction probability:'),
                            dcc.Slider(id='precision-cutoff', 
                                        min = 0.01, max = 0.99, step=0.01, value=self.cutoff,
                                        marks={0.01: '0.01', 0.25: '0.25', 0.50: '0.50',
                                                0.75: '0.75', 0.99: '0.99'}, 
                                        included=False,
                                        tooltip = {'always_visible' : False})
                        ], style={'margin-bottom': 25}),
                    ])
                ]),
            dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.Label('Cutoff percentile of samples:'),
                            dcc.Slider(id='percentile-cutoff', 
                                        min = 0.01, max = 0.99, step=0.01, value=self.cutoff,
                                        marks={0.01: '0.01', 0.25: '0.25', 0.50: '0.50',
                                                0.75: '0.75', 0.99: '0.99'}, 
                                        included=False,
                                        tooltip = {'always_visible' : False})
                        ], style={'margin-bottom': 25}),
                    ])
                ]),
            dbc.Row([
                dbc.Col([
                    html.Div([
                        dcc.Loading(id="loading-lift-curve", 
                                children=[dcc.Graph(id='lift-curve-graph')]),
                    ], style={'margin': 0}),
                    dbc.FormGroup([
                        dbc.RadioButton(
                            id="lift-curve-percentage", 
                            className="form-check-input", 
                            checked=True
                        ),
                        dbc.Label(
                            "Display percentages",
                            html_for="lift-curve-percentage",
                            className="form-check-label",
                        ),
                    ], check=True),          
                ], md=6, align="start"),
                dbc.Col([
                    html.Div([
                                dcc.Loading(id="loading-classification-graph", 
                                            children=[dcc.Graph(id='classification-graph')]),
                    ], style={'margin': 0}),

                    dbc.FormGroup([
                                dbc.RadioButton(
                                    id="classification-percentage", 
                                    className="form-check-input", 
                                    checked=True
                                ),
                                dbc.Label(
                                    "Display percentages",
                                    html_for="classification-percentage",
                                    className="form-check-label",
                                ),
                    ], check=True),
                ], md=6, align="start"),
            ]),
            dbc.Row([    
                dbc.Col([
                    dcc.Loading(id="loading-roc-auc-graph", 
                                children=[dcc.Graph(id='roc-auc-graph')]),
                ], md=6),
                dbc.Col([
                    dcc.Loading(id="loading-pr-auc-graph", 
                                children=[dcc.Graph(id='pr-auc-graph')]),
                ], md=6),
            ]),
        ], fluid=True)

    def register_callbacks(self, app, **kwargs):
        self.label_selector.register_callbacks(app)
        @app.callback(
            [Output('bin-size-div', 'style'),
             Output('quantiles-div', 'style')],
            [Input('binsize-or-quantiles', 'value')],
        )
        def update_div_visibility(bins_or_quantiles):
            if bins_or_quantiles=='bin_size':
                return {}, {'display': 'none'}
            elif bins_or_quantiles=='quantiles':
                return {'display': 'none'}, {}
            raise PreventUpdate   
            
        @app.callback(
            Output('lift-curve-graph', 'figure'),
            [Input('lift-curve-percentage', 'checked'),
             Input('precision-cutoff', 'value'),
             Input('label-store', 'data')],
            [State('tabs', 'value')],
        )
        def update_precision_graph(percentage, cutoff, pos_label, tab):
            return self.explainer.plot_lift_curve(cutoff=cutoff, percentage=percentage, pos_label=pos_label)

        @app.callback(
            Output('precision-graph', 'figure'),
            [Input('precision-binsize', 'value'),
             Input('precision-quantiles', 'value'),
             Input('binsize-or-quantiles', 'value'),
             Input('precision-cutoff', 'value'),
             Input('precision-multiclass', 'checked'),
             Input('label-store', 'data')],
            [State('tabs', 'value')],
        )
        def update_precision_graph(bin_size, quantiles, bins, cutoff, multiclass, pos_label, tab):
            if bins=='bin_size':
                return self.explainer.plot_precision(
                    bin_size=bin_size, cutoff=cutoff, multiclass=multiclass, pos_label=pos_label)
            elif bins=='quantiles':
                return self.explainer.plot_precision(
                    quantiles=quantiles, cutoff=cutoff, multiclass=multiclass, pos_label=pos_label)
            raise PreventUpdate

        @app.callback(
            Output('classification-graph', 'figure'),
            [Input('classification-percentage', 'checked'),
             Input('precision-cutoff', 'value'),
             Input('label-store', 'data')],
            [State('tabs', 'value')],
        )
        def update_precision_graph(percentage, cutoff, pos_label, tab):
            return self.explainer.plot_classification(cutoff=cutoff, percentage=percentage, pos_label=pos_label)

        @app.callback(
             Output('confusionmatrix-graph', 'figure'),
            [Input('precision-cutoff', 'value'),
             Input('confusionmatrix-percentage', 'checked'),
             Input('confusionmatrix-binary', 'checked'),
             Input('label-store', 'data')],
            [State('tabs', 'value')],
        )
        def update_precision_graph(cutoff, normalized, binary, pos_label, tab):
            return self.explainer.plot_confusion_matrix(
                        cutoff=cutoff, normalized=normalized, binary=binary, pos_label=pos_label)

        @app.callback(
            Output('roc-auc-graph', 'figure'),
            [Input('precision-cutoff', 'value'),
             Input('label-store', 'data'),
             Input('tabs', 'value')],
        )
        def update_precision_graph(cutoff, pos_label, tab):
            return self.explainer.plot_roc_auc(cutoff=cutoff, pos_label=pos_label)

        @app.callback(
            Output('pr-auc-graph', 'figure'),
            [Input('precision-cutoff', 'value'),
             Input('label-store', 'data')],
            [State('tabs', 'value')],
        )
        def update_precision_graph(cutoff, pos_label, tab):
            return self.explainer.plot_pr_auc(cutoff=cutoff, pos_label=pos_label)

        @app.callback(
            Output('precision-cutoff', 'value'),
            [Input('percentile-cutoff', 'value'),
             Input('label-store', 'data')]
        )
        def update_cutoff(percentile, pos_label):
            return np.round(self.explainer.cutoff_from_percentile(percentile, pos_label=pos_label), 2)

class RegressionModelStats:
    def __init__(self, explainer, title="Regression Stats", 
                    standalone=False, hide_title=False,
                    round=2, logs=False, vs_actual=False, ratio=False):
        self.explainer = explainer          
        self.round, self.logs, self.vs_actual, self. ratio  = round, logs, vs_actual, ratio
        self.standalone = standalone

        if self.standalone:
            self.label_selector = TitleAndLabelSelector(
                                    explainer, title=title, 
                                    hidden=hide_title, dummy_tabs=True)
        else:
            self.label_selector = DummyComponent()

    def layout(self):
        return dbc.Container([
            self.label_selector.layout(),
            dbc.Row([dbc.Col([html.H2('Model Performance:')])]),
            dbc.Row([
                dbc.Col([
                    
                    dcc.Loading(id="loading-predicted-vs-actual-graph", 
                                children=[dcc.Graph(id='predicted-vs-actual-graph')]),
                    dbc.FormGroup(
                    [
                        dbc.RadioButton(
                            id='preds-vs-actual-logs',
                            className="form-check-input"),
                        dbc.Label("Take Logs",
                                html_for='preds-vs-actual-logs',
                                className="form-check-label"),
                    ], check=True),
                ], md=6),
                dbc.Col([
                    dcc.Loading(id="loading-model-summary", 
                                children=[dcc.Markdown(id='model-summary')]),      
                ], md=6),
            ], align="start"),
            dbc.Row([
                dbc.Col([

                    dcc.Loading(id="loading-residuals-graph", 
                                children=[dcc.Graph(id='residuals-graph')]),
                    dbc.FormGroup(
                    [
                        dbc.RadioItems(
                            options=[
                                {"label": "vs Prediction", "value": "vs_pred"},
                                {"label": "vs Actual", "value": "vs_actual"},
                            ],
                            value="vs_pred",
                            id='residuals-pred-or-actual',
                            inline=True,
                        ),
                    ]),
                    dbc.FormGroup(
                    [
                        dbc.RadioButton(
                            id='residuals-ratio',
                            className="form-check-input"),
                        dbc.Label("Display Ratio",
                                html_for='residuals-ratio',
                                className="form-check-label"),
                    ], check=True),

                ], md=6),
                dbc.Col([
                    dcc.Loading(id="loading-residuals-vs-col-graph", 
                                children=[dcc.Graph(id='residuals-vs-col-graph')]),
                    dbc.Label("Column:"),
                    dcc.Dropdown(id='residuals-col',
                        options = [{'label': col, 'value': col} 
                                        for col in self.explainer.mean_abs_shap_df(cats=False)\
                                                        .Feature.tolist()],
                        value=self.explainer.mean_abs_shap_df(cats=False)\
                                                        .Feature.tolist()[0]),
                    dbc.FormGroup(
                    [
                        dbc.RadioButton(
                            id='residuals-vs-col-ratio',
                            className="form-check-input"),
                        dbc.Label("Display Ratio",
                                html_for='residuals-vs-col-ratio',
                                className="form-check-label"),
                    ], check=True),
                ], md=6),
            ])
        ], fluid=True)
        

    def register_callbacks(self, app, **kwargs):
        self.label_selector.register_callbacks(app)

        @app.callback(
            Output('model-summary', 'children'),
            [Input('label-store', 'data')],
            [State('tabs', 'value')]
        )
        def update_model_summary(pos_label, tab):
            return self.explainer.metrics_markdown()

        Output('model-prediction', 'children')
        @app.callback(
            Output('predicted-vs-actual-graph', 'figure'),
            [Input('preds-vs-actual-logs', 'checked')],
            [State('tabs', 'value')]
        )
        def update_predicted_vs_actual_graph(logs, tab):
            return self.explainer.plot_predicted_vs_actual(logs=logs)

        @app.callback(
            Output('residuals-graph', 'figure'),
            [Input('residuals-pred-or-actual', 'value'),
             Input('residuals-ratio', 'checked')],
            [State('tabs', 'value')],
        )
        def update_residuals_graph(pred_or_actual, ratio, tab):
            vs_actual = pred_or_actual=='vs_actual'
            return self.explainer.plot_residuals(vs_actual=vs_actual, ratio=ratio)

        @app.callback(
            Output('residuals-vs-col-graph', 'figure'),
            [Input('residuals-col', 'value'),
             Input('residuals-vs-col-ratio', 'checked')],
            [State('tabs', 'value')],
        )
        def update_residuals_graph(col, ratio, gtab):
            return self.explainer.plot_residuals_vs_feature(col, ratio=ratio, dropna=True)



