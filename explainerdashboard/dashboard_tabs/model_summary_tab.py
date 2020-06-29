__all__ = [
    'ModelSummaryTab', 
    'ImportancesComponent',
    'ClassifierModelStatsComponent',
    'RegressionModelStatsComponent',
    'PrecisionComponent',
    'ConfusionMatrixComponent',
    'LiftCurveComponent',
    'ClassificationComponent',
    'RocAucComponent',
    'PrAucComponent',
    'PredictedVsActualComponent',
    'ResidualsComponent',
    'ResidualsVsColComponent'
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


class ModelSummaryTab(ExplainerComponent):
    def __init__(self, explainer, title="Model Summary",
                    header_mode="none", name=None,
                    tab_id="model_summary",
                    bin_size=0.1, quantiles=10, cutoff=0.5, 
                    logs=False, pred_or_actual="vs_pred", ratio=False, col=None,
                    importance_type="shap", depth=None, cats=True):
        super().__init__(explainer, title, header_mode, name)
        
        self.tab_id = tab_id

        if self.explainer.is_classifier:
            self.model_stats = ClassifierModelStatsComponent(explainer, 
                bin_size=bin_size, quantiles=quantiles, cutoff=cutoff) 
        elif explainer.is_regression:
            self.model_stats = RegressionModelStatsComponent(explainer,
                logs=logs, pred_or_actual=pred_or_actual, ratio=ratio)

        self.importances = ImportancesComponent(explainer, 
                importance_type=importance_type, depth=depth, cats=cats)

        self.register_components(self.model_stats, self.importances)

    def _layout(self):
        return dbc.Container([
            self.model_stats.layout(),
            self.importances.layout()
        ], fluid=True)


class PrecisionComponent(ExplainerComponent):
    def __init__(self, explainer, title="Precision Plot",
                    header_mode="none", name=None,
                    hide_cutoff=False, hide_binsize=False, hide_binmethod=False,
                    hide_multiclass=False,
                    bin_size=0.1, quantiles=10, cutoff=0.5,
                    precision_or_binsize='bin_size', multiclass=False):
        super().__init__(explainer, title, header_mode, name)

        self.hide_cutoff, self.hide_binsize = hide_cutoff, hide_binsize
        self.hide_binmethod, self.hide_multiclass = hide_binmethod, hide_multiclass
        
        self.bin_size, self.quantiles, self.cutoff = bin_size, quantiles, cutoff 
        self.precision_or_binsize, self.multiclass = precision_or_binsize, multiclass
        self.cutoff_name = 'precision-cutoff-' + self.name
        self.register_dependencies("preds")

    def _layout(self):
        return html.Div([
            self.header.layout(),
            html.Div([
                dcc.Loading(id="loading-precision-graph-"+self.name, 
                        children=[dcc.Graph(id='precision-graph-'+self.name)]),
            ], style={'margin': 0}),
            make_hideable(
                html.Div([
                    dbc.Label('Bin size:', html_for='precision-binsize-'+self.name),
                    html.Div([
                        dcc.Slider(id='precision-binsize-'+self.name, 
                                min = 0.01, max = 0.5, step=0.01, value=self.bin_size,
                                marks={0.01: '0.01', 0.05: '0.05', 0.10: '0.10',
                                    0.20: '0.20', 0.25: '0.25' , 0.33: '0.33', 
                                    0.5: '0.5'}, 
                                included=False,
                                tooltip = {'always_visible' : False})
                    ], style={'margin-bottom': 25}),
                ], id='precision-bin-size-div-'+self.name, style=dict(margin=5)),  
                hide=self.hide_binsize),
            make_hideable(
                html.Div([
                    dbc.Label('Quantiles:', html_for='precision-quantiles-'+self.name),
                    html.Div([
                        dcc.Slider(id='precision-quantiles-'+self.name, 
                                    min = 1, max = 20, step=1, value=self.quantiles,
                                    marks={1: '1', 5: '5', 10: '10', 15: '15', 20:'20'}, 
                                    included=False,
                                    tooltip = {'always_visible' : False}),
                    ], style={'margin-bottom':25}),
                ], id='precision-quantiles-div-'+self.name), hide=self.hide_binsize),
            make_hideable(
                html.Div([
                    html.Label('Cutoff prediction probability:'),
                    dcc.Slider(id='precision-cutoff-'+self.name, 
                                min = 0.01, max = 0.99, step=0.01, value=self.cutoff,
                                marks={0.01: '0.01', 0.25: '0.25', 0.50: '0.50',
                                        0.75: '0.75', 0.99: '0.99'}, 
                                included=False,
                                tooltip = {'always_visible' : False})
                ], style={'margin-bottom': 25}), hide=self.hide_cutoff),
            make_hideable(
                html.Div([
                    dbc.Label('Binning Method:', html_for='precision-binsize-or-quantiles-'+self.name),
                    dbc.RadioItems(
                        id='precision-binsize-or-quantiles-'+self.name,
                        options=[
                            {'label': 'Bin Size', 
                            'value': 'bin_size'},
                            {'label': 'Quantiles', 
                            'value': 'quantiles'}
                        ],
                        value=self.precision_or_binsize,
                        inline=True),
                ]), hide=self.hide_binmethod),
            make_hideable(
                html.Div([
                    dbc.FormGroup([
                            dbc.RadioButton(
                                id="precision-multiclass-"+self.name, 
                                className="form-check-input",
                                checked=self.multiclass
                            ),
                            dbc.Label(
                                "Display all classes",
                                html_for="precision-multiclass-"+self.name,
                                className="form-check-label",
                            ),
                    ], check=True),
                ]), hide=self.hide_multiclass),     
        ])

    def _register_callbacks(self, app):
        @app.callback(
            [Output('precision-bin-size-div-'+self.name, 'style'),
             Output('precision-quantiles-div-'+self.name, 'style')],
            [Input('precision-binsize-or-quantiles-'+self.name, 'value')],
        )
        def update_div_visibility(bins_or_quantiles):
            if self.hide_binsize:
                return dict(display='none'), dict(display='none')
            if bins_or_quantiles=='bin_size':
                return {}, dict(display='none')
            elif bins_or_quantiles=='quantiles':
                return dict(display='none'), {}
            raise PreventUpdate   

        @app.callback(
            Output('precision-graph-'+self.name, 'figure'),
            [Input('precision-binsize-'+self.name, 'value'),
             Input('precision-quantiles-'+self.name, 'value'),
             Input('precision-binsize-or-quantiles-'+self.name, 'value'),
             Input('precision-cutoff-'+self.name, 'value'),
             Input('precision-multiclass-'+self.name, 'checked'),
             Input('pos-label', 'value')],
            [State('tabs', 'value')],
        )
        def update_precision_graph(bin_size, quantiles, bins, cutoff, multiclass, pos_label, tab):
            if bins == 'bin_size':
                return self.explainer.plot_precision(
                    bin_size=bin_size, cutoff=cutoff, multiclass=multiclass, pos_label=pos_label)
            elif bins == 'quantiles':
                return self.explainer.plot_precision(
                    quantiles=quantiles, cutoff=cutoff, multiclass=multiclass, pos_label=pos_label)
            raise PreventUpdate

class ConfusionMatrixComponent(ExplainerComponent):
    def __init__(self, explainer, title="Confusion Matrix",
                    header_mode="none", name=None,
                    hide_cutoff=False, hide_percentage=False, hide_binary=False,
                    cutoff=0.5, percentage=True, binary=True):
        super().__init__(explainer, title, header_mode, name)

        self.hide_cutoff, self.hide_percentage = hide_cutoff, hide_percentage
        self.hide_binary = hide_binary
        self.cutoff, self.percentage, self.binary = cutoff, percentage, binary
        self.cutoff_name = 'confusionmatrix-cutoff-' + self.name
        self.register_dependencies("preds")

    def _layout(self):
        return html.Div([
            dcc.Loading(id='loading-confusionmatrix-graph-'+self.name, 
                            children=[dcc.Graph(id='confusionmatrix-graph-'+self.name)]),
            make_hideable(
                html.Div([
                    html.Label('Cutoff prediction probability:'),
                    dcc.Slider(id='confusionmatrix-cutoff-'+self.name, 
                                min = 0.01, max = 0.99, step=0.01, value=self.cutoff,
                                marks={0.01: '0.01', 0.25: '0.25', 0.50: '0.50',
                                        0.75: '0.75', 0.99: '0.99'}, 
                                included=False,
                                tooltip = {'always_visible' : False})
                ], style={'margin-bottom': 25}), hide=self.hide_cutoff),
            make_hideable(
                html.Div([
                    dbc.FormGroup([
                            dbc.RadioButton(
                                id='confusionmatrix-percentage-'+self.name, 
                                className="form-check-input", 
                                checked=self.percentage
                            ),
                            dbc.Label(
                                "Display percentages",
                                html_for="confusionmatrix-percentage-"+self.name,
                                className="form-check-label",
                            ),
                    ], check=True),
                ]), hide=self.hide_percentage),
            make_hideable(
                html.Div([
                    dbc.FormGroup([
                            dbc.RadioButton(
                                id="confusionmatrix-binary-"+self.name, 
                                className="form-check-input", 
                                checked=self.binary
                            ),
                            dbc.Label(
                                "Binary (use cutoff for positive vs not positive)",
                                html_for="confusionmatrix-binary-"+self.name,
                                className="form-check-label",
                            ),
                    ], check=True),
                ]), hide=self.hide_binary),
        ])

    def register_callbacks(self, app):
        @app.callback(
             Output('confusionmatrix-graph-'+self.name, 'figure'),
            [Input('confusionmatrix-cutoff-'+self.name, 'value'),
             Input('confusionmatrix-percentage-'+self.name, 'checked'),
             Input('confusionmatrix-binary-'+self.name, 'checked'),
             Input('pos-label', 'value')],
            [State('tabs', 'value')],
        )
        def update_confusionmatrix_graph(cutoff, normalized, binary, pos_label, tab):
            return self.explainer.plot_confusion_matrix(
                        cutoff=cutoff, normalized=normalized, binary=binary, pos_label=pos_label)

class LiftCurveComponent(ExplainerComponent):
    def __init__(self, explainer, title="Lift Curve",
                    header_mode="none", name=None,
                    hide_cutoff=False, hide_percentage=False,
                    cutoff=0.5, percentage=True):
        super().__init__(explainer, title, header_mode, name)

        self.hide_cutoff, self.hide_percentage = hide_cutoff, hide_percentage
        self.cutoff, self.percentage = cutoff, percentage
        self.cutoff_name = 'liftcurve-cutoff-' + self.name
        self.register_dependencies("preds")

    def _layout(self):
        return html.Div([
            html.Div([
                        dcc.Loading(id='loading-lift-curve-'+self.name, 
                                children=[dcc.Graph(id='liftcurve-graph-'+self.name)]),
                    ], style={'margin': 0}),
            make_hideable(
                html.Div([
                    html.Label('Cutoff prediction probability:'),
                    dcc.Slider(id='liftcurve-cutoff-'+self.name, 
                                min = 0.01, max = 0.99, step=0.01, value=self.cutoff,
                                marks={0.01: '0.01', 0.25: '0.25', 0.50: '0.50',
                                        0.75: '0.75', 0.99: '0.99'}, 
                                included=False,
                                tooltip = {'always_visible' : False})
                ], style={'margin-bottom': 25}), hide=self.hide_cutoff),
            make_hideable(
                html.Div([
                    dbc.FormGroup([
                        dbc.RadioButton(
                            id="liftcurve-percentage-"+self.name, 
                            className="form-check-input", 
                            checked=True
                        ),
                        dbc.Label(
                            "Display percentages",
                            html_for="liftcurve-percentage-"+self.name,
                            className="form-check-label",
                        ),
                    ], check=True), 
                ]), hide=self.hide_percentage),         
        ])

    def _register_callbacks(self, app):
        @app.callback(
            Output('liftcurve-graph-'+self.name, 'figure'),
            [Input('liftcurve-cutoff-'+self.name, 'value'),
             Input('liftcurve-percentage-'+self.name, 'checked'),
             Input('pos-label', 'value')],
            [State('tabs', 'value')],
        )
        def update_precision_graph(cutoff, percentage, pos_label, tab):
            return self.explainer.plot_lift_curve(cutoff=cutoff, percentage=percentage, pos_label=pos_label)

class ClassificationComponent(ExplainerComponent):
    def __init__(self, explainer, title="Classification Plot",
                    header_mode="none", name=None,
                    hide_cutoff=False, hide_percentage=False,
                    cutoff=0.5, percentage=True):
        super().__init__(explainer, title, header_mode, name)

        self.hide_cutoff, self.hide_percentage = hide_cutoff, hide_percentage
        self.cutoff, percentage = cutoff, percentage
        self.cutoff_name = 'classification-cutoff-' + self.name
        self.register_dependencies("preds")

    def _layout(self):
        return html.Div([
            html.Div([
                dcc.Loading(id="loading-classification-graph-"+self.name, 
                            children=[dcc.Graph(id='classification-graph-'+self.name)]),
            ], style={'margin': 0}),
            make_hideable(
                html.Div([
                    html.Label('Cutoff prediction probability:'),
                    dcc.Slider(id='classification-cutoff-'+self.name, 
                                min = 0.01, max = 0.99, step=0.01, value=self.cutoff,
                                marks={0.01: '0.01', 0.25: '0.25', 0.50: '0.50',
                                        0.75: '0.75', 0.99: '0.99'}, 
                                included=False,
                                tooltip = {'always_visible' : False})
                ], style={'margin-bottom': 25}), hide=self.hide_cutoff),
            make_hideable(
                html.Div([
                    dbc.FormGroup([
                        dbc.RadioButton(
                            id="classification-percentage-"+self.name, 
                            className="form-check-input", 
                            checked=True
                        ),
                        dbc.Label(
                            "Display percentages",
                            html_for="classification-percentage-"+self.name,
                            className="form-check-label",
                        ),
                    ], check=True),
                ]), hide=self.hide_percentage),
        ])

    def register_callbacks(self, app):
        @app.callback(
            Output('classification-graph-'+self.name, 'figure'),
            [Input('classification-cutoff-'+self.name, 'value'),
             Input('classification-percentage-'+self.name, 'checked'),
             Input('pos-label', 'value')],
            [State('tabs', 'value')],
        )
        def update_precision_graph(cutoff, percentage, pos_label, tab):
            return self.explainer.plot_classification(
                    cutoff=cutoff, percentage=percentage, pos_label=pos_label)

class RocAucComponent(ExplainerComponent):
    def __init__(self, explainer, title="Precision Plot",
                    header_mode="none", name=None, 
                    hide_cutoff=False,
                    cutoff=0.5):
        super().__init__(explainer, title, header_mode, name)

        self.hide_cutoff = hide_cutoff
        self.cutoff=cutoff
        self.cutoff_name = 'rocauc-cutoff-' + self.name
        self.register_dependencies("preds")

    def _layout(self):
        return html.Div([
            dcc.Loading(id="loading-roc-auc-graph-"+self.name, 
                                children=[dcc.Graph(id='rocauc-graph-'+self.name)]),
            make_hideable(
                html.Div([
                    html.Label('Cutoff prediction probability:'),
                    dcc.Slider(id='rocauc-cutoff-'+self.name, 
                                min = 0.01, max = 0.99, step=0.01, value=self.cutoff,
                                marks={0.01: '0.01', 0.25: '0.25', 0.50: '0.50',
                                        0.75: '0.75', 0.99: '0.99'}, 
                                included=False,
                                tooltip = {'always_visible' : False},
                                updatemode='drag' )
                ], style={'margin-bottom': 25}), hide=self.hide_cutoff),
        ])

    def _register_callbacks(self, app):
        @app.callback(
            Output('rocauc-graph-'+self.name, 'figure'),
            [Input('rocauc-cutoff-'+self.name, 'value'),
             Input('pos-label', 'value'),
             Input('tabs', 'value')],
        )
        def update_precision_graph(cutoff, pos_label, tab):
            return self.explainer.plot_roc_auc(cutoff=cutoff, pos_label=pos_label)


class PrAucComponent(ExplainerComponent):
    def __init__(self, explainer, title="Precision Plot",
                    header_mode="none", name=None,
                    hide_cutoff=False,
                    cutoff=0.5):
        super().__init__(explainer, title, header_mode, name)

        self.hide_cutoff = hide_cutoff
        self.cutoff = cutoff
        self.cutoff_name = 'prauc-cutoff-' + self.name
        self.register_dependencies("preds")

    def _layout(self):
        return html.Div([
            dcc.Loading(id="loading-pr-auc-graph-"+self.name, 
                                children=[dcc.Graph(id='prauc-graph-'+self.name)]),
            make_hideable(
                html.Div([
                    html.Label('Cutoff prediction probability:'),
                    dcc.Slider(id='prauc-cutoff-'+self.name, 
                                min = 0.01, max = 0.99, step=0.01, value=self.cutoff,
                                marks={0.01: '0.01', 0.25: '0.25', 0.50: '0.50',
                                        0.75: '0.75', 0.99: '0.99'}, 
                                included=False,
                                tooltip = {'always_visible' : False})
                ], style={'margin-bottom': 25}), hide=self.hide_cutoff),
        ])

    def _register_callbacks(self, app):
        @app.callback(
            Output('prauc-graph-'+self.name, 'figure'),
            [Input('prauc-cutoff-'+self.name, 'value'),
             Input('pos-label', 'value')],
            [State('tabs', 'value')],
        )
        def update_precision_graph(cutoff, pos_label, tab):
            return self.explainer.plot_pr_auc(cutoff=cutoff, pos_label=pos_label)

class CutoffConnector(ExplainerComponent):
    """
    updates cutoff properties of other components given by component list
    e.g. 'precision-cutoff', 'confusionmatrix-cutoff', etc. 
    """
    def __init__(self, explainer, title="Global cutoff",
                        header_mode="none", name=None,
                        cutoff_components=None,
                        hide_cutoff=False, hide_percentile=False,
                        cutoff=0.5, percentile=None):
        super().__init__(explainer, title, header_mode, name)

        self.cutoff_names = [comp.cutoff_name for comp in cutoff_components]

        self.hide_cutoff = hide_cutoff
        self.hide_percentile = hide_percentile
        self.cutoff, self.percentile = cutoff, percentile
        self.register_dependencies(['preds', 'pred_percentiles'])

    def _layout(self):
        return html.Div([
            make_hideable(
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.Label('Cutoff prediction probability:'),
                            dcc.Slider(id='cutoffconnector-cutoff-'+self.name, 
                                        min = 0.01, max = 0.99, step=0.01, value=self.cutoff,
                                        marks={0.01: '0.01', 0.25: '0.25', 0.50: '0.50',
                                                0.75: '0.75', 0.99: '0.99'}, 
                                        included=False,
                                        tooltip = {'always_visible' : False})
                        ], style={'margin-bottom': 25}),
                    ])
                ]), hide=self.hide_cutoff),
            make_hideable(
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.Label('Cutoff percentile of samples:'),
                            dcc.Slider(id='cutoffconnector-percentile-'+self.name, 
                                        min = 0.01, max = 0.99, step=0.01, value=self.percentile,
                                        marks={0.01: '0.01', 0.25: '0.25', 0.50: '0.50',
                                                0.75: '0.75', 0.99: '0.99'}, 
                                        included=False,
                                        tooltip = {'always_visible' : False})
                        ], style={'margin-bottom': 25}),
                    ])
                ]), hide=self.hide_percentile),
        ])

    def _register_callbacks(self, app):
        @app.callback(
            Output('cutoffconnector-cutoff-'+self.name, 'value'),
            [Input('cutoffconnector-percentile-'+self.name, 'value'),
             Input('pos-label', 'value')]
        )
        def update_cutoff(percentile, pos_label):
            if percentile is not None:
                return np.round(self.explainer.cutoff_from_percentile(percentile, pos_label=pos_label), 2)
            raise PreventUpdate

        @app.callback(
            [Output(cut, 'value') for cut in self.cutoff_names],
            [Input('cutoffconnector-cutoff-'+self.name, 'value')]
        )
        def update_cutoffs(cutoff):
            return tuple(cutoff for i in range(len(self.cutoff_names)))


class PredictedVsActualComponent(ExplainerComponent):
    def __init__(self, explainer, title="Predicted vs Actual",
                    header_mode="none", name=None,
                    hide_logs=False,
                    logs=False):
        super().__init__(explainer, title, header_mode, name)
        self.hide_logs = hide_logs
        self.logs = logs
        self.register_dependencies(['preds'])

    def _layout(self):
        return html.Div([
            dcc.Loading(id="loading-pred-vs-actual-graph-"+self.name, 
                                children=[dcc.Graph(id='pred-vs-actual-graph-'+self.name)]),
            make_hideable(
                html.Div([
                    dbc.FormGroup(
                    [
                        dbc.RadioButton(
                            id='pred-vs-actual-logs-'+self.name,
                            className="form-check-input",
                            checked=self.logs),
                        dbc.Label("Take Logs",
                                html_for='pred-vs-actual-logs-'+self.name,
                                className="form-check-label"),
                    ], check=True),
                ]), hide=self.hide_logs),
        ])

    def _register_callbacks(self, app):
        @app.callback(
            Output('pred-vs-actual-graph-'+self.name, 'figure'),
            [Input('pred-vs-actual-logs-'+self.name, 'checked')],
            [State('tabs', 'value')]
        )
        def update_predicted_vs_actual_graph(logs, tab):
            return self.explainer.plot_predicted_vs_actual(logs=logs)

class ResidualsComponent(ExplainerComponent):
    def __init__(self, explainer, title="Residuals",
                    header_mode="none", name=None,
                    hide_pred_or_actual=False, hide_ratio=False,
                    pred_or_actual="vs_pred", ratio=False):
        super().__init__(explainer, title, header_mode, name)

        self.hide_pred_or_actual = hide_pred_or_actual
        self.hide_ratio = hide_ratio
        self.pred_or_actual = pred_or_actual
        self.ratio = ratio
        self.register_dependencies(['preds', 'residuals'])

    def _layout(self):
        return html.Div([
            dcc.Loading(id="loading-residuals-graph-"+self.name, 
                                children=[dcc.Graph(id='residuals-graph-'+self.name)]),
            make_hideable(
                html.Div([
                    dbc.FormGroup(
                    [
                        dbc.RadioItems(
                            options=[
                                {"label": "vs Prediction", "value": "vs_pred"},
                                {"label": "vs Actual", "value": "vs_actual"},
                            ],
                            value=self.pred_or_actual,
                            id='residuals-pred-or-actual-'+self.name,
                            inline=True,
                        ),
                    ]),
                ]), hide=self.hide_pred_or_actual),
            make_hideable(
                html.Div([
                    dbc.FormGroup(
                    [
                        dbc.RadioButton(
                            id='residuals-ratio-'+self.name,
                            className="form-check-input",
                            checked=self.ratio),
                        dbc.Label("Display Ratio",
                                html_for='residuals-ratio-'+self.name,
                                className="form-check-label"),
                    ], check=True),
                ]), hide=self.hide_ratio),
        ])

    def register_callbacks(self, app):
        @app.callback(
            Output('residuals-graph-'+self.name, 'figure'),
            [Input('residuals-pred-or-actual-'+self.name, 'value'),
             Input('residuals-ratio-'+self.name, 'checked')],
            [State('tabs', 'value')],
        )
        def update_residuals_graph(pred_or_actual, ratio, tab):
            vs_actual = pred_or_actual=='vs_actual'
            return self.explainer.plot_residuals(vs_actual=vs_actual, ratio=ratio)

class ResidualsVsColComponent(ExplainerComponent):
    def __init__(self, explainer, title="Residuals vs feature",
                    header_mode="none", name=None,
                    hide_col=False, hide_ratio=False,
                    col=None, ratio=False):
        super().__init__(explainer, title, header_mode, name)

        self.hide_col, self.hide_ratio = hide_col, hide_ratio
        self.col = col
        if self.col is None:
            self.col = self.explainer.mean_abs_shap_df(cats=False)\
                                                .Feature.tolist()[0]
        self.ratio = ratio
        self.register_dependencies(['preds', 'residuals'])

    def _layout(self):
        return html.Div([
            dcc.Loading(id="loading-residuals-vs-col-graph-"+self.name, 
                                children=[dcc.Graph(id='residuals-vs-col-graph-'+self.name)]),
            make_hideable(
                html.Div([
                    dbc.Label("Column:"),
                    dcc.Dropdown(id='residuals-vs-col-col-'+self.name,
                        options = [{'label': col, 'value': col} 
                                        for col in self.explainer.mean_abs_shap_df(cats=False)\
                                                        .Feature.tolist()],
                        value=self.col),
                ]), hide=self.hide_col),
            make_hideable(
                html.Div([
                    dbc.FormGroup(
                    [
                        dbc.RadioButton(
                            id='residuals-vs-col-ratio-'+self.name,
                            className="form-check-input",
                            checked=self.ratio),
                        dbc.Label("Display Ratio",
                                html_for='residuals-vs-col-ratio-'+self.name,
                                className="form-check-label"),
                    ], check=True),
                ]), hide=self.hide_ratio),
        ])

    def register_callbacks(self, app):
        @app.callback(
            Output('residuals-vs-col-graph-'+self.name, 'figure'),
            [Input('residuals-vs-col-col-'+self.name, 'value'),
             Input('residuals-vs-col-ratio-'+self.name, 'checked')],
            [State('tabs', 'value')],
        )
        def update_residuals_graph(col, ratio, tab):
            return self.explainer.plot_residuals_vs_feature(col, ratio=ratio, dropna=True)

class RegressionModelSummaryComponent(ExplainerComponent):
    def __init__(self, explainer, title="Model Summary",
                    header_mode="none", name=None):
        super().__init__(explainer, title, header_mode, name)
        self.register_dependencies(['preds'])

    def _layout(self):
        return html.Div([
            dcc.Loading(id='loading-model-summary-'+self.name, 
                                children=[dcc.Markdown(id='model-summary-'+self.name)]),
        ])

    def _register_callbacks(self, app):
        @app.callback(
            Output('model-summary-'+self.name, 'children'),
            [Input('pos-label', 'value')],
            [State('tabs', 'value')]
        )
        def update_model_summary(pos_label, tab):
            return self.explainer.metrics_markdown()

class ImportancesComponent(ExplainerComponent):
    def __init__(self, explainer, title="Importances",
                        header_mode="none", name=None,
                        hide_type=False, hide_depth=False, hide_cats=False,
                        importance_type="shap", depth=None, cats=True):
        super().__init__(explainer, title, header_mode, name)

        self.hide_type = hide_type
        self.hide_depth = hide_depth
        self.hide_cats = hide_cats
        if self.explainer.cats is None or not self.explainer.cats:
            self.hide_cats = True

        assert importance_type in ['shap', 'permutation'], \
            "importance type must be either 'shap' or 'permutation'!"
        self.importance_type = importance_type
        if depth is not None:
            depth = min(depth, len(explainer.columns_ranked_by_shap(cats)))
        self.depth = depth
        self.cats = cats
        self.register_dependencies(['shap_values', 'shap_values_cats',
            'permutation_importances', 'permutation_importances_cats'])

    def _layout(self):
        return dbc.Container([
            dbc.Row([dbc.Col([html.H2('Feature Importances:')])]),
            dbc.Row([
                make_hideable(
                    dbc.Col([
                        dbc.FormGroup([
                            dbc.Label("Importances type:"),
                            dbc.RadioItems(
                                options=[
                                    {'label': 'Permutation Importances', 
                                    'value': 'permutation'},
                                    {'label': 'SHAP values', 
                                    'value': 'shap'}
                                ],
                                value=self.importance_type,
                                id='importances-permutation-or-shap-'+self.name,
                                inline=True,
                            ),
                        ]),
                    ]), self.hide_type),
                make_hideable(
                    dbc.Col([
                        html.Label('Depth:'),
                        dcc.Dropdown(id='importances-depth-'+self.name,
                                            options = [{'label': str(i+1), 'value':i+1} 
                                                        for i in range(len(self.explainer.columns_ranked_by_shap(self.cats)))],
                                            value=self.depth)
                    ]), self.hide_depth),
                make_hideable(
                    dbc.Col([
                        dbc.Label("Grouping:"),
                        dbc.FormGroup(
                        [
                            dbc.RadioButton(
                                id='importances-group-cats-'+self.name, 
                                className="form-check-input",
                                checked=self.cats),
                            dbc.Label("Group Cats",
                                    html_for='importances-group-cats-'+self.name,
                                    className="form-check-label"),
                        ], check=True), 
                    ]),  self.hide_cats),        
            ], form=True),

            dbc.Row([
                dbc.Col([
                    dcc.Loading(id='loading-importances-graph-'+self.name, 
                            children=[dcc.Graph(id='importances-graph-'+self.name)])
                ]),
            ]), 
            ], fluid=True)
        
    def _register_callbacks(self, app, **kwargs):
        @app.callback(  
            Output('importances-graph-'+self.name, 'figure'),
            [Input('importances-depth-'+self.name, 'value'),
             Input('importances-group-cats-'+self.name, 'checked'),
             Input('importances-permutation-or-shap-'+self.name, 'value'),
             Input('pos-label', 'value')],
            [State('tabs', 'value')]
        )
        def update_importances(depth, cats, permutation_shap, pos_label, tab): 
            return self.explainer.plot_importances(
                        kind=permutation_shap, topx=depth, 
                        cats=cats, pos_label=pos_label)

class ClassifierModelStatsComponent(ExplainerComponent):
    def __init__(self, explainer, title="Classification Stats", 
                    header_mode="none", name=None,
                    bin_size=0.1, quantiles=10, cutoff=0.5):
        super().__init__(explainer, title, header_mode, name)

        self.precision = PrecisionComponent(explainer)
        self.confusionmatrix = ConfusionMatrixComponent(explainer)
        self.liftcurve = LiftCurveComponent(explainer)
        self.classification = ClassificationComponent(explainer)
        self.rocauc = RocAucComponent(explainer)
        self.prauc = PrAucComponent(explainer)

        self.cutoffconnector = CutoffConnector(explainer,
            cutoff_components=[self.precision, self.confusionmatrix, 
                self.liftcurve, self.classification, self.rocauc, self.prauc])

        self.register_components(
            self.precision, self.confusionmatrix, self.liftcurve,
            self.classification, self.rocauc, self.prauc,
            self.cutoffconnector)

    def _layout(self):
        return dbc.Container([
            dbc.Row([dbc.Col([html.H2('Model Performance:')])]),
            self.cutoffconnector.layout(),
            dbc.Row([
                dbc.Col([
                    self.precision.layout()
                ], md=6, align="start"),
                dbc.Col([
                    self.confusionmatrix.layout()
                ], md=6, align="start"),              
            ]),
            dbc.Row([
                dbc.Col([
                    self.liftcurve.layout()         
                ], md=6, align="start"),
                dbc.Col([
                    self.classification.layout()
                ], md=6, align="start"),
            ]),
            dbc.Row([    
                dbc.Col([
                    self.rocauc.layout()
                ], md=6),
                dbc.Col([
                    self.prauc.layout()
                ], md=6),
            ]),
        ], fluid=True)


class RegressionModelStatsComponent(ExplainerComponent):
    def __init__(self, explainer, title="Regression Stats", 
                    header_mode="none", name=None,
                    logs=False, pred_or_actual="vs_pred", ratio=False,
                    col=None):
        super().__init__(explainer, title, header_mode, name)
     
        assert pred_or_actual in ['vs_actual', 'vs_pred'], \
            "pred_or_actual should be 'vs_actual' or 'vs_pred'!"

        self.preds_vs_actual = PredictedVsActualComponent(explainer, logs=logs)
        self.modelsummary = RegressionModelSummaryComponent(explainer)
        self.residuals = ResidualsComponent(explainer, 
                            pred_or_actual=pred_or_actual, ratio=ratio)
        self.residuals_vs_col = ResidualsVsColComponent(explainer, 
                                    col=col, ratio=ratio)
        self.register_components([self.preds_vs_actual, self.modelsummary,
                    self.residuals, self.residuals_vs_col])

    def _layout(self):
        return dbc.Container([
            dbc.Row([dbc.Col([html.H2('Model Performance:')])]),
            dbc.Row([
                dbc.Col([
                    self.preds_vs_actual.layout()
                ], md=6),
                dbc.Col([
                    self.modelsummary.layout()     
                ], md=6),
            ], align="start"),
            dbc.Row([
                dbc.Col([
                    self.residuals.layout()
                ], md=6),
                dbc.Col([
                    self.residuals_vs_col.layout()
                ], md=6),
            ])
        ], fluid=True)