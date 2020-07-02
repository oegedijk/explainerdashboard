__all__ = [
    'PredictedVsActualComponent',
    'ResidualsComponent',
    'ResidualsVsColComponent',
    'RegressionModelSummaryComponent',
]

import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html

from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from .dashboard_methods import *


class PredictedVsActualComponent(ExplainerComponent):
    def __init__(self, explainer, title="Predicted vs Actual",
                    header_mode="none", name=None,
                    hide_logs=False,
                    logs=False):
        """Shows a plot of predictions vs y.

        Args:
            explainer (Explainer): explainer object constructed with either
                        ClassifierExplainer() or RegressionExplainer()
            title (str, optional): Title of tab or page. Defaults to 
                        "Predicted vs Actual".
            header_mode (str, optional): {"standalone", "hidden" or "none"}. 
                        Defaults to "none".
            name (str, optional): unique name to add to Component elements. 
                        If None then random uuid is generated to make sure 
                        it's unique. Defaults to None.
            hide_logs (bool, optional): Hide the logs toggle. Defaults to False.
            logs (bool, optional): Whether to use log axis. Defaults to False.
        """
        super().__init__(explainer, title, header_mode, name)
        self.hide_logs = hide_logs
        self.logs = logs
        self.register_dependencies(['preds'])

    def _layout(self):
        return html.Div([
            dbc.Row([
                dbc.Col([
                    dcc.Loading(id="loading-pred-vs-actual-graph-"+self.name, 
                                children=[dcc.Graph(id='pred-vs-actual-graph-'+self.name)]),
                ])
            ]),
            dbc.Row([
                make_hideable(
                    dbc.Col([
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
        """Residuals plot component

        Args:
            explainer (Explainer): explainer object constructed with either
                        ClassifierExplainer() or RegressionExplainer()
            title (str, optional): Title of tab or page. Defaults to 
                        "Residuals".
            header_mode (str, optional): {"standalone", "hidden" or "none"}. 
                        Defaults to "none".
            name (str, optional): unique name to add to Component elements. 
                        If None then random uuid is generated to make sure 
                        it's unique. Defaults to None.
            hide_pred_or_actual (bool, optional): hide vs predictions or vs 
                        actual for x-axis toggle. Defaults to False.
            hide_ratio (bool, optional): hide ratio toggle. Defaults to False.
            pred_or_actual (str, {'vs_actual', 'vs_pred'}, optional): Whether 
                        to plot actual or predictions on the x-axis. 
                        Defaults to "vs_pred".
            ratio (bool, optional): Show the residual/prediction or residual/actual 
                        ratio instead of raw residuals. Defaults to False.
        """
        super().__init__(explainer, title, header_mode, name)

        self.hide_pred_or_actual = hide_pred_or_actual
        self.hide_ratio = hide_ratio
        self.pred_or_actual = pred_or_actual
        self.ratio = ratio
        self.register_dependencies(['preds', 'residuals'])

    def _layout(self):
        return html.Div([
            dbc.Row([
                dbc.Col([
                    dcc.Loading(id="loading-residuals-graph-"+self.name, 
                                children=[dcc.Graph(id='residuals-graph-'+self.name)]),
                ])
            ]),
            dbc.Row([
                make_hideable(
                    dbc.Col([
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
                    dbc.Col([
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
        """Show residuals vs a particular Feature component

        Args:
            explainer (Explainer): explainer object constructed with either
                        ClassifierExplainer() or RegressionExplainer()
            title (str, optional): Title of tab or page. Defaults to 
                        "Residuals vs feature".
            header_mode (str, optional): {"standalone", "hidden" or "none"}. 
                        Defaults to "none".
            name (str, optional): unique name to add to Component elements. 
                        If None then random uuid is generated to make sure 
                        it's unique. Defaults to None.
            hide_col (bool, optional): Hide de column selector. Defaults to False.
            hide_ratio (bool, optional): Hide the ratio toggle. Defaults to False.
            col ([type], optional): Initial feature to display. Defaults to None.
            ratio (bool, optional): Whether to display residual ratio instead 
                        of residuals. Defaults to False.
        """
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
        """Show model summary statistics (RMSE, MAE, R2) component

        Args:
            explainer (Explainer): explainer object constructed with either
                        ClassifierExplainer() or RegressionExplainer()
            title (str, optional): Title of tab or page. Defaults to 
                        "Model Summary".
            header_mode (str, optional): {"standalone", "hidden" or "none"}. 
                        Defaults to "none".
            name (str, optional): unique name to add to Component elements. 
                        If None then random uuid is generated to make sure 
                        it's unique. Defaults to None.
        """
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
