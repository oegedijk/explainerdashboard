__all__ = [
    'PredictedVsActualComponent',
    'ResidualsComponent',
    'ResidualsVsColComponent',
    'ActualVsColComponent',
    'PredsVsColComponent',
    'RegressionModelSummaryComponent',
]
import pandas as pd

import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html

from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from ..dashboard_methods import *


class PredictedVsActualComponent(ExplainerComponent):
    def __init__(self, explainer, title="Predicted vs Actual", name=None,
                    hide_title=False, hide_log_x=False, hide_log_y=False,
                    logs=False, log_x=False, log_y=False):
        """Shows a plot of predictions vs y.

        Args:
            explainer (Explainer): explainer object constructed with either
                        ClassifierExplainer() or RegressionExplainer()
            title (str, optional): Title of tab or page. Defaults to 
                        "Predicted vs Actual".
            name (str, optional): unique name to add to Component elements. 
                        If None then random uuid is generated to make sure 
                        it's unique. Defaults to None.
            hide_title (bool, optional) Hide the title. Defaults to False.
            hide_log_x (bool, optional): Hide the log_x toggle. Defaults to False.
            hide_log_y (bool, optional): Hide the log_y toggle. Defaults to False.
            logs (bool, optional): Whether to use log axis. Defaults to False.
            log_x (bool, optional): log only x axis. Defaults to False.
            log_y (bool, optional): log only y axis. Defaults to False.
        """
        super().__init__(explainer, title, name)
        
        self.logs, self.log_x, self.log_y = logs, log_x, log_y
        self.register_dependencies(['preds'])

    def layout(self):
        return html.Div([
            dbc.Row([
                make_hideable(html.H3("Predictions"), hide=self.hide_title)
            ]),
            dbc.Row([
                make_hideable(
                    dbc.Col([
                        dbc.FormGroup(
                        [
                            # html.Label("Log y"),
                            dbc.RadioButton(
                                id='pred-vs-actual-logy-'+self.name,
                                className="form-check-input",
                                checked=self.log_y),   
                            dbc.Label("Log y",
                                    html_for='pred-vs-actual-logy-'+self.name,
                                    className="form-check-label"), 
                        ], check=True),
                    ], md=1, align="center"), hide=self.hide_log_y),
                dbc.Col([
                    dcc.Graph(id='pred-vs-actual-graph-'+self.name,
                                                    config=dict(modeBarButtons=[['toImage']], displaylogo=False)),
                ], md=11)
            ]),
            dbc.Row([
                make_hideable(
                    dbc.Col([
                        dbc.FormGroup(
                        [
                            dbc.RadioButton(
                                id='pred-vs-actual-logx-'+self.name,
                                className="form-check-input",
                                checked=self.log_x),
                            dbc.Label("Log x",
                                    html_for='pred-vs-actual-logx-'+self.name,
                                    className="form-check-label"),   
                        ], check=True),
                    ], md=2), hide=self.hide_log_x),
            ], justify="center")   
        ])

    def _register_callbacks(self, app):
        @app.callback(
            Output('pred-vs-actual-graph-'+self.name, 'figure'),
            [Input('pred-vs-actual-logx-'+self.name, 'checked'),
             Input('pred-vs-actual-logy-'+self.name, 'checked')],
        )
        def update_predicted_vs_actual_graph(log_x, log_y):
            return self.explainer.plot_predicted_vs_actual(log_x=log_x, log_y=log_y)

class ResidualsComponent(ExplainerComponent):
    def __init__(self, explainer, title="Residuals", name=None,
                    hide_title=False, hide_pred_or_actual=False, hide_ratio=False,
                    pred_or_actual="vs_pred", residuals="difference"):
        """Residuals plot component

        Args:
            explainer (Explainer): explainer object constructed with either
                        ClassifierExplainer() or RegressionExplainer()
            title (str, optional): Title of tab or page. Defaults to 
                        "Residuals".
            name (str, optional): unique name to add to Component elements. 
                        If None then random uuid is generated to make sure 
                        it's unique. Defaults to None.
            hide_title (bool, optional) Hide the title. Defaults to False.
            hide_pred_or_actual (bool, optional): hide vs predictions or vs 
                        actual for x-axis toggle. Defaults to False.
            hide_ratio (bool, optional): hide residual type dropdown. Defaults to False.
            pred_or_actual (str, {'vs_actual', 'vs_pred'}, optional): Whether 
                        to plot actual or predictions on the x-axis. 
                        Defaults to "vs_pred".
            residuals (str, {'difference', 'ratio', 'log-ratio'} optional): 
                    How to calcualte residuals. Defaults to 'difference'.
        """
        super().__init__(explainer, title, name)

        assert residuals in ['difference', 'ratio', 'log-ratio'], \
            ("parameter residuals should in ['difference', 'ratio', 'log-ratio']"
             f" but you passed residuals={residuals}")
        self.register_dependencies(['preds', 'residuals'])

    def layout(self):
        return html.Div([
            dbc.Row([
                make_hideable(html.H3("Residuals"), hide=self.hide_title)
            ]),
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id='residuals-graph-'+self.name,
                                                    config=dict(modeBarButtons=[['toImage']], displaylogo=False)),
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
                    ], md=3), hide=self.hide_pred_or_actual),
                make_hideable(
                    dbc.Col([
                        html.Label('Residual type:'),
                        dcc.Dropdown(id='residuals-type-'+self.name,
                                options = [{'label': 'Difference', 'value': 'difference'},
                                            {'label': 'Ratio', 'value': 'ratio'},
                                            {'label': 'Log ratio', 'value': 'log-ratio'}],
                                value=self.residuals),
                    ], md=3), hide=self.hide_ratio),
            ], justify="center")  
        ])

    def register_callbacks(self, app):
        @app.callback(
            Output('residuals-graph-'+self.name, 'figure'),
            [Input('residuals-pred-or-actual-'+self.name, 'value'),
             Input('residuals-type-'+self.name, 'value')],
        )
        def update_residuals_graph(pred_or_actual, residuals):
            vs_actual = pred_or_actual=='vs_actual'
            return self.explainer.plot_residuals(vs_actual=vs_actual, residuals=residuals)

class ResidualsVsColComponent(ExplainerComponent):
    def __init__(self, explainer, title="Residuals vs feature", name=None,
                    hide_title=False, hide_col=False, hide_ratio=False, hide_cats=False, 
                    hide_points=False, hide_winsor=False,
                    col=None, residuals='difference', cats=True, 
                    points=True, winsor=0):
        """Show residuals vs a particular Feature component

        Args:
            explainer (Explainer): explainer object constructed with either
                        ClassifierExplainer() or RegressionExplainer()
            title (str, optional): Title of tab or page. Defaults to 
                        "Residuals vs feature".
            name (str, optional): unique name to add to Component elements. 
                        If None then random uuid is generated to make sure 
                        it's unique. Defaults to None.
            hide_title (bool, optional) Hide the title. Defaults to False.
            hide_col (bool, optional): Hide de column selector. Defaults to False.
            hide_ratio (bool, optional): Hide the  toggle. Defaults to False.
            hide_cats (bool, optional): Hide group cats toggle. Defaults to False.
            hide_points (bool, optional): Hide group points toggle. Defaults to False.
            hide_winsor (bool, optional): Hide winsor input. Defaults to False.
            col ([type], optional): Initial feature to display. Defaults to None.
            residuals (str, {'difference', 'ratio', 'log-ratio'} optional): 
                    How to calcualte residuals. Defaults to 'difference'.
            cats (bool, optional): group categorical columns. Defaults to True.
            points (bool, optional): display point cloud next to violin plot 
                    for categorical cols. Defaults to True
            winsor (int, 0-50, optional): percentage of outliers to winsor out of 
                    the y-axis. Defaults to 0.
        """
        super().__init__(explainer, title, name)

        if self.col is None:
            self.col = self.explainer.columns_ranked_by_shap(self.cats)[0]
        
        assert residuals in ['difference', 'ratio', 'log-ratio'], \
            ("parameter residuals should in ['difference', 'ratio', 'log-ratio']"
             f" but you passed residuals={residuals}")
        self.register_dependencies(['preds', 'residuals'])

    def layout(self):
        return html.Div([
            dbc.Row([
                make_hideable(html.H3("Residuals vs Feature"), hide=self.hide_title)
            ]),
            dbc.Row([
                make_hideable(
                    dbc.Col([
                        dbc.Label("Column:"),
                        dcc.Dropdown(id='residuals-vs-col-col-'+self.name,
                            options=[{'label': col, 'value':col} 
                                            for col in self.explainer.columns_ranked_by_shap(self.cats)],
                            value=self.col),
                    ], md=4), hide=self.hide_col),
                make_hideable(
                        dbc.Col([
                            dbc.Label("Grouping:"),
                            dbc.FormGroup(
                            [
                                dbc.RadioButton(
                                    id='residuals-vs-col-group-cats-'+self.name, 
                                    className="form-check-input",
                                    checked=self.cats),
                                dbc.Label("Group Cats",
                                        html_for='residuals-vs-col-group-cats-'+self.name,
                                        className="form-check-label"),
                            ], check=True),
                        ], md=2), self.hide_cats),
            ]),
            dbc.Row([
                dcc.Graph(id='residuals-vs-col-graph-'+self.name,
                            config=dict(modeBarButtons=[['toImage']], displaylogo=False)),
            ]),
            dbc.Row([
                make_hideable(
                        dbc.Col([
                            html.Label('Residual type:'),
                            dcc.Dropdown(id='residuals-vs-col-residuals-type-'+self.name,
                                    options = [{'label': 'Difference', 'value': 'difference'},
                                                {'label': 'Ratio', 'value': 'ratio'},
                                                {'label': 'Log ratio', 'value': 'log-ratio'}],
                                    value=self.residuals),
                        ], md=3), hide=self.hide_ratio),
                make_hideable(
                        dbc.Col([ 
                            dbc.Label("Winsor:"),
                            dbc.Input(id='residuals-vs-col-winsor-'+self.name, 
                                    value=self.winsor,
                                type="number", min=0, max=49, step=1),
                        ], md=2), hide=self.hide_winsor),  
                make_hideable(
                        dbc.Col([
                            dbc.Label("Points:"),
                            dbc.FormGroup(
                            [
                                dbc.RadioButton(
                                    id='residuals-vs-col-show-points-'+self.name, 
                                    className="form-check-input",
                                    checked=self.points),
                                dbc.Label("Show points",
                                        html_for='residuals-vs-col-show-points-'+self.name,
                                        className="form-check-label"),
                            ], check=True),
                        ],  md=3), self.hide_points),
            ]),
        ])

    def register_callbacks(self, app):
        @app.callback(
            Output('residuals-vs-col-graph-'+self.name, 'figure'),
            [Input('residuals-vs-col-col-'+self.name, 'value'),
             Input('residuals-vs-col-residuals-type-'+self.name, 'value'),
             Input('residuals-vs-col-show-points-'+self.name, 'checked'),
             Input('residuals-vs-col-winsor-'+self.name, 'value')],
        )
        def update_residuals_graph(col, residuals, points, winsor):
            return self.explainer.plot_residuals_vs_feature(
                        col, residuals=residuals, points=points, 
                        winsor=winsor, dropna=True)

        @app.callback(
            Output('residuals-vs-col-col-'+self.name, 'options'),
            [Input('residuals-vs-col-group-cats-'+self.name, 'checked')])
        def update_dependence_shap_scatter_graph(cats):
            return [{'label': col, 'value': col} 
                for col in self.explainer.columns_ranked_by_shap(cats)]


class ActualVsColComponent(ExplainerComponent):
    def __init__(self, explainer, title="Observed vs feature", name=None,
                    hide_title=False, hide_col=False, hide_ratio=False, hide_cats=False, 
                    hide_points=False, hide_winsor=False,
                    col=None, cats=True, 
                    points=True, winsor=0):
        """Show residuals vs a particular Feature component

        Args:
            explainer (Explainer): explainer object constructed with either
                        ClassifierExplainer() or RegressionExplainer()
            title (str, optional): Title of tab or page. Defaults to 
                        "Observed vs feature".
            name (str, optional): unique name to add to Component elements. 
                        If None then random uuid is generated to make sure 
                        it's unique. Defaults to None.
            hide_title (bool, optional) Hide the title. Defaults to False.
            hide_col (bool, optional): Hide de column selector. Defaults to False.
            hide_ratio (bool, optional): Hide the  toggle. Defaults to False.
            hide_cats (bool, optional): Hide group cats toggle. Defaults to False.
            hide_points (bool, optional): Hide group points toggle. Defaults to False.
            hide_winsor (bool, optional): Hide winsor input. Defaults to False.
            col ([type], optional): Initial feature to display. Defaults to None.
            cats (bool, optional): group categorical columns. Defaults to True.
            points (bool, optional): display point cloud next to violin plot 
                    for categorical cols. Defaults to True
            winsor (int, 0-50, optional): percentage of outliers to winsor out of 
                    the y-axis. Defaults to 0.
        """
        super().__init__(explainer, title, name)

        if self.col is None:
            self.col = self.explainer.columns_ranked_by_shap(self.cats)[0]

        self.register_dependencies(['preds'])

    def layout(self):
        return html.Div([
            dbc.Row([
                make_hideable(html.H3(self.title), hide=self.hide_title)
            ]),
            dbc.Row([
                make_hideable(
                    dbc.Col([
                        dbc.Label("Column:"),
                        dcc.Dropdown(id='observed-vs-col-col-'+self.name,
                            options=[{'label': col, 'value':col} 
                                            for col in self.explainer.columns_ranked_by_shap(self.cats)],
                            value=self.col),
                    ], md=4), hide=self.hide_col),
                make_hideable(
                        dbc.Col([
                            dbc.Label("Grouping:"),
                            dbc.FormGroup(
                            [
                                dbc.RadioButton(
                                    id='observed-vs-col-group-cats-'+self.name, 
                                    className="form-check-input",
                                    checked=self.cats),
                                dbc.Label("Group Cats",
                                        html_for='observed-vs-col-group-cats-'+self.name,
                                        className="form-check-label"),
                            ], check=True),
                        ], md=2), self.hide_cats),
            ]),
            dbc.Row([
                dcc.Graph(id='observed-vs-col-graph-'+self.name,
                            config=dict(modeBarButtons=[['toImage']], displaylogo=False)),
            ]),
            dbc.Row([
                make_hideable(
                        dbc.Col([ 
                            dbc.Label("Winsor:"),
                            dbc.Input(id='observed-vs-col-winsor-'+self.name, 
                                    value=self.winsor,
                                type="number", min=0, max=49, step=1),
                        ], md=2), hide=self.hide_winsor),  
                make_hideable(
                        dbc.Col([
                            dbc.Label("Points:"),
                            dbc.FormGroup(
                            [
                                dbc.RadioButton(
                                    id='observed-vs-col-show-points-'+self.name, 
                                    className="form-check-input",
                                    checked=self.points),
                                dbc.Label("Show points",
                                        html_for='observed-vs-col-show-points-'+self.name,
                                        className="form-check-label"),
                            ], check=True),
                        ],  md=3), self.hide_points),
            ]),
        ])

    def register_callbacks(self, app):
        @app.callback(
            Output('observed-vs-col-graph-'+self.name, 'figure'),
            [Input('observed-vs-col-col-'+self.name, 'value'),
             Input('observed-vs-col-show-points-'+self.name, 'checked'),
             Input('observed-vs-col-winsor-'+self.name, 'value')],
        )
        def update_observed_vs_col_graph(col, points, winsor):
            return self.explainer.plot_y_vs_feature(
                        col, points=points, winsor=winsor, dropna=True)

        @app.callback(
            Output('observed-vs-col-col-'+self.name, 'options'),
            [Input('observed-vs-col-group-cats-'+self.name, 'checked')])
        def update_observed_vs_col_options(cats):
            return [{'label': col, 'value': col} 
                for col in self.explainer.columns_ranked_by_shap(cats)]


class PredsVsColComponent(ExplainerComponent):
    def __init__(self, explainer, title="Predictions vs feature", name=None,
                    hide_title=False, hide_col=False, hide_ratio=False, hide_cats=False, 
                    hide_points=False, hide_winsor=False,
                    col=None, cats=True, 
                    points=True, winsor=0):
        """Show residuals vs a particular Feature component

        Args:
            explainer (Explainer): explainer object constructed with either
                        ClassifierExplainer() or RegressionExplainer()
            title (str, optional): Title of tab or page. Defaults to 
                        "Predicions vs feature".
            name (str, optional): unique name to add to Component elements. 
                        If None then random uuid is generated to make sure 
                        it's unique. Defaults to None.
            hide_title (bool, optional) Hide the title. Defaults to False.
            hide_col (bool, optional): Hide de column selector. Defaults to False.
            hide_ratio (bool, optional): Hide the  toggle. Defaults to False.
            hide_cats (bool, optional): Hide group cats toggle. Defaults to False.
            hide_points (bool, optional): Hide group points toggle. Defaults to False.
            hide_winsor (bool, optional): Hide winsor input. Defaults to False.
            col ([type], optional): Initial feature to display. Defaults to None.
            cats (bool, optional): group categorical columns. Defaults to True.
            points (bool, optional): display point cloud next to violin plot 
                    for categorical cols. Defaults to True
            winsor (int, 0-50, optional): percentage of outliers to winsor out of 
                    the y-axis. Defaults to 0.
        """
        super().__init__(explainer, title, name)

        if self.col is None:
            self.col = self.explainer.columns_ranked_by_shap(self.cats)[0]

        self.register_dependencies(['preds'])

    def layout(self):
        return html.Div([
            dbc.Row([
                make_hideable(html.H3(self.title), hide=self.hide_title)
            ]),
            dbc.Row([
                make_hideable(
                    dbc.Col([
                        dbc.Label("Column:"),
                        dcc.Dropdown(id='preds-vs-col-col-'+self.name,
                            options=[{'label': col, 'value':col} 
                                            for col in self.explainer.columns_ranked_by_shap(self.cats)],
                            value=self.col),
                    ], md=4), hide=self.hide_col),
                make_hideable(
                        dbc.Col([
                            dbc.Label("Grouping:"),
                            dbc.FormGroup(
                            [
                                dbc.RadioButton(
                                    id='preds-vs-col-group-cats-'+self.name, 
                                    className="form-check-input",
                                    checked=self.cats),
                                dbc.Label("Group Cats",
                                        html_for='preds-vs-col-group-cats-'+self.name,
                                        className="form-check-label"),
                            ], check=True),
                        ], md=2), self.hide_cats),
            ]),
            dbc.Row([
                dcc.Graph(id='preds-vs-col-graph-'+self.name,
                            config=dict(modeBarButtons=[['toImage']], displaylogo=False)),
            ]),
            dbc.Row([
                make_hideable(
                        dbc.Col([ 
                            dbc.Label("Winsor:"),
                            dbc.Input(id='preds-vs-col-winsor-'+self.name, 
                                    value=self.winsor,
                                type="number", min=0, max=49, step=1),
                        ], md=2), hide=self.hide_winsor),  
                make_hideable(
                        dbc.Col([
                            dbc.Label("Points:"),
                            dbc.FormGroup(
                            [
                                dbc.RadioButton(
                                    id='preds-vs-col-show-points-'+self.name, 
                                    className="form-check-input",
                                    checked=self.points),
                                dbc.Label("Show points",
                                        html_for='preds-vs-col-show-points-'+self.name,
                                        className="form-check-label"),
                            ], check=True),
                        ],  md=3), self.hide_points),
            ]),
        ])

    def register_callbacks(self, app):
        @app.callback(
            Output('preds-vs-col-graph-'+self.name, 'figure'),
            [Input('preds-vs-col-col-'+self.name, 'value'),
             Input('preds-vs-col-show-points-'+self.name, 'checked'),
             Input('preds-vs-col-winsor-'+self.name, 'value')],
        )
        def update_preds_vs_col_graph(col, points, winsor):
            return self.explainer.plot_preds_vs_feature(
                        col, points=points, winsor=winsor, dropna=True)

        @app.callback(
            Output('preds-vs-col-col-'+self.name, 'options'),
            [Input('preds-vs-col-group-cats-'+self.name, 'checked')])
        def update_preds_vs_col_options(cats):
            return [{'label': col, 'value': col} 
                for col in self.explainer.columns_ranked_by_shap(cats)]


class RegressionModelSummaryComponent(ExplainerComponent):
    def __init__(self, explainer, title="Model Summary", name=None, round=3):
        """Show model summary statistics (RMSE, MAE, R2) component

        Args:
            explainer (Explainer): explainer object constructed with either
                        ClassifierExplainer() or RegressionExplainer()
            title (str, optional): Title of tab or page. Defaults to 
                        "Model Summary".
            name (str, optional): unique name to add to Component elements. 
                        If None then random uuid is generated to make sure 
                        it's unique. Defaults to None.
            round (int): rounding to perform to metric floats.
        """
        super().__init__(explainer, title, name)
        self.register_dependencies(['preds', 'residuals'])

    def layout(self):
        metrics_df = (pd.DataFrame(self.explainer.metrics(), index=["Score"]).T
                        .rename_axis(index="metric").reset_index().round(self.round))
        return html.Div([
            html.H3("Regression performance metrics:"),
            dbc.Table.from_dataframe(metrics_df, striped=False, bordered=False, hover=False)
        ])
