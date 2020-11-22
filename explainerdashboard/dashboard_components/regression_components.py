__all__ = [
    'PredictedVsActualComponent',
    'ResidualsComponent',
    'RegressionVsColComponent',
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
                    logs=False, log_x=False, log_y=False, **kwargs):
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

        self.description = f"""
        Plot shows the observed {self.explainer.target} and the predicted 
        {self.explainer.target} in the same plot. A perfect model would have
        all the points on the diagonal (predicted matches observed). The further
        away point are from the diagonal the worse the model is in predicting
        {self.explainer.target}.
        """
        self.register_dependencies(['preds'])

    def layout(self):
        return html.Div([
            dbc.Row([
                make_hideable(
                html.Div([
                    html.H3(self.title, id='pred-vs-actual-title-'+self.name),
                    dbc.Tooltip(self.description, target='pred-vs-actual-title-'+self.name),
                ]), hide=self.hide_title),
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
                            dbc.Tooltip("By using a log axis, it is easier to see relative "
                                    "errors instead of absolute errors.",
                                    target='pred-vs-actual-logy-'+self.name),
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
                            dbc.Tooltip("By using a log axis, it is easier to see relative "
                                    "errors instead of absolute errors.",
                                    target='pred-vs-actual-logx-'+self.name),
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
                    pred_or_actual="vs_pred", residuals="difference", **kwargs):
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
        self.description = f"""
        The residuals are the difference between the observed {self.explainer.target}
        and predicted {self.explainer.target}. In this plot you can check if 
        the residuals are higher or lower for higher/lower actual/predicted outcomes. 
        So you can check if the model works better or worse for different {self.explainer.target}
        levels.
        """
        self.register_dependencies(['preds', 'residuals'])

    def layout(self):
        return html.Div([
            dbc.Row([
                make_hideable(
                html.Div([
                    html.H3(self.title, id='residuals-title-'+self.name),
                    dbc.Tooltip(self.description, target='residuals-title-'+self.name),
                ]), hide=self.hide_title),
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
                        ], id='residuals-pred-or-actual-form-'+self.name),
                        dbc.Tooltip("Select what you would like to put on the x-axis:"
                            f" observed {self.explainer.target} or predicted {self.explainer.target}.",
                            target='residuals-pred-or-actual-form-'+self.name),
                    ], md=3), hide=self.hide_pred_or_actual),
                make_hideable(
                    dbc.Col([
                        html.Label('Residual type:', id='residuals-type-label-'+self.name),
                        dbc.Tooltip("Type of residuals to display: y-preds (difference), "
                                    "y/preds (ratio) or log(y/preds) (logratio).", 
                                    target='residuals-type-label-'+self.name),
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


class RegressionVsColComponent(ExplainerComponent):
    def __init__(self, explainer, title="Plot vs feature", name=None,
                    hide_title=False, hide_col=False, hide_ratio=False, hide_cats=False, 
                    hide_points=False, hide_winsor=False,
                    col=None, display='difference', cats=True, 
                    points=True, winsor=0, **kwargs):
        """Show residuals, observed or preds vs a particular Feature component

        Args:
            explainer (Explainer): explainer object constructed with either
                        ClassifierExplainer() or RegressionExplainer()
            title (str, optional): Title of tab or page. Defaults to 
                        "Plot vs feature".
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
            display (str, {'observed', 'predicted', difference', 'ratio', 'log-ratio'} optional): 
                    What to display on y axis. Defaults to 'difference'.
            cats (bool, optional): group categorical columns. Defaults to True.
            points (bool, optional): display point cloud next to violin plot 
                    for categorical cols. Defaults to True
            winsor (int, 0-50, optional): percentage of outliers to winsor out of 
                    the y-axis. Defaults to 0.
        """
        super().__init__(explainer, title, name)

        if self.col is None:
            self.col = self.explainer.columns_ranked_by_shap(self.cats)[0]
        
        assert self.display in {'observed', 'predicted', 'difference', 'ratio', 'log-ratio'}, \
            ("parameter display should in {'observed', 'predicted', 'difference', 'ratio', 'log-ratio'}"
             f" but you passed display={self.display}!")

        self.description = f"""
        This plot shows either residuals (difference between observed {self.explainer.target}
        and predicted {self.explainer.target}) plotted against the values of different features,
        or the observed or predicted {self.explainer.target}.
        This allows you to inspect whether the model is more wrong for particular
        range of feature values than others. 
        """
        self.register_dependencies(['preds', 'residuals'])

    def layout(self):
        return html.Div([
            dbc.Row([
                make_hideable(
                html.Div([
                    html.H3(self.title, id='reg-vs-col-title-'+self.name),
                    dbc.Tooltip(self.description, target='reg-vs-col-title-'+self.name),
                ]), hide=self.hide_title),
            ]),
            dbc.Row([
                make_hideable(
                    dbc.Col([
                        dbc.Label("Feature:", id='reg-vs-col-col-label-'+self.name),
                        dbc.Tooltip("Select the feature to display on the x-axis.", 
                                    target='reg-vs-col-col-label-'+self.name),
                        dcc.Dropdown(id='reg-vs-col-col-'+self.name,
                            options=[{'label': col, 'value':col} 
                                            for col in self.explainer.columns_ranked_by_shap(self.cats)],
                            value=self.col),
                    ], md=4), hide=self.hide_col),
                make_hideable(
                        dbc.Col([
                            html.Label('Display:', id='reg-vs-col-display-type-label-'+self.name),
                            dbc.Tooltip(f"Select what to display on the y axis: observed {self.explainer.target}, "
                                        f"predicted {self.explainer.target} or residuals. Residuals can either "
                                        "be calculated by takind the difference (y-preds), "
                                        "ratio (y/preds) or log ratio log(y/preds). The latter makes it easier to "
                                        "see relative differences.", 
                                        target='reg-vs-col-display-type-label-'+self.name),
                            dcc.Dropdown(id='reg-vs-col-display-type-'+self.name,
                                    options = [{'label': 'Observed', 'value': 'observed'},
                                                {'label': 'Predicted', 'value': 'predicted'},
                                                {'label': 'Residuals: Difference', 'value': 'difference'},
                                                {'label': 'Residuals: Ratio', 'value': 'ratio'},
                                                {'label': 'Residuals: Log ratio', 'value': 'log-ratio'}],
                                    value=self.display),
                        ], md=4), hide=self.hide_ratio),
                make_hideable(
                        dbc.Col([
                            dbc.Label("Grouping:", id='reg-vs-col-group-cats-label-'+self.name),
                            dbc.Tooltip("group onehot encoded features together.",
                                        target='reg-vs-col-group-cats-label-'+self.name),
                            dbc.FormGroup(
                            [
                                dbc.RadioButton(
                                    id='reg-vs-col-group-cats-'+self.name, 
                                    className="form-check-input",
                                    checked=self.cats),
                                dbc.Label("Group Cats",
                                        html_for='reg-vs-col-group-cats-'+self.name,
                                        className="form-check-label"),
                            ], check=True),
                        ], md=2), self.hide_cats),
            ]),
            dbc.Row([
                dcc.Graph(id='reg-vs-col-graph-'+self.name,
                            config=dict(modeBarButtons=[['toImage']], displaylogo=False)),
            ]),
            dbc.Row([
                
                make_hideable(
                        dbc.Col([ 
                            dbc.Label("Winsor:", id='reg-vs-col-winsor-label-'+self.name),
                            dbc.Tooltip("Excluded the highest and lowest y values from the plot. "
                                        "When you have some real outliers it can help to remove them"
                                        " from the plot so it is easier to see the overall pattern.", 
                                    target='reg-vs-col-winsor-label-'+self.name),
                            dbc.Input(id='reg-vs-col-winsor-'+self.name, 
                                    value=self.winsor,
                                type="number", min=0, max=49, step=1),
                        ], md=2), hide=self.hide_winsor),  
                make_hideable(
                        dbc.Col([
                            dbc.Label("Points:", id='reg-vs-col-show-points-label-'+self.name),
                            dbc.Tooltip("For categorical features, display a point cloud next to the violin plot.", 
                                    target='reg-vs-col-show-points-label-'+self.name),
                            dbc.FormGroup(
                            [
                                dbc.RadioButton(
                                    id='reg-vs-col-show-points-'+self.name, 
                                    className="form-check-input",
                                    checked=self.points),
                                dbc.Label("Show points",
                                        html_for='reg-vs-col-show-points-'+self.name,
                                        className="form-check-label"),
                            ], check=True),
                        ],  md=3), self.hide_points),
            ]),
        ])

    def register_callbacks(self, app):
        @app.callback(
            Output('reg-vs-col-graph-'+self.name, 'figure'),
            [Input('reg-vs-col-col-'+self.name, 'value'),
             Input('reg-vs-col-display-type-'+self.name, 'value'),
             Input('reg-vs-col-show-points-'+self.name, 'checked'),
             Input('reg-vs-col-winsor-'+self.name, 'value')],
        )
        def update_residuals_graph(col, display, points, winsor):
            if display == 'observed':
                return self.explainer.plot_y_vs_feature(
                        col, points=points, winsor=winsor, dropna=True)
            elif display == 'predicted':
                return self.explainer.plot_preds_vs_feature(
                        col, points=points, winsor=winsor, dropna=True)
            else:
                return self.explainer.plot_residuals_vs_feature(
                            col, residuals=display, points=points, 
                            winsor=winsor, dropna=True)

        @app.callback(
            Output('reg-vs-col-col-'+self.name, 'options'),
            [Input('reg-vs-col-group-cats-'+self.name, 'checked')])
        def update_dependence_shap_scatter_graph(cats):
            return [{'label': col, 'value': col} 
                for col in self.explainer.columns_ranked_by_shap(cats)]


class RegressionModelSummaryComponent(ExplainerComponent):
    def __init__(self, explainer, title="Model Summary", name=None, 
                    hide_title=False, round=3, **kwargs):
        """Show model summary statistics (RMSE, MAE, R2) component

        Args:
            explainer (Explainer): explainer object constructed with either
                        ClassifierExplainer() or RegressionExplainer()
            title (str, optional): Title of tab or page. Defaults to 
                        "Model Summary".
            name (str, optional): unique name to add to Component elements. 
                        If None then random uuid is generated to make sure 
                        it's unique. Defaults to None.
            hide_title (bool, optional): hide title
            round (int): rounding to perform to metric floats.
        """
        super().__init__(explainer, title, name)
        self.description = f"""
        In the table below you can find a number of regression performance 
        metrics that describe how well the model is able to predict 
        {self.explainer.target}.
        """
        self.register_dependencies(['preds', 'residuals'])

    def layout(self):
        metrics_dict = self.explainer.metrics_descriptions()
        metrics_df = (pd.DataFrame(self.explainer.metrics(), index=["Score"]).T
                        .rename_axis(index="metric").reset_index().round(self.round))
        metrics_table = dbc.Table.from_dataframe(metrics_df, striped=False, bordered=False, hover=False)      
        metrics_table, tooltips = get_dbc_tooltips(metrics_table, 
                                                    metrics_dict, 
                                                    "reg-model-summary-div-hover", 
                                                    self.name)
        return html.Div([
            dbc.Row([
                make_hideable(
                html.Div([
                    html.H3(self.title, id='reg-model-summary-title-'+self.name),
                    dbc.Tooltip(self.description, target='reg-model-summary-title-'+self.name),
                ]), hide=self.hide_title),
            ]),
            metrics_table,
            *tooltips
        ])
