__all__ = [
    'RegressionRandomIndexComponent',
    'RegressionPredictionSummaryComponent',
    'PredictedVsActualComponent',
    'ResidualsComponent',
    'RegressionVsColComponent',
    'RegressionModelSummaryComponent',
]

import numpy as np
import pandas as pd

import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html

from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from ..dashboard_methods import *


class RegressionRandomIndexComponent(ExplainerComponent):
    def __init__(self, explainer, title="Select Random Index", name=None,
                        subtitle="Select from list or pick at random",
                        hide_title=False,   hide_subtitle=False,
                        hide_index=False, hide_pred_slider=False,
                        hide_residual_slider=False, hide_pred_or_y=False,
                        hide_abs_residuals=False, hide_button=False,
                        index=None, pred_slider=None, y_slider=None,
                        residual_slider=None, abs_residual_slider=None,
                        pred_or_y="preds", abs_residuals=True, round=2,
                        description=None, **kwargs):
        """Select a random index subject to constraints component

        Args:
            explainer (Explainer): explainer object constructed with either
                        ClassifierExplainer() or RegressionExplainer()
            title (str, optional): Title of tab or page. Defaults to
                        "Select Random Index".
            name (str, optional): unique name to add to Component elements.
                        If None then random uuid is generated to make sure
                        it's unique. Defaults to None.
            subtitle (str): subtitle
            hide_title (bool, optional): hide title
            hide_subtitle (bool, optional): Hide subtitle. Defaults to False.
            hide_index (bool, optional): Hide index selector.
                        Defaults to False.
            hide_pred_slider (bool, optional): Hide prediction slider.
                        Defaults to False.
            hide_residual_slider (bool, optional): hide residuals slider.
                        Defaults to False.
            hide_pred_or_y (bool, optional): hide prediction or actual toggle.
                        Defaults to False.
            hide_abs_residuals (bool, optional): hide absolute residuals toggle.
                        Defaults to False.
            hide_button (bool, optional): hide button. Defaults to False.
            index ({str, int}, optional): Initial index to display.
                        Defaults to None.
            pred_slider ([lb, ub], optional): Initial values for prediction
                        values slider [lowerbound, upperbound]. Defaults to None.
            y_slider ([lb, ub], optional): Initial values for y slider
                        [lower bound, upper bound]. Defaults to None.
            residual_slider ([lb, ub], optional): Initial values for residual slider
                        [lower bound, upper bound]. Defaults to None.
            abs_residual_slider ([lb, ub], optional): Initial values for absolute
                        residuals slider [lower bound, upper bound]
                        Defaults to None.
            pred_or_y (str, {'preds', 'y'}, optional): Initial use predictions
                        or y slider. Defaults to "preds".
            abs_residuals (bool, optional): Initial use residuals or absolute
                        residuals. Defaults to True.
            round (int, optional): rounding used for slider spacing. Defaults to 2.
            description (str, optional): Tooltip to display when hover over
                component title. When None default text is shown. 
        """
        super().__init__(explainer, title, name)
        assert self.explainer.is_regression, \
            ("explainer is not a RegressionExplainer so the RegressionRandomIndexComponent "
             "will not work. Try using the ClassifierRandomIndexComponent instead.")

        self.index_name = 'random-index-reg-index-'+self.name

        if self.explainer.y_missing:
            self.hide_residual_slider = True
            self.hide_pred_or_y = True
            self.hide_abs_residuals = True
            self.pred_or_y = "preds"
            self.y_slider = [0.0, 1.0]
            self.residual_slider = [0.0, 1.0]
            self.abs_residual_slider = [0.0, 1.0]

        if self.pred_slider is None:
            self.pred_slider = [self.explainer.preds.min(), self.explainer.preds.max()]

        if not self.explainer.y_missing:
            if self.y_slider is None:
                self.y_slider = [self.explainer.y.min(), self.explainer.y.max()]

            if self.residual_slider is None:
                self.residual_slider = [self.explainer.residuals.min(), self.explainer.residuals.max()]

            if self.abs_residual_slider is None:
                self.abs_residual_slider = [self.explainer.abs_residuals.min(), self.explainer.abs_residuals.max()]

            assert (len(self.pred_slider)==2 and self.pred_slider[0]<=self.pred_slider[1]), \
                "pred_slider should be a list of a [lower_bound, upper_bound]!"

            assert (len(self.y_slider)==2 and self.y_slider[0]<=self.y_slider[1]), \
                "y_slider should be a list of a [lower_bound, upper_bound]!"

            assert (len(self.residual_slider)==2 and self.residual_slider[0]<=self.residual_slider[1]), \
                "residual_slider should be a list of a [lower_bound, upper_bound]!"

            assert (len(self.abs_residual_slider)==2 and self.abs_residual_slider[0]<=self.abs_residual_slider[1]), \
                "abs_residual_slider should be a list of a [lower_bound, upper_bound]!"

        self.y_slider = [float(y) for y in self.y_slider]
        self.pred_slider = [float(p) for p in self.pred_slider]
        self.residual_slider = [float(r) for r in self.residual_slider]
        self.abs_residual_slider = [float(a) for a in self.abs_residual_slider]

        assert self.pred_or_y in ['preds', 'y'], "pred_or_y should be in ['preds', 'y']!"

        if self.description is None: self.description = f"""
        You can select a {self.explainer.index_name} directly by choosing it 
        from the dropdown (if you start typing you can search inside the list),
        or hit the Random {self.explainer.index_name} button to randomly select
        a {self.explainer.index_name} that fits the constraints. For example
        you can select a {self.explainer.index_name} with a very high predicted
        {self.explainer.target}, or a very low observed {self.explainer.target},
        or a {self.explainer.index_name} whose predicted {self.explainer.target} 
        was very far off from the observed {self.explainer.target} and so had a 
        high (absolute) residual.
        """

    def layout(self):
        return dbc.Card([
            make_hideable(
                dbc.CardHeader([
                    html.Div([
                        html.H3(f"Select {self.explainer.index_name}", id='random-index-reg-title-'+self.name),
                        make_hideable(html.H6(self.subtitle, className='card-subtitle'), hide=self.hide_subtitle),
                        dbc.Tooltip(self.description, target='random-index-reg-title-'+self.name),
                    ]), 
                ]), hide=self.hide_title),
            dbc.CardBody([
                dbc.Row([
                    make_hideable(
                        dbc.Col([
                                dcc.Dropdown(id='random-index-reg-index-'+self.name,
                                        options = [{'label': str(idx), 'value':idx}
                                                        for idx in self.explainer.idxs],
                                        value=self.index)
                            ], md=8), hide=self.hide_index),
                    make_hideable(
                        dbc.Col([
                            dbc.Button(f"Random {self.explainer.index_name}", color="primary", id='random-index-reg-button-'+self.name, block=True),
                            dbc.Tooltip(f"Select a random {self.explainer.index_name} according to the constraints",  
                                            target='random-index-reg-button-'+self.name),
                        ], md=4), hide=self.hide_button),
                    ], form=True),
                    dbc.Row([
                        make_hideable(
                        dbc.Col([
                            html.Div([
                                dbc.Label("Predicted range:", id='random-index-reg-pred-slider-label-'+self.name,
                                    html_for='random-index-reg-pred-slider-'+self.name),
                                dbc.Tooltip(f"Only select {self.explainer.index_name} where the "
                                            f"predicted {self.explainer.target} was within the following range:", 
                                            target='random-index-reg-pred-slider-label-'+self.name),
                                dcc.RangeSlider(
                                    id='random-index-reg-pred-slider-'+self.name,
                                    min=float(self.explainer.preds.min()),
                                    max=float(self.explainer.preds.max()),
                                    step=np.float_power(10, -self.round),
                                    value=[self.pred_slider[0], self.pred_slider[1]],
                                    marks={float(self.explainer.preds.min()): str(np.round(self.explainer.preds.min(), self.round)),
                                          float(self.explainer.preds.max()): str(np.round(self.explainer.preds.max(), self.round))},
                                    allowCross=False,
                                    tooltip = {'always_visible' : False}
                                )
                            ], id='random-index-reg-pred-slider-div-'+self.name),
                            html.Div([
                                dbc.Label("Observed range:", id='random-index-reg-y-slider-label-'+self.name,
                                    html_for='random-index-reg-y-slider-'+self.name),
                                dbc.Tooltip(f"Only select {self.explainer.index_name} where the "
                                            f"observed {self.explainer.target} was within the following range:", 
                                            target='random-index-reg-y-slider-label-'+self.name),
                                dcc.RangeSlider(
                                    id='random-index-reg-y-slider-'+self.name,
                                    min=float(self.explainer.y.min()),
                                    max=float(self.explainer.y.max()),
                                    step=np.float_power(10, -self.round),
                                    value=[self.y_slider[0], self.y_slider[1]],
                                    marks={float(self.explainer.y.min()): str(np.round(self.explainer.y.min(), self.round)),
                                           float(self.explainer.y.max()): str(np.round(self.explainer.y.max(), self.round))},
                                    allowCross=False,
                                    tooltip = {'always_visible' : False}
                                )
                            ], id='random-index-reg-y-slider-div-'+self.name),
                        ], md=8), hide=self.hide_pred_slider),
                        make_hideable(
                            dbc.Col([
                                dbc.Label("Range:", id='random-index-reg-preds-or-y-label-'+self.name, html_for='random-index-reg-preds-or-y-'+self.name),
                                dbc.Select(
                                    id='random-index-reg-preds-or-y-'+self.name,
                                    options=[
                                        {'label': 'Predicted', 'value': 'preds'},
                                        {'label': 'Observed', 'value': 'y'},
                                    ],
                                    value=self.pred_or_y),
                                dbc.Tooltip(f"You can either only select a random {self.explainer.index_name}"
                                            f"from within a certain range of observed {self.explainer.target} or"
                                            f"from within a certain range of predicted {self.explainer.target}.", 
                                    target='random-index-reg-preds-or-y-label-'+self.name)
                            ], md=4), hide=self.hide_pred_or_y),
                    ]),
                    dbc.Row([
                        make_hideable(
                            dbc.Col([
                                html.Div([
                                    dbc.Label("Residuals range:", id='random-index-reg-residual-slider-label-'+self.name,
                                        html_for='random-index-reg-residual-slider-'+self.name),
                                    dbc.Tooltip(f"Only select {self.explainer.index_name} where the "
                                                f"residual (difference between observed {self.explainer.target} and predicted {self.explainer.target})"
                                                " was within the following range:", 
                                        target='random-index-reg-residual-slider-label-'+self.name),
                                    dcc.RangeSlider(
                                        id='random-index-reg-residual-slider-'+self.name,
                                        min=float(self.explainer.residuals.min()),
                                        max=float(self.explainer.residuals.max()),
                                        step=np.float_power(10, -self.round),
                                        value=[self.residual_slider[0], self.residual_slider[1]],
                                        marks={float(self.explainer.residuals.min()): str(np.round(self.explainer.residuals.min(), self.round)),
                                            float(self.explainer.residuals.max()): str(np.round(self.explainer.residuals.max(), self.round))},
                                        allowCross=False,
                                        tooltip={'always_visible' : False}
                                    )
                                ], id='random-index-reg-residual-slider-div-'+self.name),
                                html.Div([
                                    dbc.Label("Absolute residuals", id='random-index-reg-abs-residual-slider-label'+self.name,
                                        html_for='random-index-reg-abs-residual-slider-'+self.name),
                                    dbc.Tooltip(f"Only select {self.explainer.index_name} where the absolute "
                                                f"residual (difference between observed {self.explainer.target} and predicted {self.explainer.target})"
                                                " was within the following range:", 
                                        target='random-index-reg-abs-residual-slider-label'+self.name),
                                    dcc.RangeSlider(
                                        id='random-index-reg-abs-residual-slider-'+self.name,
                                        min=float(self.explainer.abs_residuals.min()),
                                        max=float(self.explainer.abs_residuals.max()),
                                        step=np.float_power(10, -self.round),
                                        value=[self.abs_residual_slider[0], self.abs_residual_slider[1]],
                                        marks={float(self.explainer.abs_residuals.min()): str(np.round(self.explainer.abs_residuals.min(), self.round)),
                                            float(self.explainer.abs_residuals.max()): str(np.round(self.explainer.abs_residuals.max(), self.round))},
                                        allowCross=False,
                                        tooltip={'always_visible' : False}
                                    )
                                ], id='random-index-reg-abs-residual-slider-div-'+self.name),
                            ], md=8), hide=self.hide_residual_slider),
                            make_hideable(
                                dbc.Col([
                                    dbc.Label("Residuals:", id='random-index-reg-abs-residual-label-'+self.name,
                                                html_for='random-index-reg-abs-residual-'+self.name),
                                    dbc.Select(
                                        id='random-index-reg-abs-residual-'+self.name,
                                        options=[
                                            {'label': 'Residuals', 'value': 'relative'},
                                            {'label': 'Absolute Residuals', 'value': 'absolute'},
                                        ],
                                        value='absolute' if self.abs_residuals else 'relative'),
                                    dbc.Tooltip(f"You can either only select random a {self.explainer.index_name} "
                                                f"from within a certain range of residuals "
                                                f"(difference between observed and predicted {self.explainer.target}), "
                                                f"so for example only {self.explainer.index_name} for whom the prediction "
                                                f"was too high or too low."
                                                f"Or you can select only from a certain absolute residual range. So for "
                                                f"example only select {self.explainer.index_name} for which the prediction was at "
                                                f"least a certain amount of {self.explainer.units} off.",
                                                target='random-index-reg-abs-residual-label-'+self.name),
                                ], md=4), hide=self.hide_abs_residuals),
                    ]),
                    # make_hideable(
                    #     html.Div([
                    #     html.Div([
                    #         dbc.Row([
                    #             dbc.Col([
                    #                 html.Div([
                    #                     dbc.Label("Residuals range:", id='random-index-reg-residual-slider-label-'+self.name,
                    #                         html_for='random-index-reg-residual-slider-'+self.name),
                    #                     dbc.Tooltip(f"Only select {self.explainer.index_name} where the "
                    #                                 f"residual (difference between observed {self.explainer.target} and predicted {self.explainer.target})"
                    #                                 " was within the following range:", 
                    #                         target='random-index-reg-residual-slider-label-'+self.name),
                    #                     dcc.RangeSlider(
                    #                         id='random-index-reg-residual-slider-'+self.name,
                    #                         min=float(self.explainer.residuals.min()),
                    #                         max=float(self.explainer.residuals.max()),
                    #                         step=np.float_power(10, -self.round),
                    #                         value=[self.residual_slider[0], self.residual_slider[1]],
                    #                         marks={float(self.explainer.residuals.min()): str(np.round(self.explainer.residuals.min(), self.round)),
                    #                             float(self.explainer.residuals.max()): str(np.round(self.explainer.residuals.max(), self.round))},
                    #                         allowCross=False,
                    #                         tooltip={'always_visible' : False}
                    #                     )
                    #                 ], style={'margin-bottom':0})
                    #             ], md=8)
                    #         ]),
                    #     ], id='random-index-reg-residual-slider-div-'+self.name),
                    #     html.Div([
                    #         dbc.Row([
                    #             dbc.Col([
                    #                 html.Div([
                    #                     dbc.Label("Absolute residuals", id='random-index-reg-abs-residual-slider-label'+self.name,
                    #                         html_for='random-index-reg-abs-residual-slider-'+self.name),
                    #                     dbc.Tooltip(f"Only select {self.explainer.index_name} where the absolute "
                    #                                 f"residual (difference between observed {self.explainer.target} and predicted {self.explainer.target})"
                    #                                 " was within the following range:", 
                    #                         target='random-index-reg-abs-residual-slider-label'+self.name),
                    #                     dcc.RangeSlider(
                    #                         id='random-index-reg-abs-residual-slider-'+self.name,
                    #                         min=float(self.explainer.abs_residuals.min()),
                    #                         max=float(self.explainer.abs_residuals.max()),
                    #                         step=np.float_power(10, -self.round),
                    #                         value=[self.abs_residual_slider[0], self.abs_residual_slider[1]],
                    #                         marks={float(self.explainer.abs_residuals.min()): str(np.round(self.explainer.abs_residuals.min(), self.round)),
                    #                             float(self.explainer.abs_residuals.max()): str(np.round(self.explainer.abs_residuals.max(), self.round))},
                    #                         allowCross=False,
                    #                         tooltip={'always_visible' : False}
                    #                     )
                    #                 ], style={'margin-bottom':0})
                    #             ], md=8)
                    #         ])
                    #     ], id='random-index-reg-abs-residual-slider-div-'+self.name),
                    # ]), hide=self.hide_residual_slider),
                    # dbc.Row([
                    #     make_hideable(
                    #         dbc.Col([
                    #             dbc.Label("Residuals:", id='random-index-reg-abs-residual-label-'+self.name,
                    #                         html_for='random-index-reg-abs-residual-'+self.name),
                    #             dbc.Select(
                    #                 id='random-index-reg-abs-residual-'+self.name,
                    #                 options=[
                    #                     {'label': 'Residuals', 'value': 'relative'},
                    #                     {'label': 'Absolute Residuals', 'value': 'absolute'},
                    #                 ],
                    #                 value='absolute' if self.abs_residuals else 'relative'),
                    #             dbc.Tooltip(f"You can either only select random a {self.explainer.index_name} "
                    #                         f"from within a certain range of residuals "
                    #                         f"(difference between observed and predicted {self.explainer.target}), "
                    #                         f"so for example only {self.explainer.index_name} for whom the prediction "
                    #                         f"was too high or too low."
                    #                         f"Or you can select only from a certain absolute residual range. So for "
                    #                         f"example only select {self.explainer.index_name} for which the prediction was at "
                    #                         f"least a certain amount of {self.explainer.units} off.",
                    #             target='random-index-reg-abs-residual-label-'+self.name),
                    #         ], md=4), hide=self.hide_pred_or_y),
                    #     make_hideable(
                    #         dbc.Col([
                    #             html.Div([
                    #                 dbc.Select(
                    #                     id='random-index-reg-abs-residual-'+self.name,
                    #                     options=[
                    #                         {'label': 'Use Residuals', 'value': 'relative'},
                    #                         {'label': 'Use Absolute Residuals', 'value': 'absolute'},
                    #                     ],
                    #                     value='absolute' if self.abs_residuals else 'relative'),
                    #             ], id='random-index-reg-abs-residual-div-'+self.name),
                    #             dbc.Tooltip(f"You can either only select random a {self.explainer.index_name} "
                    #                         f"from within a certain range of residuals "
                    #                         f"(difference between observed and predicted {self.explainer.target}), "
                    #                         f"so for example only {self.explainer.index_name} for whom the prediction "
                    #                         f"was too high or too low."
                    #                         f"Or you can select only from a certain absolute residual range. So for "
                    #                         f"example only select {self.explainer.index_name} for which the prediction was at "
                    #                         f"least a certain amount of {self.explainer.units} off.",
                    #             target='random-index-reg-abs-residual-div-'+self.name),

                    #         ], md=4), hide=self.hide_abs_residuals),
                    ]),
            ])

    def component_callbacks(self, app):
        @app.callback(
            [Output('random-index-reg-pred-slider-div-'+self.name, 'style'),
             Output('random-index-reg-y-slider-div-'+self.name, 'style')],
            [Input('random-index-reg-preds-or-y-'+self.name, 'value')])
        def update_reg_hidden_div_pred_sliders(preds_or_y):
            if preds_or_y == 'preds':
                return (None, dict(display="none"))
            elif preds_or_y == 'y':
                return (dict(display="none"), None)
            raise PreventUpdate

        @app.callback(
            [Output('random-index-reg-residual-slider-div-'+self.name, 'style'),
             Output('random-index-reg-abs-residual-slider-div-'+self.name, 'style')],
            [Input('random-index-reg-abs-residual-'+self.name, 'value')])
        def update_reg_hidden_div_pred_sliders(abs_residuals):
            if abs_residuals == 'absolute':
                return (dict(display="none"), None)
            else:
                return (None, dict(display="none"))
            raise PreventUpdate

        @app.callback(
            [Output('random-index-reg-residual-slider-'+self.name, 'min'),
             Output('random-index-reg-residual-slider-'+self.name, 'max'),
             Output('random-index-reg-residual-slider-'+self.name, 'value'),
             Output('random-index-reg-residual-slider-'+self.name, 'marks'),
             Output('random-index-reg-abs-residual-slider-'+self.name, 'min'),
             Output('random-index-reg-abs-residual-slider-'+self.name, 'max'),
             Output('random-index-reg-abs-residual-slider-'+self.name, 'value'),
             Output('random-index-reg-abs-residual-slider-'+self.name, 'marks'),],
            [Input('random-index-reg-pred-slider-'+self.name, 'value'),
             Input('random-index-reg-y-slider-'+self.name, 'value')],
            [State('random-index-reg-preds-or-y-'+self.name, 'value'),
             State('random-index-reg-residual-slider-'+self.name, 'value'),
             State('random-index-reg-abs-residual-slider-'+self.name, 'value')])
        def update_residual_slider_limits(pred_range, y_range, preds_or_y, residuals_range, abs_residuals_range):
            if preds_or_y=='preds':
                min_residuals = self.explainer.residuals[(self.explainer.preds >= pred_range[0]) & (self.explainer.preds <= pred_range[1])].min()
                max_residuals = self.explainer.residuals[(self.explainer.preds >= pred_range[0]) & (self.explainer.preds <= pred_range[1])].max()
                min_abs_residuals = self.explainer.abs_residuals[(self.explainer.preds >= pred_range[0]) & (self.explainer.preds <= pred_range[1])].min()
                max_abs_residuals = self.explainer.abs_residuals[(self.explainer.preds >= pred_range[0]) & (self.explainer.preds <= pred_range[1])].max()
            elif preds_or_y=='y':
                min_residuals = self.explainer.residuals[(self.explainer.y >= y_range[0]) & (self.explainer.y <= y_range[1])].min()
                max_residuals = self.explainer.residuals[(self.explainer.y >= y_range[0]) & (self.explainer.y <= y_range[1])].max()
                min_abs_residuals = self.explainer.abs_residuals[(self.explainer.y >= y_range[0]) & (self.explainer.y <= y_range[1])].min()
                max_abs_residuals = self.explainer.abs_residuals[(self.explainer.y >= y_range[0]) & (self.explainer.y <= y_range[1])].max()

            new_residuals_range = [max(min_residuals, residuals_range[0]), min(max_residuals, residuals_range[1])]
            new_abs_residuals_range = [max(min_abs_residuals, abs_residuals_range[0]), min(max_abs_residuals, abs_residuals_range[1])]
            residuals_marks = {min_residuals: str(np.round(min_residuals, self.round)),
                                max_residuals: str(np.round(max_residuals, self.round))}
            abs_residuals_marks = {min_abs_residuals: str(np.round(min_abs_residuals, self.round)),
                                    max_abs_residuals: str(np.round(max_abs_residuals, self.round))}
            return (min_residuals, max_residuals, new_residuals_range, residuals_marks,
                    min_abs_residuals, max_abs_residuals, new_abs_residuals_range, abs_residuals_marks)

        @app.callback(
            Output('random-index-reg-index-'+self.name, 'value'),
            [Input('random-index-reg-button-'+self.name, 'n_clicks')],
            [State('random-index-reg-pred-slider-'+self.name, 'value'),
             State('random-index-reg-y-slider-'+self.name, 'value'),
             State('random-index-reg-residual-slider-'+self.name, 'value'),
             State('random-index-reg-abs-residual-slider-'+self.name, 'value'),
             State('random-index-reg-preds-or-y-'+self.name, 'value'),
             State('random-index-reg-abs-residual-'+self.name, 'value')])
        def update_index(n_clicks, pred_range, y_range, residual_range, abs_residuals_range, preds_or_y, abs_residuals):
            if n_clicks is None and self.index is not None:
                raise PreventUpdate
            if preds_or_y == 'preds':
                if abs_residuals=='absolute':
                    return self.explainer.random_index(
                                pred_min=pred_range[0], pred_max=pred_range[1],
                                abs_residuals_min=abs_residuals_range[0],
                                abs_residuals_max=abs_residuals_range[1],
                                return_str=True)
                else:
                    return self.explainer.random_index(
                                pred_min=pred_range[0], pred_max=pred_range[1],
                                residuals_min=residual_range[0],
                                residuals_max=residual_range[1],
                                return_str=True)
            elif preds_or_y == 'y':
                if abs_residuals=='absolute':
                    return self.explainer.random_index(
                                y_min=y_range[0], y_max=y_range[1],
                                abs_residuals_min=abs_residuals_range[0],
                                abs_residuals_max=abs_residuals_range[1],
                                return_str=True)
                else:
                    return self.explainer.random_index(
                                y_min=pred_range[0], y_max=pred_range[1],
                                residuals_min=residual_range[0],
                                residuals_max=residual_range[1],
                                return_str=True)


class RegressionPredictionSummaryComponent(ExplainerComponent):
    def __init__(self, explainer, title="Prediction", name=None,
                    hide_index=False, hide_title=False, 
                    hide_subtitle=False, hide_table=False,
                    feature_input_component=None,
                    index=None,  round=3, description=None,
                    **kwargs):
        """Shows a summary for a particular prediction

        Args:
            explainer (Explainer): explainer object constructed with either
                        ClassifierExplainer() or RegressionExplainer()
            title (str, optional): Title of tab or page. Defaults to 
                        "Prediction".
            name (str, optional): unique name to add to Component elements. 
                        If None then random uuid is generated to make sure 
                        it's unique. Defaults to None.
            hide_index (bool, optional): hide index selector. Defaults to False.
            hide_title (bool, optional): hide title. Defaults to False.
            hide_subtitle (bool, optional): Hide subtitle. Defaults to False.
            hide_table (bool, optional): hide the results table
            feature_input_component (FeatureInputComponent): A FeatureInputComponent
                that will give the input to the graph instead of the index selector.
                If not None, hide_index=True. Defaults to None.
            index ({int, str}, optional): Index to display prediction summary for. Defaults to None.
            description (str, optional): Tooltip to display when hover over
                component title. When None default text is shown. 
        """
        super().__init__(explainer, title, name)

        self.index_name = 'reg-prediction-index-'+self.name

        if self.feature_input_component is not None:
            self.hide_index = True

        if self.description is None: self.description = f"""
        Shows the predicted {self.explainer.target} and the observed {self.explainer.target},
        as well as the difference between the two (residual)
        """

    def layout(self):
        return dbc.Card([
            make_hideable(
                dbc.CardHeader([
                    html.H3(self.title, id='reg-prediction-title-'+self.name, className='card-title'),
                    dbc.Tooltip(self.description, target='reg-prediction-title-'+self.name),
                ]), hide=self.hide_title), 
            dbc.CardBody([
                dbc.Row([
                    make_hideable(
                        dbc.Col([
                            dbc.Label(f"{self.explainer.index_name}:"),
                            dcc.Dropdown(id='reg-prediction-index-'+self.name, 
                                    options = [{'label': str(idx), 'value':idx} 
                                                    for idx in self.explainer.idxs],
                                    value=self.index)
                        ], md=6), hide=self.hide_index),          
                ]),
                dbc.Row([
                    dbc.Col([
                        html.Div(id='reg-prediction-div-'+self.name) 
                    ])
                ])
            ])
        ])

    def component_callbacks(self, app):
        if self.feature_input_component is None:
            @app.callback(
                Output('reg-prediction-div-'+self.name, 'children'),
                [Input('reg-prediction-index-'+self.name, 'value')])
            def update_output_div(index):
                if index is not None:
                    preds_df = self.explainer.prediction_result_df(index, round=self.round)
                    return make_hideable(
                        dbc.Table.from_dataframe(preds_df, striped=False, bordered=False, hover=False),
                        hide=self.hide_table)  
                raise PreventUpdate
        else:
            @app.callback(
                Output('reg-prediction-div-'+self.name, 'children'),
                [*self.feature_input_component._feature_callback_inputs])
            def update_output_div(*inputs):
                X_row = self.explainer.get_row_from_input(inputs, ranked_by_shap=True)   
                preds_df = self.explainer.prediction_result_df(X_row=X_row, round=self.round)
                return make_hideable(
                    dbc.Table.from_dataframe(preds_df, striped=False, bordered=False, hover=False),
                    hide=self.hide_table)  


class PredictedVsActualComponent(ExplainerComponent):
    def __init__(self, explainer, title="Predicted vs Actual", name=None,
                    subtitle="How close is the predicted value to the observed?",
                    hide_title=False, hide_subtitle=False, 
                    hide_log_x=False, hide_log_y=False,
                    logs=False, log_x=False, log_y=False, description=None,
                    **kwargs):
        """Shows a plot of predictions vs y.

        Args:
            explainer (Explainer): explainer object constructed with either
                        ClassifierExplainer() or RegressionExplainer()
            title (str, optional): Title of tab or page. Defaults to 
                        "Predicted vs Actual".
            name (str, optional): unique name to add to Component elements. 
                        If None then random uuid is generated to make sure 
                        it's unique. Defaults to None.
            subtitle (str): subtitle
            hide_title (bool, optional) Hide the title. Defaults to False.
            hide_subtitle (bool, optional): Hide subtitle. Defaults to False.
            hide_log_x (bool, optional): Hide the log_x toggle. Defaults to False.
            hide_log_y (bool, optional): Hide the log_y toggle. Defaults to False.
            logs (bool, optional): Whether to use log axis. Defaults to False.
            log_x (bool, optional): log only x axis. Defaults to False.
            log_y (bool, optional): log only y axis. Defaults to False.
            description (str, optional): Tooltip to display when hover over
                component title. When None default text is shown. 
        """
        super().__init__(explainer, title, name)
        
        self.logs, self.log_x, self.log_y = logs, log_x, log_y

        if self.description is None: self.description = f"""
        Plot shows the observed {self.explainer.target} and the predicted 
        {self.explainer.target} in the same plot. A perfect model would have
        all the points on the diagonal (predicted matches observed). The further
        away point are from the diagonal the worse the model is in predicting
        {self.explainer.target}.
        """
        self.register_dependencies(['preds'])

    def layout(self):
        return dbc.Card([
            make_hideable(
                dbc.CardHeader([
                    html.Div([
                        html.H3(self.title, id='pred-vs-actual-title-'+self.name),
                        make_hideable(html.H6(self.subtitle, className='card-subtitle'), hide=self.hide_subtitle),
                        dbc.Tooltip(self.description, target='pred-vs-actual-title-'+self.name),
                    ]), 
                ]), hide=self.hide_title),
            dbc.CardBody([
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
                ], justify="center"),
            ]), 
        ])

    def component_callbacks(self, app):
        @app.callback(
            Output('pred-vs-actual-graph-'+self.name, 'figure'),
            [Input('pred-vs-actual-logx-'+self.name, 'checked'),
             Input('pred-vs-actual-logy-'+self.name, 'checked')],
        )
        def update_predicted_vs_actual_graph(log_x, log_y):
            return self.explainer.plot_predicted_vs_actual(log_x=log_x, log_y=log_y)

class ResidualsComponent(ExplainerComponent):
    def __init__(self, explainer, title="Residuals", name=None,
                    subtitle="How much is the model off?",
                    hide_title=False, hide_subtitle=False, hide_footer=False,
                    hide_pred_or_actual=False, hide_ratio=False,
                    pred_or_actual="vs_pred", residuals="difference",
                    description=None, **kwargs):
        """Residuals plot component

        Args:
            explainer (Explainer): explainer object constructed with either
                        ClassifierExplainer() or RegressionExplainer()
            title (str, optional): Title of tab or page. Defaults to 
                        "Residuals".
            name (str, optional): unique name to add to Component elements. 
                        If None then random uuid is generated to make sure 
                        it's unique. Defaults to None.
            subtitle (str): subtitle
            hide_title (bool, optional) Hide the title. Defaults to False.
            hide_subtitle (bool, optional): Hide subtitle. Defaults to False.
            hide_footer (bool, optional): hide the footer at the bottom of the component
            hide_pred_or_actual (bool, optional): hide vs predictions or vs 
                        actual for x-axis toggle. Defaults to False.
            hide_ratio (bool, optional): hide residual type dropdown. Defaults to False.
            pred_or_actual (str, {'vs_actual', 'vs_pred'}, optional): Whether 
                        to plot actual or predictions on the x-axis. 
                        Defaults to "vs_pred".
            residuals (str, {'difference', 'ratio', 'log-ratio'} optional): 
                    How to calcualte residuals. Defaults to 'difference'.
            description (str, optional): Tooltip to display when hover over
                component title. When None default text is shown. 
        """
        super().__init__(explainer, title, name)

        assert residuals in ['difference', 'ratio', 'log-ratio'], \
            ("parameter residuals should in ['difference', 'ratio', 'log-ratio']"
             f" but you passed residuals={residuals}")
        if self.description is None: self.description = f"""
        The residuals are the difference between the observed {self.explainer.target}
        and predicted {self.explainer.target}. In this plot you can check if 
        the residuals are higher or lower for higher/lower actual/predicted outcomes. 
        So you can check if the model works better or worse for different {self.explainer.target}
        levels.
        """
        self.register_dependencies(['preds', 'residuals'])

    def layout(self):
        return dbc.Card([
            make_hideable(
                dbc.CardHeader([
                    html.Div([
                        html.H3(self.title, id='residuals-title-'+self.name),
                        make_hideable(html.H6(self.subtitle, className='card-subtitle'), hide=self.hide_subtitle),
                        dbc.Tooltip(self.description, target='residuals-title-'+self.name),
                    ]), 
                ]), hide=self.hide_title),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        dcc.Graph(id='residuals-graph-'+self.name,
                                config=dict(modeBarButtons=[['toImage']], displaylogo=False)),
                    ])
                ])
            ]),
            make_hideable(
            dbc.CardFooter([
                dbc.Row([
                    make_hideable(
                        dbc.Col([
                            dbc.FormGroup(
                            [
                                dbc.Label("Horizontal axis:", html_for='residuals-pred-or-actual-'+self.name),
                                dbc.Select(
                                    options=[
                                        {"label": "Predicted", "value": "vs_pred"},
                                        {"label": "Observed", "value": "vs_actual"},
                                    ],
                                    value=self.pred_or_actual,
                                    id='residuals-pred-or-actual-'+self.name,
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
                            dbc.Select(id='residuals-type-'+self.name,
                                    options = [{'label': 'Difference', 'value': 'difference'},
                                                {'label': 'Ratio', 'value': 'ratio'},
                                                {'label': 'Log ratio', 'value': 'log-ratio'}],
                                    value=self.residuals),
                        ], md=3), hide=self.hide_ratio),
                ]),
            ]), hide=self.hide_footer)
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
                    subtitle="Are predictions and residuals correlated with features?",
                    hide_title=False, hide_subtitle=False, hide_footer=False,
                    hide_col=False, hide_ratio=False, hide_cats=False, 
                    hide_points=False, hide_winsor=False,
                    col=None, display='difference', cats=True, 
                    points=True, winsor=0, description=None, **kwargs):
        """Show residuals, observed or preds vs a particular Feature component

        Args:
            explainer (Explainer): explainer object constructed with either
                        ClassifierExplainer() or RegressionExplainer()
            title (str, optional): Title of tab or page. Defaults to 
                        "Plot vs feature".
            name (str, optional): unique name to add to Component elements. 
                        If None then random uuid is generated to make sure 
                        it's unique. Defaults to None.
            subtitle (str): subtitle
            hide_title (bool, optional) Hide the title. Defaults to False.
            hide_subtitle (bool, optional): Hide subtitle. Defaults to False.
            hide_footer (bool, optional): hide the footer at the bottom of the component
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
            description (str, optional): Tooltip to display when hover over
                component title. When None default text is shown. 
        """
        super().__init__(explainer, title, name)

        if self.col is None:
            self.col = self.explainer.columns_ranked_by_shap(self.cats)[0]
        
        assert self.display in {'observed', 'predicted', 'difference', 'ratio', 'log-ratio'}, \
            ("parameter display should in {'observed', 'predicted', 'difference', 'ratio', 'log-ratio'}"
             f" but you passed display={self.display}!")

        if self.description is None: self.description = f"""
        This plot shows either residuals (difference between observed {self.explainer.target}
        and predicted {self.explainer.target}) plotted against the values of different features,
        or the observed or predicted {self.explainer.target}.
        This allows you to inspect whether the model is more wrong for particular
        range of feature values than others. 
        """
        self.register_dependencies(['preds', 'residuals'])

    def layout(self):
        return dbc.Card([
            make_hideable(
                dbc.CardHeader([
                    html.Div([
                        html.H3(self.title, id='reg-vs-col-title-'+self.name),
                        make_hideable(html.H6(self.subtitle, className='card-subtitle'), hide=self.hide_subtitle),
                        dbc.Tooltip(self.description, target='reg-vs-col-title-'+self.name),
                    ]), 
                ]), hide=self.hide_title),
            dbc.CardBody([
                dbc.Row([
                    make_hideable(
                        dbc.Col([
                            dbc.Label("Feature:", id='reg-vs-col-col-label-'+self.name),
                            dbc.Tooltip("Select the feature to display on the x-axis.", 
                                        target='reg-vs-col-col-label-'+self.name),
                            dbc.Select(id='reg-vs-col-col-'+self.name,
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
                                dbc.Select(id='reg-vs-col-display-type-'+self.name,
                                        options = [{'label': 'Observed', 'value': 'observed'},
                                                    {'label': 'Predicted', 'value': 'predicted'},
                                                    {'label': 'Residuals: Difference', 'value': 'difference'},
                                                    {'label': 'Residuals: Ratio', 'value': 'ratio'},
                                                    {'label': 'Residuals: Log ratio', 'value': 'log-ratio'}],
                                        value=self.display),
                            ], md=4), hide=self.hide_ratio),
                    make_hideable(
                            dbc.Col([
                                dbc.FormGroup([
                                    dbc.Label("Grouping:", id='reg-vs-col-group-cats-label-'+self.name),
                                    dbc.Tooltip("Group onehot encoded categorical variables together", 
                                                target='reg-vs-col-group-cats-label-'+self.name),
                                    dbc.Checklist(
                                        options=[{"label": "Group cats", "value": True}],
                                        value=[True] if self.cats else [],
                                        id='reg-vs-col-group-cats-'+self.name,
                                        inline=True,
                                        switch=True,
                                    ),
                                ]),
                            ], md=2), self.hide_cats),
                ]),
                dbc.Row([
                    dbc.Col([
                        dcc.Graph(id='reg-vs-col-graph-'+self.name,
                                config=dict(modeBarButtons=[['toImage']], displaylogo=False)),
                    ]) 
                ]),
            ]),
            make_hideable(
            dbc.CardFooter([
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
                        ], md=4), hide=self.hide_winsor),  
                    make_hideable(
                        dbc.Col([
                            html.Div([
                                dbc.FormGroup([
                                        dbc.Label("Scatter:"),
                                        dbc.Tooltip("For categorical features, display "
                                                "a point cloud next to the violin plots.", 
                                                    target='reg-vs-col-show-points-'+self.name),
                                        dbc.Checklist(
                                            options=[{"label": "Show point cloud", "value": True}],
                                            value=[True] if self.points else [],
                                            id='reg-vs-col-show-points-'+self.name,
                                            inline=True,
                                            switch=True,
                                        ),
                                    ]),
                            ], id='reg-vs-col-show-points-div-'+self.name)
                        ],  md=4), self.hide_points),
                ])
            ]), hide=self.hide_footer)
        ])

    def register_callbacks(self, app):
        @app.callback(
            [Output('reg-vs-col-graph-'+self.name, 'figure'),
             Output('reg-vs-col-show-points-div-'+self.name, 'style')],
            [Input('reg-vs-col-col-'+self.name, 'value'),
             Input('reg-vs-col-display-type-'+self.name, 'value'),
             Input('reg-vs-col-show-points-'+self.name, 'value'),
             Input('reg-vs-col-winsor-'+self.name, 'value')],
        )
        def update_residuals_graph(col, display, points, winsor):
            style = {} if col in self.explainer.cats else dict(display="none")
            if display == 'observed':
                return self.explainer.plot_y_vs_feature(
                        col, points=bool(points), winsor=winsor, dropna=True), style
            elif display == 'predicted':
                return self.explainer.plot_preds_vs_feature(
                        col, points=bool(points), winsor=winsor, dropna=True), style
            else:
                return self.explainer.plot_residuals_vs_feature(
                            col, residuals=display, points=bool(points), 
                            winsor=winsor, dropna=True), style

        @app.callback(
            Output('reg-vs-col-col-'+self.name, 'options'),
            [Input('reg-vs-col-group-cats-'+self.name, 'value')])
        def update_dependence_shap_scatter_graph(cats):
            return [{'label': col, 'value': col} 
                for col in self.explainer.columns_ranked_by_shap(bool(cats))]


class RegressionModelSummaryComponent(ExplainerComponent):
    def __init__(self, explainer, title="Model Summary", name=None, 
                    subtitle="Quantitative metrics for model performance",
                    hide_title=False, hide_subtitle=False, 
                    round=3, description=None, **kwargs):
        """Show model summary statistics (RMSE, MAE, R2) component

        Args:
            explainer (Explainer): explainer object constructed with either
                        ClassifierExplainer() or RegressionExplainer()
            title (str, optional): Title of tab or page. Defaults to 
                        "Model Summary".
            name (str, optional): unique name to add to Component elements. 
                        If None then random uuid is generated to make sure 
                        it's unique. Defaults to None.
            subtitle (str): subtitle
            hide_title (bool, optional): hide title
            hide_subtitle (bool, optional): Hide subtitle. Defaults to False.
            round (int): rounding to perform to metric floats.
            description (str, optional): Tooltip to display when hover over
                component title. When None default text is shown. 
        """
        super().__init__(explainer, title, name)
        if self.description is None: self.description = f"""
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
        return dbc.Card([
            make_hideable(
                dbc.CardHeader([
                    html.Div([
                        html.H3(self.title, id='reg-model-summary-title-'+self.name),
                        make_hideable(html.H6(self.subtitle, className='card-subtitle'), hide=self.hide_subtitle),
                        dbc.Tooltip(self.description, target='reg-model-summary-title-'+self.name),
                    ]), 
                ]), hide=self.hide_title),
            dbc.CardBody([
                metrics_table,
                *tooltips
            ]),
        ])
