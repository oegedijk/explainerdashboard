__all__ = [
    'ClassifierRandomIndexComponent',
    'RegressionRandomIndexComponent',
    'CutoffPercentileComponent',
    'PosLabelConnector',
    'CutoffConnector',
    'IndexConnector',
    'HighlightConnector'
]

import numpy as np

import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html

from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from ..dashboard_methods import  *


class ClassifierRandomIndexComponent(ExplainerComponent):
    def __init__(self, explainer, title="Select Random Index", name=None,
                        hide_title=False, hide_index=False, hide_slider=False,
                        hide_labels=False, hide_pred_or_perc=False,
                        hide_selector=False, hide_button=False,
                        pos_label=None, index=None, slider= None, labels=None,
                        pred_or_perc='predictions', **kwargs):
        """Select a random index subject to constraints component

        Args:
            explainer (Explainer): explainer object constructed with either
                        ClassifierExplainer() or RegressionExplainer()
            title (str, optional): Title of tab or page. Defaults to
                        "Select Random Index".
            name (str, optional): unique name to add to Component elements.
                        If None then random uuid is generated to make sure
                        it's unique. Defaults to None.
            hide_title (bool, optional): Hide title. Defaults to False.
            hide_index (bool, optional): Hide index selector. Defaults to False.
            hide_slider (bool, optional): Hide prediction/percentile slider.
                        Defaults to False.
            hide_labels (bool, optional): Hide label selector Defaults to False.
            hide_pred_or_perc (bool, optional): Hide prediction/percentiles
                        toggle. Defaults to False.
            hide_selector (bool, optional): hide pos label selectors. Defaults to False.
            hide_button (bool, optional): Hide button. Defaults to False.
            pos_label ({int, str}, optional): initial pos label. Defaults to explainer.pos_label
            index ({str, int}, optional): Initial index to display.
                        Defaults to None.
            slider ([float,float], optional): initial slider position
                        [lower bound, upper bound]. Defaults to None.
            labels ([str], optional): list of initial labels(str) to include.
                        Defaults to None.
            pred_or_perc (str, optional): Whether to use prediction or
                        percentiles slider. Defaults to 'predictions'.
        """
        super().__init__(explainer, title, name)

        self.index_name = 'random-index-clas-index-'+self.name

        if self.slider is None:
            self.slider = [0.0, 1.0]

        if self.labels is None:
            self.labels = self.explainer.labels

        if self.explainer.y_missing:
            self.hide_labels = True

        self.selector = PosLabelSelector(explainer, name=self.name, pos_label=pos_label)

        assert (len(self.slider) == 2 and
                self.slider[0] >= 0 and self.slider[0] <=1 and
                self.slider[1] >= 0.0 and self.slider[1] <= 1.0 and
                self.slider[0] <= self.slider[1]), \
                    "slider should be e.g. [0.5, 1.0]"

        assert all([lab in self.explainer.labels for lab in self.labels]), \
            f"These labels are not in explainer.labels: {[lab for lab in self.labels if lab not in explainer.labels]}!"

        assert self.pred_or_perc in ['predictions', 'percentiles'], \
            "pred_or_perc should either be `predictions` or `percentiles`!"

        self.description = f"""
        You can select a {self.explainer.index_name} directly by choosing it 
        from the dropdown (if you start typing you can search inside the list),
        or hit the Random {self.explainer.index_name} button to randomly select
        a {self.explainer.index_name} that fits the constraints. For example
        you can select a {self.explainer.index_name} where the observed
        {self.explainer.target} is {self.explainer.labels[0]} but the
        predicted probability of {self.explainer.labels[1]} is very high. This
        allows you to for example sample only false positives or only false negatives.
        """

    def layout(self):
        return html.Div([
            dbc.Row([
                make_hideable(
                html.Div([
                    html.H3(f"Select {self.explainer.index_name}", id='random-index-clas-title-'+self.name),
                    dbc.Tooltip(self.description, target='random-index-clas-title-'+self.name),
                ]), hide=self.hide_title),
            ]),

            dbc.Row([
                make_hideable(
                    dbc.Col([
                        dcc.Dropdown(id='random-index-clas-index-'+self.name,
                                options = [{'label': str(idx), 'value':idx}
                                                for idx in self.explainer.idxs],
                                value=self.index),
                    ], md=8), hide=self.hide_index),
                make_hideable(
                    dbc.Col([
                        dbc.Button(f"Random {self.explainer.index_name}", color="primary", id='random-index-clas-button-'+self.name, block=True),
                        dbc.Tooltip(f"Select a random {self.explainer.index_name} according to the constraints",  
                                        target='random-index-clas-button-'+self.name),
                    ], md=4), hide=self.hide_button),
            ], form=True),
            dbc.Row([
                make_hideable(
                    dbc.Col([
                        dbc.Label(f"Observed {self.explainer.target}:", id='random-index-clas-labels-label-'+self.name),
                        dbc.Tooltip(f"Only select a random {self.explainer.index_name} where the observed "
                                    f"{self.explainer.target} is one of the selected labels:",
                                target='random-index-clas-labels-label-'+self.name),
                        dcc.Dropdown(
                            id='random-index-clas-labels-'+self.name,
                            options=[{'label': lab, 'value': lab} for lab in self.explainer.labels],
                            multi=True,
                            value=self.labels),
                    ], md=12), hide=self.hide_labels),
            ]),
            dbc.Row([
                make_hideable(
                    dbc.Col([
                        html.Div([
                            dbc.Label(id='random-index-clas-slider-label-'+self.name,
                                children="Predicted probability range:",
                                html_for='prediction-range-slider-'+self.name,),
                            dbc.Tooltip(f"Only select a random {self.explainer.index_name} where the "
                                         "predicted probability of positive label is in the following range:",
                                        id='random-index-clas-slider-label-tooltip-'+self.name,
                                        target='random-index-clas-slider-label-'+self.name),
                            dcc.RangeSlider(
                                id='random-index-clas-slider-'+self.name,
                                min=0.0, max=1.0, step=0.01,
                                value=self.slider,  allowCross=False,
                                marks={0.0:'0.0', 0.1:'0.1', 0.2:'0.2', 0.3:'0.3',
                                        0.4:'0.4', 0.5:'0.5', 0.6:'0.6', 0.7:'0.7',
                                        0.8:'0.8', 0.9:'0.9', 1.0:'1.0'},
                                tooltip = {'always_visible' : False})
                        ], style={'margin-bottom':25})
                    ], md=5), hide=self.hide_slider),
                make_hideable(
                    dbc.Col([
                        html.Div([
                        dbc.RadioItems(
                            id='random-index-clas-pred-or-perc-'+self.name,
                            options=[
                                {'label': 'Use predicted probability', 'value': 'predictions'},
                                {'label': 'Use predicted percentile', 'value': 'percentiles'},
                            ],
                            value=self.pred_or_perc,
                            inline=True)
                        ], id='random-index-clas-pred-or-perc-div-'+self.name),
                        dbc.Tooltip("Instead of selecting from a range of predicted probabilities "
                                    "you can also select from a range of predicted percentiles. "
                                    "For example if you set the slider to (0.9-1.0) you would"
                                    f" only sample random {self.explainer.index_name} from the top "
                                    "10% highest predicted probabilities.",
                                target='random-index-clas-pred-or-perc-div-'+self.name),
                    ], md=4, align="center"), hide=self.hide_pred_or_perc),
                make_hideable(
                    dbc.Col([
                        self.selector.layout()
                    ], md=3), hide=self.hide_selector),
            ], justify="start"),
        ])

    def _register_callbacks(self, app):
        @app.callback(
            Output('random-index-clas-index-'+self.name, 'value'),
            [Input('random-index-clas-button-'+self.name, 'n_clicks')],
            [State('random-index-clas-slider-'+self.name, 'value'),
             State('random-index-clas-labels-'+self.name, 'value'),
             State('random-index-clas-pred-or-perc-'+self.name, 'value'),
             State('pos-label-'+self.name, 'value')])
        def update_index(n_clicks, slider_range, labels, pred_or_perc, pos_label):
            if pred_or_perc == 'predictions':
                return self.explainer.random_index(y_values=labels,
                    pred_proba_min=slider_range[0], pred_proba_max=slider_range[1],
                    return_str=True, pos_label=pos_label)
            elif pred_or_perc == 'percentiles':
                return self.explainer.random_index(y_values=labels,
                    pred_percentile_min=slider_range[0], pred_percentile_max=slider_range[1],
                    return_str=True, pos_label=pos_label)

        @app.callback(
            [Output('random-index-clas-slider-label-'+self.name, 'children'),
             Output('random-index-clas-slider-label-tooltip-'+self.name, 'children')],
            [Input('random-index-clas-pred-or-perc-'+self.name, 'value'),
             Input('pos-label-'+self.name, 'value')]
        )
        def update_slider_label(pred_or_perc, pos_label):
            if pred_or_perc == 'predictions':
                return (
                    "Predicted probability range:",
                    f"Only select a random {self.explainer.index_name} where the "
                    f"predicted probability of {self.explainer.labels[pos_label]}"
                    " is in the following range:"
                )
            elif pred_or_perc == 'percentiles':
                return (
                    "Predicted percentile range:",
                    f"Only select a random {self.explainer.index_name} where the "
                    f"predicted probability of {self.explainer.labels[pos_label]}"
                    " is in the following percentile range. For example you can "
                    "only sample from the top 10% highest predicted probabilities."
                )
            raise PreventUpdate

class RegressionRandomIndexComponent(ExplainerComponent):
    def __init__(self, explainer, title="Select Random Index", name=None,
                        hide_title=False, hide_index=False, hide_pred_slider=False,
                        hide_residual_slider=False, hide_pred_or_y=False,
                        hide_abs_residuals=False, hide_button=False,
                        index=None, pred_slider=None, y_slider=None,
                        residual_slider=None, abs_residual_slider=None,
                        pred_or_y="preds", abs_residuals=True, round=2, **kwargs):
        """Select a random index subject to constraints component

        Args:
            explainer (Explainer): explainer object constructed with either
                        ClassifierExplainer() or RegressionExplainer()
            title (str, optional): Title of tab or page. Defaults to
                        "Select Random Index".
            name (str, optional): unique name to add to Component elements.
                        If None then random uuid is generated to make sure
                        it's unique. Defaults to None.
            hide_title (bool, optional): hide title
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
        """
        super().__init__(explainer, title, name)

        self.index_name = 'random-index-reg-index-'+self.name

        if self.explainer.y_missing:
            self.hide_residual_slider = True
            self.hide_pred_or_y = True
            self.hide_abs_residuals = True
            self.pred_or_y = "preds"
            self.y_slider = [0, 1]
            self.residual_slider = [0, 1]
            self.abs_residual_slider = [0, 1]

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

        assert self.pred_or_y in ['preds', 'y'], "pred_or_y should be in ['preds', 'y']!"

        self.description = f"""
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
        return html.Div([
            dbc.Row([
                make_hideable(
                html.Div([
                    html.H3(f"Select {self.explainer.index_name}", id='random-index-reg-title-'+self.name),
                    dbc.Tooltip(self.description, target='random-index-reg-title-'+self.name),
                ]), hide=self.hide_title),
            ]),
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
            make_hideable(
                html.Div([
                html.Div([
                    dbc.Row([
                            dbc.Col([
                                html.Div([
                                    dbc.Label("Predicted range:", id='random-index-reg-pred-slider-label-'+self.name,
                                        html_for='random-index-reg-pred-slider-'+self.name),
                                    dbc.Tooltip(f"Only select {self.explainer.index_name} where the "
                                                f"predicted {self.explainer.target} was within the following range:", 
                                                target='random-index-reg-pred-slider-label-'+self.name),
                                    dcc.RangeSlider(
                                        id='random-index-reg-pred-slider-'+self.name,
                                        min=self.explainer.preds.min(),
                                        max=self.explainer.preds.max(),
                                        step=np.float_power(10, -self.round),
                                        value=[self.pred_slider[0], self.pred_slider[1]],
                                        marks={self.explainer.preds.min(): str(np.round(self.explainer.preds.min(), self.round)),
                                            self.explainer.preds.max(): str(np.round(self.explainer.preds.max(), self.round))},
                                        allowCross=False,
                                        tooltip = {'always_visible' : False}
                                    )
                                ], style={'margin-bottom':0})
                            ])
                    ]),
                ], id='random-index-reg-pred-slider-div-'+self.name),
                html.Div([
                    dbc.Row([
                            dbc.Col([
                                html.Div([
                                    dbc.Label("Observed range:", id='random-index-reg-y-slider-label-'+self.name,
                                        html_for='random-index-reg-y-slider-'+self.name),
                                    dbc.Tooltip(f"Only select {self.explainer.index_name} where the "
                                                f"observed {self.explainer.target} was within the following range:", 
                                    target='random-index-reg-y-slider-label-'+self.name),
                                    dcc.RangeSlider(
                                        id='random-index-reg-y-slider-'+self.name,
                                        min=self.explainer.y.min(),
                                        max=self.explainer.y.max(),
                                        step=np.float_power(10, -self.round),
                                        value=[self.y_slider[0], self.y_slider[1]],
                                        marks={self.explainer.y.min(): str(np.round(self.explainer.y.min(), self.round)),
                                            self.explainer.y.max(): str(np.round(self.explainer.y.max(), self.round))},
                                        allowCross=False,
                                        tooltip = {'always_visible' : False}
                                    )
                                ], style={'margin-bottom':0})
                            ]),
                    ]),
                ], id='random-index-reg-y-slider-div-'+self.name),
                ]), hide=self.hide_pred_slider),
            make_hideable(
                html.Div([
                html.Div([
                    dbc.Row([
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
                                    min=self.explainer.residuals.min(),
                                    max=self.explainer.residuals.max(),
                                    step=np.float_power(10, -self.round),
                                    value=[self.residual_slider[0], self.residual_slider[1]],
                                    marks={self.explainer.residuals.min(): str(np.round(self.explainer.residuals.min(), self.round)),
                                        self.explainer.residuals.max(): str(np.round(self.explainer.residuals.max(), self.round))},
                                    allowCross=False,
                                    tooltip={'always_visible' : False}
                                )
                            ], style={'margin-bottom':0})
                        ])
                    ]),
                ], id='random-index-reg-residual-slider-div-'+self.name),
                html.Div([
                    dbc.Row([
                        dbc.Col([
                            html.Div([
                                dbc.Label("Absolute residuals", id='random-index-reg-abs-residual-slider-label'+self.name,
                                    html_for='random-index-reg-abs-residual-slider-'+self.name),
                                dbc.Tooltip(f"Only select {self.explainer.index_name} where the absolute "
                                            f"residual (difference between observed {self.explainer.target} and predicted {self.explainer.target})"
                                            " was within the following range:", 
                                    target='random-index-reg-abs-residual-slider-label'+self.name),
                                dcc.RangeSlider(
                                    id='random-index-reg-abs-residual-slider-'+self.name,
                                    min=self.explainer.abs_residuals.min(),
                                    max=self.explainer.abs_residuals.max(),
                                    step=np.float_power(10, -self.round),
                                    value=[self.abs_residual_slider[0], self.abs_residual_slider[1]],
                                    marks={self.explainer.abs_residuals.min(): str(np.round(self.explainer.abs_residuals.min(), self.round)),
                                        self.explainer.abs_residuals.max(): str(np.round(self.explainer.abs_residuals.max(), self.round))},
                                    allowCross=False,
                                    tooltip={'always_visible' : False}
                                )
                            ], style={'margin-bottom':0})
                        ])
                    ])
                ], id='random-index-reg-abs-residual-slider-div-'+self.name),
            ]), hide=self.hide_residual_slider),
            dbc.Row([
                make_hideable(
                    dbc.Col([
                        html.Div([
                        dbc.RadioItems(
                            id='random-index-reg-preds-or-y-'+self.name,
                            options=[
                                {'label': 'Use Predicted Range', 'value': 'preds'},
                                {'label': 'Use Observed Range', 'value': 'y'},
                            ],
                            value=self.pred_or_y,
                            inline=True)
                        ], id='random-index-reg-preds-or-y-div-'+self.name),
                        dbc.Tooltip(f"You can either only select a random {self.explainer.index_name}"
                                    f"from within a certain range of observed {self.explainer.target} or"
                                    f"from within a certain range of predicted {self.explainer.target}.", 
                            target='random-index-reg-preds-or-y-div-'+self.name)
                    ], md=4), hide=self.hide_pred_or_y),
                make_hideable(
                    dbc.Col([
                        html.Div([
                            dbc.RadioItems(
                                id='random-index-reg-abs-residual-'+self.name,
                                options=[
                                    {'label': 'Use Residuals', 'value': 'relative'},
                                    {'label': 'Use Absolute Residuals', 'value': 'absolute'},
                                ],
                                value='absolute' if self.abs_residuals else 'relative',
                                inline=True),
                        ], id='random-index-reg-abs-residual-div-'+self.name),
                        dbc.Tooltip(f"You can either only select random a {self.explainer.index_name} "
                                    f"from within a certain range of residuals "
                                    f"(difference between observed and predicted {self.explainer.target}), "
                                    f"so for example only {self.explainer.index_name} for whom the prediction "
                                    f"was too high or too low."
                                    f"Or you can select only from a certain absolute residual range. So for "
                                    f"example only select {self.explainer.index_name} for which the prediction was at "
                                    f"least a certain amount of {self.explainer.units} off.",
                        target='random-index-reg-abs-residual-div-'+self.name),

                    ], md=4), hide=self.hide_abs_residuals),
            ]),
        ])

    def _register_callbacks(self, app):
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


class CutoffPercentileComponent(ExplainerComponent):
    def __init__(self, explainer, title="Global cutoff", name=None,
                        hide_cutoff=False, hide_percentile=False,
                        hide_selector=False,
                        pos_label=None, cutoff=0.5, percentile=None, **kwargs):
        """
        Slider to set a cutoff for Classifier components, based on setting the
        cutoff at a certain percentile of predictions, e.g.:
        percentile=0.8 means "mark the 20% highest scores as positive".

        This cutoff can then be conencted with other components like e.g.
        RocAucComponent with a CutoffConnector.

        Args:
            explainer (Explainer): explainer object constructed with either
                        ClassifierExplainer() or RegressionExplainer()
            title (str, optional): Title of tab or page. Defaults to
                        "Global Cutoff".
            name (str, optional): unique name to add to Component elements.
                        If None then random uuid is generated to make sure
                        it's unique. Defaults to None.
            hide_cutoff (bool, optional): Hide the cutoff slider. Defaults to False.
            hide_percentile (bool, optional): Hide percentile slider. Defaults to False.
            hide_selector (bool, optional): hide pos label selectors. Defaults to False.
            pos_label ({int, str}, optional): initial pos label.
                        Defaults to explainer.pos_label
            cutoff (float, optional): Initial cutoff. Defaults to 0.5.
            percentile ([type], optional): Initial percentile. Defaults to None.
        """
        super().__init__(explainer, title, name)

        self.cutoff_name = 'cutoffconnector-cutoff-'+self.name

        self.selector = PosLabelSelector(explainer, name=self.name, pos_label=pos_label)
        self.register_dependencies(['preds', 'pred_percentiles'])

    def layout(self):
        return html.Div([
            dbc.Row([
                dbc.Col([
                    dbc.Row([
                        make_hideable(
                            dbc.Col([
                                html.Div([
                                    html.Label('Cutoff prediction probability:'),
                                    dcc.Slider(id='cutoffconnector-cutoff-'+self.name,
                                                min = 0.01, max = 0.99, step=0.01, value=self.cutoff,
                                                marks={0.01: '0.01', 0.25: '0.25', 0.50: '0.50',
                                                        0.75: '0.75', 0.99: '0.99'},
                                                included=False,
                                                tooltip = {'always_visible' : False}),
                                    
                                ], style={'margin-bottom': 15}, id='cutoffconnector-cutoff-div-'+self.name),
                                dbc.Tooltip(f"Scores above this cutoff will be labeled positive",
                                                target='cutoffconnector-cutoff-div-'+self.name,
                                                placement='bottom'),
                            ]), hide=self.hide_cutoff),
                    ]),
                    dbc.Row([
                            make_hideable(
                            dbc.Col([
                                html.Div([
                                    html.Label('Cutoff percentile of samples:'),
                                    dcc.Slider(id='cutoffconnector-percentile-'+self.name,
                                                min = 0.01, max = 0.99, step=0.01, value=self.percentile,
                                                marks={0.01: '0.01', 0.25: '0.25', 0.50: '0.50',
                                                        0.75: '0.75', 0.99: '0.99'},
                                                included=False,
                                                tooltip = {'always_visible' : False}),
                                    
                                ], style={'margin-bottom': 15}, id='cutoffconnector-percentile-div-'+self.name),
                                dbc.Tooltip(f"example: if set to percentile=0.9: label the top 10% highest scores as positive, the rest negative.",
                                                target='cutoffconnector-percentile-div-'+self.name,
                                                placement='bottom'),
                            ]), hide=self.hide_percentile),
                    ])
                ]),
                make_hideable(
                    dbc.Col([
                        self.selector.layout()
                    ], width=2), hide=self.hide_selector),
            ])
        ])


    def _register_callbacks(self, app):
        @app.callback(
            Output('cutoffconnector-cutoff-'+self.name, 'value'),
            [Input('cutoffconnector-percentile-'+self.name, 'value'),
             Input('pos-label-'+self.name, 'value')]
        )
        def update_cutoff(percentile, pos_label):
            if percentile is not None:
                return np.round(self.explainer.cutoff_from_percentile(percentile, pos_label=pos_label), 2)
            raise PreventUpdate

class PosLabelConnector(ExplainerComponent):
    def __init__(self, input_pos_label, output_pos_labels):
        self.input_pos_label_name = self._get_pos_label(input_pos_label)
        self.output_pos_label_names = self._get_pos_labels(output_pos_labels)
        # if self.input_pos_label_name in self.output_pos_label_names:
        #     # avoid circulat callbacks
        #     self.output_pos_label_names.remove(self.input_pos_label_name)

    def _get_pos_label(self, input_pos_label):
        if isinstance(input_pos_label, PosLabelSelector):
            return 'pos-label-' + input_pos_label.name
        elif hasattr(input_pos_label, 'selector') and isinstance(input_pos_label.selector, PosLabelSelector):
            return 'pos-label-' + input_pos_label.selector.name
        elif isinstance(input_pos_label, str):
            return input_pos_label
        else:
            raise ValueError("input_pos_label should either be a str, "
                    "PosLabelSelector or an instance with a .selector property"
                    " that is a PosLabelSelector!")

    def _get_pos_labels(self, output_pos_labels):
        def get_pos_labels(o):
            if isinstance(o, PosLabelSelector):
                return ['pos-label-'+o.name]
            elif isinstance(o, str):
                return [str]
            elif hasattr(o, 'pos_labels'):
                return o.pos_labels
            return []

        if hasattr(output_pos_labels, '__iter__'):
            pos_labels = []
            for comp in output_pos_labels:
                pos_labels.extend(get_pos_labels(comp))
            return list(set(pos_labels))
        else:
            return get_pos_labels(output_pos_labels)

    def _register_callbacks(self, app):
        if self.output_pos_label_names:
            @app.callback(
                [Output(pos_label_name, 'value') for pos_label_name in self.output_pos_label_names],
                [Input(self.input_pos_label_name, 'value')]
            )
            def update_pos_labels(pos_label):
                return tuple(pos_label for i in range(len(self.output_pos_label_names)))



class CutoffConnector(ExplainerComponent):
    def __init__(self, input_cutoff, output_cutoffs):
        """Connect the cutoff selector of input_cutoff with those of output_cutoffs.

        You can use this to connect a CutoffPercentileComponent with a
        RocAucComponent for example,

        When you change the cutoff in input_cutoff, all the cutoffs in output_cutoffs
        will automatically be updated.

        Args:
            input_cutoff ([{str, ExplainerComponent}]): Either a str or an
                        ExplainerComponent. If str should be equal to the
                        name of the cutoff property. If ExplainerComponent then
                        should have a .cutoff_name property.
            output_cutoffs (list(str, ExplainerComponent)): list of str of
                        ExplainerComponents.
        """
        self.input_cutoff_name = self.cutoff_name(input_cutoff)
        self.output_cutoff_names = self.cutoff_name(output_cutoffs)
        if not isinstance(self.output_cutoff_names, list):
            self.output_cutoff_names = [self.output_cutoff_names]

    @staticmethod
    def cutoff_name(cutoffs):
        def get_cutoff_name(o):
            if isinstance(o, str): return o
            elif isinstance(o, ExplainerComponent):
                if not hasattr(o, "cutoff_name"):
                    raise ValueError(f"{o} does not have an .cutoff_name property!")
                return o.cutoff_name
            raise ValueError(f"{o} is neither str nor an ExplainerComponent with an .cutoff_name property")

        if hasattr(cutoffs, '__iter__'):
            cutoff_name_list = []
            for cutoff in cutoffs:
                cutoff_name_list.append(get_cutoff_name(cutoff))
            return cutoff_name_list
        else:
            return get_cutoff_name(cutoffs)

    def _register_callbacks(self, app):
        @app.callback(
            [Output(cutoff_name, 'value') for cutoff_name in self.output_cutoff_names],
            [Input(self.input_cutoff_name, 'value')]
        )
        def update_cutoffs(cutoff):
            return tuple(cutoff for i in range(len(self.output_cutoff_names)))


class IndexConnector(ExplainerComponent):
    def __init__(self, input_index, output_indexes):
        """Connect the index selector of input_index with those of output_indexes.

        You can use this to connect a RandomIndexComponent with a
        PredictionSummaryComponent for example.

        When you change the index in input_index, all the indexes in output_indexes
        will automatically be updated.

        Args:
            input_index ([{str, ExplainerComponent}]): Either a str or an
                        ExplainerComponent. If str should be equal to the
                        name of the index property. If ExplainerComponent then
                        should have a .index_name property.
            output_indexes (list(str, ExplainerComponent)): list of str of
                        ExplainerComponents.
        """
        self.input_index_name = self.index_name(input_index)
        self.output_index_names = self.index_name(output_indexes)
        if not isinstance(self.output_index_names, list):
            self.output_index_names = [self.output_index_names]

    @staticmethod
    def index_name(indexes):#, multi=False):
        def get_index_name(o):
            if isinstance(o, str): return o
            elif isinstance(o, ExplainerComponent):
                if not hasattr(o, "index_name"):
                    raise ValueError(f"{o} does not have an .index_name property!")
                return o.index_name
            raise ValueError(f"{o} is neither str nor an ExplainerComponent with an .index_name property")

        if hasattr(indexes, '__iter__'):
            index_name_list = []
            for index in indexes:
                index_name_list.append(get_index_name(index))
            return index_name_list
        else:
            return get_index_name(indexes)

    def _register_callbacks(self, app):
        @app.callback(
            [Output(index_name, 'value') for index_name in self.output_index_names],
            [Input(self.input_index_name, 'value')]
        )
        def update_indexes(index):
            return tuple(index for i in range(len(self.output_index_names)))


class HighlightConnector(ExplainerComponent):
    def __init__(self, input_highlight, output_highlights):
        """Connect the highlight selector of input_highlight with those of output_highlights.

        You can use this to connect a DecisionTreesComponent component to a
        DecisionPathGraphComponent for example.

        When you change the highlight in input_highlight, all the highlights in output_highlights
        will automatically be updated.

        Args:
            input_highlight ([{str, ExplainerComponent}]): Either a str or an
                        ExplainerComponent. If str should be equal to the
                        name of the highlight property. If ExplainerComponent then
                        should have a .highlight_name property.
            output_highlights (list(str, ExplainerComponent)): list of str of
                        ExplainerComponents.
        """
        self.input_highlight_name = self.highlight_name(input_highlight)
        self.output_highlight_names = self.highlight_name(output_highlights)
        if not isinstance(self.output_highlight_names, list):
            self.output_highlight_names = [self.output_highlight_names]

    @staticmethod
    def highlight_name(highlights):
        def get_highlight_name(o):
            if isinstance(o, str): return o
            elif isinstance(o, ExplainerComponent):
                if not hasattr(o, "highlight_name"):
                    raise ValueError(f"{o} does not have an .highlight_name property!")
                return o.highlight_name
            raise ValueError(f"{o} is neither str nor an ExplainerComponent with an .highlight_name property")

        if hasattr(highlights, '__iter__'):
            highlight_name_list = []
            for highlight in highlights:
                highlight_name_list.append(get_highlight_name(highlight))
            return highlight_name_list
        else:
            return get_highlight_name(highlights)

    def _register_callbacks(self, app):
        @app.callback(
            [Output(highlight_name, 'value') for highlight_name in self.output_highlight_names],
            [Input(self.input_highlight_name, 'value')])
        def update_highlights(highlight):
            return tuple(highlight for i in range(len(self.output_highlight_names)))
