__all__ = [
    "ClassifierRandomIndexComponent",
    "ClassifierPredictionSummaryComponent",
    "PrecisionComponent",
    "ConfusionMatrixComponent",
    "LiftCurveComponent",
    "ClassificationComponent",
    "RocAucComponent",
    "PrAucComponent",
    "CumulativePrecisionComponent",
    "ClassifierModelSummaryComponent",
]

import numpy as np
import pandas as pd

import dash
from dash import html, dcc, Input, Output, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

import plotly.graph_objs as go

from ..dashboard_methods import *
from .. import to_html


class ClassifierRandomIndexComponent(ExplainerComponent):
    _state_props = dict(index=("random-index-clas-index-", "value"))

    def __init__(
        self,
        explainer,
        title="Select Random Index",
        name=None,
        subtitle="Select from list or pick at random",
        hide_title=False,
        hide_subtitle=False,
        hide_index=False,
        hide_slider=False,
        hide_labels=False,
        hide_pred_or_perc=False,
        hide_selector=False,
        hide_button=False,
        index_dropdown=True,
        pos_label=None,
        index=None,
        slider=None,
        labels=None,
        pred_or_perc="predictions",
        description=None,
        **kwargs,
    ):
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
            hide_title (bool, optional): Hide title. Defaults to False.
            hide_subtitle (bool, optional): Hide subtitle. Defaults to False.
            hide_index (bool, optional): Hide index selector. Defaults to False.
            hide_slider (bool, optional): Hide prediction/percentile slider.
                        Defaults to False.
            hide_labels (bool, optional): Hide label selector Defaults to False.
            hide_pred_or_perc (bool, optional): Hide prediction/percentiles
                        toggle. Defaults to False.
            hide_selector (bool, optional): hide pos label selectors. Defaults to False.
            hide_button (bool, optional): Hide button. Defaults to False.
            index_dropdown (bool, optional): Use dropdown for index input instead
                of free text input. Defaults to True.
            pos_label ({int, str}, optional): initial pos label. Defaults to explainer.pos_label
            index ({str, int}, optional): Initial index to display.
                        Defaults to None.
            slider ([float,float], optional): initial slider position
                        [lower bound, upper bound]. Defaults to None.
            labels ([str], optional): list of initial labels(str) to include.
                        Defaults to None.
            pred_or_perc (str, optional): Whether to use prediction or
                        percentiles slider. Defaults to 'predictions'.
            description (str, optional): Tooltip to display when hover over
                component title. When None default text is shown.
        """
        super().__init__(explainer, title, name)
        assert self.explainer.is_classifier, (
            "explainer is not a ClassifierExplainer "
            "so the ClassifierRandomIndexComponent "
            " will not work. Try using the RegressionRandomIndexComponent instead."
        )
        self.index_name = "random-index-clas-index-" + self.name

        if self.slider is None:
            self.slider = [0.0, 1.0]

        if self.labels is None:
            self.labels = self.explainer.labels

        if self.explainer.y_missing:
            self.hide_labels = True

        assert (
            len(self.slider) == 2
            and self.slider[0] >= 0
            and self.slider[0] <= 1
            and self.slider[1] >= 0.0
            and self.slider[1] <= 1.0
            and self.slider[0] <= self.slider[1]
        ), "slider should be e.g. [0.5, 1.0]"

        assert all(
            [lab in self.explainer.labels for lab in self.labels]
        ), f"These labels are not in explainer.labels: {[lab for lab in self.labels if lab not in explainer.labels]}!"

        assert self.pred_or_perc in [
            "predictions",
            "percentiles",
        ], "pred_or_perc should either be `predictions` or `percentiles`!"

        self.selector = PosLabelSelector(explainer, name=self.name, pos_label=pos_label)
        self.index_selector = IndexSelector(
            explainer,
            "random-index-clas-index-" + self.name,
            index=index,
            index_dropdown=index_dropdown,
            **kwargs,
        )

        if self.description is None:
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
        return dbc.Card(
            [
                make_hideable(
                    dbc.CardHeader(
                        [
                            html.Div(
                                [
                                    html.H3(
                                        f"Select {self.explainer.index_name}",
                                        id="random-index-clas-title-" + self.name,
                                    ),
                                    make_hideable(
                                        make_hideable(
                                            html.H6(
                                                self.subtitle, className="card-subtitle"
                                            ),
                                            hide=self.hide_subtitle,
                                        ),
                                        hide=self.hide_subtitle,
                                    ),
                                    dbc.Tooltip(
                                        self.description,
                                        target="random-index-clas-title-" + self.name,
                                    ),
                                ]
                            ),
                        ]
                    ),
                    hide=self.hide_title,
                ),
                dbc.CardBody(
                    [
                        dbc.Row(
                            [
                                make_hideable(
                                    dbc.Col([self.selector.layout()], md=2),
                                    hide=self.hide_selector,
                                ),
                            ]
                        ),
                        dbc.Row(
                            [
                                make_hideable(
                                    dbc.Col(
                                        [self.index_selector.layout()], width=8, md=8
                                    ),
                                    hide=self.hide_index,
                                ),
                                make_hideable(
                                    dbc.Col(
                                        [
                                            dbc.Button(
                                                f"Random {self.explainer.index_name}",
                                                color="primary",
                                                id="random-index-clas-button-"
                                                + self.name,
                                            ),
                                            dbc.Tooltip(
                                                f"Select a random {self.explainer.index_name} according to the constraints",
                                                target="random-index-clas-button-"
                                                + self.name,
                                            ),
                                        ],
                                        width=4,
                                        md=4,
                                    ),
                                    hide=self.hide_button,
                                ),
                            ],
                            class_name="mb-2",
                        ),
                        dbc.Row(
                            [
                                make_hideable(
                                    dbc.Col(
                                        [
                                            dbc.Label(
                                                f"Observed {self.explainer.target}:",
                                                id="random-index-clas-labels-label-"
                                                + self.name,
                                            ),
                                            dbc.Tooltip(
                                                f"Only select a random {self.explainer.index_name} where the observed "
                                                f"{self.explainer.target} is one of the selected labels:",
                                                target="random-index-clas-labels-label-"
                                                + self.name,
                                            ),
                                            dcc.Dropdown(
                                                id="random-index-clas-labels-"
                                                + self.name,
                                                options=[
                                                    {"label": lab, "value": lab}
                                                    for lab in self.explainer.labels
                                                ],
                                                multi=True,
                                                value=self.labels,
                                            ),
                                        ],
                                        width=8,
                                        md=8,
                                    ),
                                    hide=self.hide_labels,
                                ),
                                make_hideable(
                                    dbc.Col(
                                        [
                                            dbc.Label(
                                                "Range:",
                                                html_for="random-index-clas-pred-or-perc-"
                                                + self.name,
                                            ),
                                            dbc.Select(
                                                id="random-index-clas-pred-or-perc-"
                                                + self.name,
                                                options=[
                                                    {
                                                        "label": "probability",
                                                        "value": "predictions",
                                                    },
                                                    {
                                                        "label": "percentile",
                                                        "value": "percentiles",
                                                    },
                                                ],
                                                value=self.pred_or_perc,
                                            ),
                                            dbc.Tooltip(
                                                "Instead of selecting from a range of predicted probabilities "
                                                "you can also select from a range of predicted percentiles. "
                                                "For example if you set the slider to percentile (0.9-1.0) you would"
                                                f" only sample random {self.explainer.index_name} from the top "
                                                "10% highest predicted probabilities.",
                                                target="random-index-clas-pred-or-perc-div-"
                                                + self.name,
                                            ),
                                        ],
                                        width=4,
                                        id="random-index-clas-pred-or-perc-div-"
                                        + self.name,
                                    ),
                                    hide=self.hide_pred_or_perc,
                                ),
                            ],
                            class_name="mb-2",
                        ),
                        dbc.Row(
                            [
                                make_hideable(
                                    dbc.Col(
                                        [
                                            html.Div(
                                                [
                                                    dbc.Label(
                                                        id="random-index-clas-slider-label-"
                                                        + self.name,
                                                        children="Predicted probability range:",
                                                        html_for="prediction-range-slider-"
                                                        + self.name,
                                                    ),
                                                    dbc.Tooltip(
                                                        f"Only select a random {self.explainer.index_name} where the "
                                                        "predicted probability of positive label is in the following range:",
                                                        id="random-index-clas-slider-label-tooltip-"
                                                        + self.name,
                                                        target="random-index-clas-slider-label-"
                                                        + self.name,
                                                    ),
                                                    dcc.RangeSlider(
                                                        id="random-index-clas-slider-"
                                                        + self.name,
                                                        min=0.0,
                                                        max=1.0,
                                                        step=0.01,
                                                        value=self.slider,
                                                        allowCross=False,
                                                        marks={
                                                            0.0: "0.0",
                                                            0.2: "0.2",
                                                            0.4: "0.4",
                                                            0.6: "0.6",
                                                            0.8: "0.8",
                                                            1.0: "1.0",
                                                        },
                                                        tooltip={
                                                            "always_visible": False
                                                        },
                                                    ),
                                                ]
                                            )
                                        ]
                                    ),
                                    hide=self.hide_slider,
                                ),
                            ],
                            justify="start",
                        ),
                    ]
                ),
            ],
            class_name="h-100",
        )

    def to_html(self, state_dict=None, add_header=True):
        args = self.get_state_args(state_dict)

        html = to_html.card(
            f"Selected index: <b>{self.explainer.get_index(args['index'])}</b>",
            title=self.title,
        )
        if add_header:
            return to_html.add_header(html)
        return html

    def component_callbacks(self, app):
        @app.callback(
            Output("random-index-clas-index-" + self.name, "value"),
            [Input("random-index-clas-button-" + self.name, "n_clicks")],
            [
                State("random-index-clas-slider-" + self.name, "value"),
                State("random-index-clas-labels-" + self.name, "value"),
                State("random-index-clas-pred-or-perc-" + self.name, "value"),
                State("pos-label-" + self.name, "value"),
            ],
        )
        def update_index(n_clicks, slider_range, labels, pred_or_perc, pos_label):
            triggers = [
                trigger["prop_id"] for trigger in dash.callback_context.triggered
            ]
            if f"random-index-clas-button-{self.name}.n_clicks" not in triggers:
                raise PreventUpdate
            if pred_or_perc == "predictions":
                index = self.explainer.random_index(
                    y_values=labels,
                    pred_proba_min=slider_range[0],
                    pred_proba_max=slider_range[1],
                    return_str=True,
                    pos_label=pos_label,
                )
                return index
            elif pred_or_perc == "percentiles":
                index = self.explainer.random_index(
                    y_values=labels,
                    pred_percentile_min=slider_range[0],
                    pred_percentile_max=slider_range[1],
                    return_str=True,
                    pos_label=pos_label,
                )
                return index
            raise PreventUpdate

        @app.callback(
            [
                Output("random-index-clas-slider-label-" + self.name, "children"),
                Output(
                    "random-index-clas-slider-label-tooltip-" + self.name, "children"
                ),
            ],
            [
                Input("random-index-clas-pred-or-perc-" + self.name, "value"),
                Input("pos-label-" + self.name, "value"),
            ],
        )
        def update_slider_label(pred_or_perc, pos_label):
            if pred_or_perc == "predictions":
                return (
                    "Predicted probability range:",
                    f"Only select a random {self.explainer.index_name} where the "
                    f"predicted probability of {self.explainer.labels[pos_label]}"
                    " is in the following range:",
                )
            elif pred_or_perc == "percentiles":
                return (
                    "Predicted percentile range:",
                    f"Only select a random {self.explainer.index_name} where the "
                    f"predicted probability of {self.explainer.labels[pos_label]}"
                    " is in the following percentile range. For example you can "
                    "only sample from the top 10% highest predicted probabilities.",
                )
            raise PreventUpdate


class ClassifierPredictionSummaryComponent(ExplainerComponent):
    _state_props = dict(
        index=("clas-prediction-index-", "value"), pos_label=("pos-label-", "value")
    )

    def __init__(
        self,
        explainer,
        title="Prediction",
        name=None,
        hide_index=False,
        hide_title=False,
        hide_subtitle=False,
        hide_table=False,
        hide_piechart=False,
        hide_star_explanation=False,
        hide_selector=False,
        index_dropdown=True,
        feature_input_component=None,
        pos_label=None,
        index=None,
        round=3,
        description=None,
        **kwargs,
    ):
        """Shows a summary for a particular prediction

        Args:
            explainer (Explainer): explainer object constructed with either
                ClassifierExplainer() or RegressionExplainer()
            title (str, optional): Title of tab or page. Defaults to
                "Prediction Summary".
            name (str, optional): unique name to add to Component elements.
                If None then random uuid is generated to make sure
                it's unique. Defaults to None.
            hide_index (bool, optional): hide index selector. Defaults to False.
            hide_title (bool, optional): hide title. Defaults to False.
            hide_subtitle (bool, optional): Hide subtitle. Defaults to False.
            hide_table (bool, optional): hide the results table
            hide_piechart (bool, optional): hide the results piechart
            hide_star_explanation (bool, optional): hide the `* indicates..`
                Defaults to False.
            hide_selector (bool, optional): hide pos label selectors.
                Defaults to False.
            index_dropdown (bool, optional): Use dropdown for index input instead
                of free text input. Defaults to True.
            feature_input_component (FeatureInputComponent): A FeatureInputComponent
                that will give the input to the graph instead of the index selector.
                If not None, hide_index=True. Defaults to None.
            pos_label ({int, str}, optional): initial pos label.
                Defaults to explainer.pos_label
            index ({int, str}, optional): Index to display prediction summary
                for. Defaults to None.
            round (int, optional): rounding to apply to pred_proba float.
                Defaults to 3.
            description (str, optional): Tooltip to display when hover over
                component title. When None default text is shown.
        """
        super().__init__(explainer, title, name)

        self.index_name = "clas-prediction-index-" + self.name
        self.selector = PosLabelSelector(explainer, name=self.name, pos_label=pos_label)
        self.index_selector = IndexSelector(
            explainer,
            "clas-prediction-index-" + self.name,
            index=index,
            index_dropdown=index_dropdown,
            **kwargs,
        )

        if self.feature_input_component is not None:
            self.exclude_callbacks(self.feature_input_component)
            self.hide_index = True

        if self.description is None:
            self.description = f"""
        Shows the predicted probability for each {self.explainer.target} label.
        """

        self.register_dependencies("metrics")

    def layout(self):
        return dbc.Card(
            [
                make_hideable(
                    dbc.CardHeader(
                        [
                            html.Div(
                                [
                                    html.H3(
                                        self.title,
                                        id="clas-prediction-index-title-" + self.name,
                                        className="card-title",
                                    ),
                                    dbc.Tooltip(
                                        self.description,
                                        target="clas-prediction-index-title-"
                                        + self.name,
                                    ),
                                ]
                            ),
                        ]
                    ),
                    hide=self.hide_title,
                ),
                dbc.CardBody(
                    [
                        dbc.Row(
                            [
                                make_hideable(
                                    dbc.Col(
                                        [
                                            dbc.Label(f"{self.explainer.index_name}:"),
                                            self.index_selector.layout(),
                                        ],
                                        md=6,
                                    ),
                                    hide=self.hide_index,
                                ),
                                make_hideable(
                                    dbc.Col([self.selector.layout()], width=3),
                                    hide=self.hide_selector,
                                ),
                            ]
                        ),
                        dbc.Row(
                            [
                                make_hideable(
                                    dbc.Col(
                                        [
                                            html.Div(
                                                id="clas-prediction-div-" + self.name
                                            ),
                                            make_hideable(
                                                html.Div("* indicates observed label")
                                                if not self.explainer.y_missing
                                                else None,
                                                hide=self.hide_star_explanation,
                                            ),
                                        ]
                                    ),
                                    hide=self.hide_table,
                                ),
                                make_hideable(
                                    dbc.Col(
                                        [
                                            dcc.Graph(
                                                id="clas-prediction-graph-" + self.name,
                                                config=dict(
                                                    modeBarButtons=[["toImage"]],
                                                    displaylogo=False,
                                                ),
                                            )
                                        ]
                                    ),
                                    hide=self.hide_piechart,
                                ),
                            ]
                        ),
                    ]
                ),
            ],
            class_name="h-100",
        )

    def _format_preds_df(self, preds_df):
        preds_df.probability = np.round(
            100 * preds_df.probability.values, self.round
        ).astype(str)
        preds_df.probability = preds_df.probability + " %"
        if "logodds" in preds_df.columns:
            preds_df.logodds = np.round(preds_df.logodds.values, self.round).astype(str)

        if self.explainer.model_output != "logodds":
            preds_df = preds_df[["label", "probability"]]
        return preds_df

    def get_state_tuples(self):
        _state_tuples = super().get_state_tuples()
        if self.feature_input_component is not None:
            _state_tuples.extend(self.feature_input_component.get_state_tuples())
        return sorted(list(set(_state_tuples)))

    def to_html(self, state_dict=None, add_header=True):
        args = self.get_state_args(state_dict)
        if self.feature_input_component is None:
            if args["index"] is not None:
                fig = self.explainer.plot_prediction_result(
                    args["index"], showlegend=False
                )
                preds_df = self.explainer.prediction_result_df(
                    args["index"], round=self.round, logodds=True
                )
                preds_df = self._format_preds_df(preds_df)
                html = to_html.row(to_html.table_from_df(preds_df), to_html.fig(fig))
            else:
                html = "no index selected"
        else:
            inputs = {
                k: v
                for k, v in self.feature_input_component.get_state_args(
                    state_dict
                ).items()
                if k != "index"
            }
            inputs = list(inputs.values())
            if len(inputs) == len(
                self.feature_input_component._input_features
            ) and not any([i is None for i in inputs]):
                X_row = self.explainer.get_row_from_input(inputs, ranked_by_shap=True)
                fig = self.explainer.plot_prediction_result(
                    X_row=X_row, showlegend=False
                )
                preds_df = self.explainer.prediction_result_df(
                    X_row=X_row, round=self.round, logodds=True
                )
                preds_df = self._format_preds_df(preds_df)
                html = to_html.row(to_html.table_from_df(preds_df), to_html.fig(fig))
            else:
                html = f"<div>input data incorrect</div>"

        html = to_html.card(html, title=self.title)
        if add_header:
            return to_html.add_header(html)
        return html

    def component_callbacks(self, app):
        if self.feature_input_component is None:

            @app.callback(
                [
                    Output("clas-prediction-div-" + self.name, "children"),
                    Output("clas-prediction-graph-" + self.name, "figure"),
                ],
                [
                    Input("clas-prediction-index-" + self.name, "value"),
                    Input("pos-label-" + self.name, "value"),
                ],
            )
            def update_output_div(index, pos_label):
                if index is None or not self.explainer.index_exists(index):
                    raise PreventUpdate
                fig = self.explainer.plot_prediction_result(index, showlegend=False)
                preds_df = self.explainer.prediction_result_df(
                    index, round=self.round, logodds=True
                )
                preds_df = self._format_preds_df(preds_df)
                preds_table = dbc.Table.from_dataframe(
                    preds_df, striped=False, bordered=False, hover=False
                )
                return preds_table, fig

        else:

            @app.callback(
                [
                    Output("clas-prediction-div-" + self.name, "children"),
                    Output("clas-prediction-graph-" + self.name, "figure"),
                ],
                [
                    Input("pos-label-" + self.name, "value"),
                    *self.feature_input_component._feature_callback_inputs,
                ],
            )
            def update_output_div(pos_label, *inputs):
                X_row = self.explainer.get_row_from_input(inputs, ranked_by_shap=True)
                fig = self.explainer.plot_prediction_result(
                    X_row=X_row, showlegend=False
                )

                preds_df = self.explainer.prediction_result_df(
                    X_row=X_row, round=self.round, logodds=True
                )
                preds_df = self._format_preds_df(preds_df)
                preds_table = dbc.Table.from_dataframe(
                    preds_df, striped=False, bordered=False, hover=False
                )
                return preds_table, fig


class PrecisionComponent(ExplainerComponent):
    _state_props = dict(
        bin_size=("precision-binsize-", "value"),
        quantiles=("precision-quantiles-", "value"),
        quantiles_or_binsize=("precision-binsize-or-quantiles-", "value"),
        cutoff=("precision-cutoff-", "value"),
        multiclass=("precision-multiclass-", "value"),
        pos_label=("pos-label-", "value"),
    )

    def __init__(
        self,
        explainer,
        title="Precision Plot",
        name=None,
        subtitle="Does fraction positive increase with predicted probability?",
        hide_title=False,
        hide_subtitle=False,
        hide_footer=False,
        hide_cutoff=False,
        hide_binsize=False,
        hide_binmethod=False,
        hide_multiclass=False,
        hide_selector=False,
        hide_popout=False,
        pos_label=None,
        bin_size=0.1,
        quantiles=10,
        cutoff=0.5,
        quantiles_or_binsize="bin_size",
        multiclass=False,
        description=None,
        **kwargs,
    ):
        """Shows a precision graph with toggles.

        Args:
            explainer (Explainer): explainer object constructed with either
                        ClassifierExplainer() or RegressionExplainer()
            title (str, optional): Title of tab or page. Defaults to
                        "Precision Plot".
            name (str, optional): unique name to add to Component elements.
                        If None then random uuid is generated to make sure
                        it's unique. Defaults to None.
            subtitle (str): subtitle
            hide_title (bool, optional): hide title
            hide_subtitle (bool, optional): Hide subtitle. Defaults to False.
            hide_footer (bool, optional): hide the footer at the bottom of the component
            hide_cutoff (bool, optional): Hide cutoff slider. Defaults to False.
            hide_binsize (bool, optional): hide binsize/quantiles slider. Defaults to False.
            hide_selector(bool, optional): hide pos label selector. Defaults to False.
            hide_binmethod (bool, optional): Hide binsize/quantiles toggle. Defaults to False.
            hide_multiclass (bool, optional): Hide multiclass toggle. Defaults to False.
            hide_selector (bool, optional): Hide pos label selector. Default to True.
            hide_popout (bool, optional): hide popout button
            pos_label ({int, str}, optional): initial pos label. Defaults to explainer.pos_label
            bin_size (float, optional): Size of bins in probability space. Defaults to 0.1.
            quantiles (int, optional): Number of quantiles to divide plot. Defaults to 10.
            cutoff (float, optional): Cutoff to display in graph. Defaults to 0.5.
            quantiles_or_binsize (str, {'quantiles', 'bin_size'}, optional): Default bin method. Defaults to 'bin_size'.
            multiclass (bool, optional): Display all classes. Defaults to False.
            description (str, optional): Tooltip to display when hover over
                component title. When None default text is shown.
        """
        super().__init__(explainer, title, name)

        self.cutoff_name = "precision-cutoff-" + self.name

        self.selector = PosLabelSelector(explainer, name=self.name, pos_label=pos_label)

        if self.description is None:
            self.description = f"""
        On this plot you can see the relation between the predicted probability
        that a {self.explainer.index_name} belongs to the positive class, and
        the percentage of observed {self.explainer.index_name} in the positive class.
        The observations get binned together in groups of roughly 
        equal predicted probabilities, and the percentage of positives is calculated
        for each bin. A perfectly calibrated model would show a straight line
        from the bottom left corner to the top right corner. A strong model would
        classify most observations correctly and close to 0% or 100% probability.
        """
        self.popout = GraphPopout(
            "precision-" + self.name + "popout",
            "precision-graph-" + self.name,
            self.title,
            self.description,
        )
        self.register_dependencies("preds", "pred_probas", "pred_percentiles")

    def layout(self):
        return dbc.Card(
            [
                make_hideable(
                    dbc.CardHeader(
                        [
                            html.Div(
                                [
                                    html.H3(
                                        self.title,
                                        id="precision-title-" + self.name,
                                        className="card-title",
                                    ),
                                    make_hideable(
                                        html.H6(
                                            self.subtitle, className="card-subtitle"
                                        ),
                                        hide=self.hide_subtitle,
                                    ),
                                    dbc.Tooltip(
                                        self.description,
                                        target="precision-title-" + self.name,
                                    ),
                                ]
                            ),
                        ]
                    ),
                    hide=self.hide_title,
                ),
                dbc.CardBody(
                    [
                        dbc.Row(
                            [
                                make_hideable(
                                    dbc.Col([self.selector.layout()], width=3),
                                    hide=self.hide_selector,
                                )
                            ],
                            justify="end",
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.Div(
                                            [
                                                dcc.Graph(
                                                    id="precision-graph-" + self.name,
                                                    config=dict(
                                                        modeBarButtons=[["toImage"]],
                                                        displaylogo=False,
                                                    ),
                                                ),
                                            ],
                                            style={"margin": 0},
                                        ),
                                    ]
                                )
                            ]
                        ),
                        dbc.Row(
                            [
                                make_hideable(
                                    dbc.Col(
                                        [self.popout.layout()], md=2, align="start"
                                    ),
                                    hide=self.hide_popout,
                                ),
                            ],
                            justify="end",
                        ),
                    ]
                ),
                make_hideable(
                    dbc.CardFooter(
                        [
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            make_hideable(
                                                html.Div(
                                                    [
                                                        html.Div(
                                                            [
                                                                dbc.Label(
                                                                    "Bin size:",
                                                                    html_for="precision-binsize-"
                                                                    + self.name,
                                                                ),
                                                                html.Div(
                                                                    [
                                                                        dcc.Slider(
                                                                            id="precision-binsize-"
                                                                            + self.name,
                                                                            min=0.01,
                                                                            max=0.5,
                                                                            step=0.01,
                                                                            value=self.bin_size,
                                                                            marks={
                                                                                0.01: "0.01",
                                                                                0.05: "0.05",
                                                                                0.10: "0.10",
                                                                                0.20: "0.20",
                                                                                0.25: "0.25",
                                                                                0.33: "0.33",
                                                                                0.5: "0.5",
                                                                            },
                                                                            included=False,
                                                                            tooltip={
                                                                                "always_visible": False
                                                                            },
                                                                        )
                                                                    ],
                                                                    style={
                                                                        "margin-bottom": 5
                                                                    },
                                                                ),
                                                            ],
                                                            id="precision-bin-size-div-"
                                                            + self.name,
                                                            style=dict(margin=5),
                                                        ),
                                                        dbc.Tooltip(
                                                            "Size of the bins to divide prediction score by",
                                                            target="precision-bin-size-div-"
                                                            + self.name,
                                                            placement="bottom",
                                                        ),
                                                    ]
                                                ),
                                                hide=self.hide_binsize,
                                            ),
                                            make_hideable(
                                                html.Div(
                                                    [
                                                        html.Div(
                                                            [
                                                                dbc.Label(
                                                                    "Quantiles:",
                                                                    html_for="precision-quantiles-"
                                                                    + self.name,
                                                                ),
                                                                html.Div(
                                                                    [
                                                                        dcc.Slider(
                                                                            id="precision-quantiles-"
                                                                            + self.name,
                                                                            min=1,
                                                                            max=20,
                                                                            step=1,
                                                                            value=self.quantiles,
                                                                            marks={
                                                                                1: "1",
                                                                                5: "5",
                                                                                10: "10",
                                                                                15: "15",
                                                                                20: "20",
                                                                            },
                                                                            included=False,
                                                                            tooltip={
                                                                                "always_visible": False
                                                                            },
                                                                        ),
                                                                    ],
                                                                    style={
                                                                        "margin-bottom": 5
                                                                    },
                                                                ),
                                                            ],
                                                            id="precision-quantiles-div-"
                                                            + self.name,
                                                        ),
                                                        dbc.Tooltip(
                                                            "Number of equally populated bins to divide prediction score by",
                                                            target="precision-quantiles-div-"
                                                            + self.name,
                                                            placement="bottom",
                                                        ),
                                                    ]
                                                ),
                                                hide=self.hide_binsize,
                                            ),
                                            make_hideable(
                                                html.Div(
                                                    [
                                                        html.Div(
                                                            [
                                                                html.Label(
                                                                    "Cutoff prediction probability:"
                                                                ),
                                                                dcc.Slider(
                                                                    id="precision-cutoff-"
                                                                    + self.name,
                                                                    min=0.01,
                                                                    max=0.99,
                                                                    step=0.01,
                                                                    value=self.cutoff,
                                                                    marks={
                                                                        0.01: "0.01",
                                                                        0.25: "0.25",
                                                                        0.50: "0.50",
                                                                        0.75: "0.75",
                                                                        0.99: "0.99",
                                                                    },
                                                                    included=False,
                                                                    tooltip={
                                                                        "always_visible": False
                                                                    },
                                                                    updatemode="drag",
                                                                ),
                                                            ],
                                                            id="precision-cutoff-div-"
                                                            + self.name,
                                                        ),
                                                        dbc.Tooltip(
                                                            f"Scores above this cutoff will be labeled positive",
                                                            target="precision-cutoff-div-"
                                                            + self.name,
                                                            placement="bottom",
                                                        ),
                                                    ],
                                                    style={"margin-bottom": 5},
                                                ),
                                                hide=self.hide_cutoff,
                                            ),
                                        ]
                                    ),
                                ]
                            ),
                            dbc.Row(
                                [
                                    make_hideable(
                                        dbc.Col(
                                            [
                                                dbc.Label(
                                                    "Binning Method:",
                                                    html_for="precision-binsize-or-quantiles-"
                                                    + self.name,
                                                ),
                                                dbc.Select(
                                                    id="precision-binsize-or-quantiles-"
                                                    + self.name,
                                                    options=[
                                                        {
                                                            "label": "Bins",
                                                            "value": "bin_size",
                                                        },
                                                        {
                                                            "label": "Quantiles",
                                                            "value": "quantiles",
                                                        },
                                                    ],
                                                    value=self.quantiles_or_binsize,
                                                    size="sm",
                                                ),
                                                dbc.Tooltip(
                                                    "Divide the x-axis by equally sized ranges of prediction scores (bins),"
                                                    " or bins with the same number of observations (counts) in each bin: quantiles",
                                                    target="precision-binsize-or-quantiles-"
                                                    + self.name,
                                                ),
                                            ],
                                            width=4,
                                        ),
                                        hide=self.hide_binmethod,
                                    ),
                                    make_hideable(
                                        dbc.Col(
                                            [
                                                dbc.Row(
                                                    [
                                                        dbc.Label(
                                                            "Multi class:",
                                                            id="precision-multiclass-label-"
                                                            + self.name,
                                                        ),
                                                        dbc.Tooltip(
                                                            "Display the observed proportion for all class"
                                                            " labels, not just positive label.",
                                                            target="precision-multiclass-"
                                                            + self.name,
                                                        ),
                                                        dbc.Checklist(
                                                            options=[
                                                                {
                                                                    "label": "Display all classes",
                                                                    "value": True,
                                                                }
                                                            ],
                                                            value=[True]
                                                            if self.multiclass
                                                            else [],
                                                            id="precision-multiclass-"
                                                            + self.name,
                                                            inline=True,
                                                            switch=True,
                                                        ),
                                                    ]
                                                ),
                                            ],
                                            width=4,
                                        ),
                                        hide=self.hide_multiclass,
                                    ),
                                ]
                            ),
                        ]
                    ),
                    hide=self.hide_footer,
                ),
            ],
            class_name="h-100",
        )

    def to_html(self, state_dict=None, add_header=True):
        args = self.get_state_args(state_dict)
        if args["quantiles_or_binsize"] == "bin_size":
            fig = self.explainer.plot_precision(
                bin_size=args["bin_size"],
                cutoff=args["cutoff"],
                multiclass=bool(args["multiclass"]),
                pos_label=args["pos_label"],
            )
        elif args["quantiles_or_binsize"] == "quantiles":
            fig = self.explainer.plot_precision(
                quantiles=args["quantiles"],
                cutoff=args["cutoff"],
                multiclass=bool(args["multiclass"]),
                pos_label=args["pos_label"],
            )

        html = to_html.card(to_html.fig(fig), title=self.title, subtitle=self.subtitle)
        if add_header:
            return to_html.add_header(html)
        return html

    def component_callbacks(self, app):
        @app.callback(
            [
                Output("precision-bin-size-div-" + self.name, "style"),
                Output("precision-quantiles-div-" + self.name, "style"),
            ],
            [Input("precision-binsize-or-quantiles-" + self.name, "value")],
        )
        def update_div_visibility(bins_or_quantiles):
            if self.hide_binsize:
                return dict(display="none"), dict(display="none")
            if bins_or_quantiles == "bin_size":
                return {}, dict(display="none")
            elif bins_or_quantiles == "quantiles":
                return dict(display="none"), {}
            raise PreventUpdate

        @app.callback(
            Output("precision-graph-" + self.name, "figure"),
            [
                Input("precision-binsize-" + self.name, "value"),
                Input("precision-quantiles-" + self.name, "value"),
                Input("precision-binsize-or-quantiles-" + self.name, "value"),
                Input("precision-cutoff-" + self.name, "value"),
                Input("precision-multiclass-" + self.name, "value"),
                Input("pos-label-" + self.name, "value"),
            ],
            [State("precision-graph-" + self.name, "figure")],
        )
        def update_precision_graph(
            bin_size, quantiles, bins, cutoff, multiclass, pos_label, fig
        ):
            ctx = dash.callback_context
            trigger = ctx.triggered[0]["prop_id"].split(".")[0]
            if trigger == "precision-cutoff-" + self.name and fig is not None:
                return go.Figure(fig).update_shapes(
                    dict(
                        type="line",
                        xref="x",
                        yref="y2",
                        x0=cutoff,
                        x1=cutoff,
                        y0=0,
                        y1=1.0,
                    )
                )
            if bins == "bin_size":
                return self.explainer.plot_precision(
                    bin_size=bin_size,
                    cutoff=cutoff,
                    multiclass=bool(multiclass),
                    pos_label=pos_label,
                )
            elif bins == "quantiles":
                return self.explainer.plot_precision(
                    quantiles=quantiles,
                    cutoff=cutoff,
                    multiclass=bool(multiclass),
                    pos_label=pos_label,
                )
            raise PreventUpdate


class ConfusionMatrixComponent(ExplainerComponent):
    _state_props = dict(
        cutoff=("confusionmatrix-cutoff-", "value"),
        percentage=("confusionmatrix-percentage-", "value"),
        normalize=("confusionmatrix-normalize-", "value"),
        binary=("confusionmatrix-binary-", "value"),
        pos_label=("pos-label-", "value"),
    )

    def __init__(
        self,
        explainer,
        title="Confusion Matrix",
        name=None,
        subtitle="How many false positives and false negatives?",
        hide_title=False,
        hide_subtitle=False,
        hide_footer=False,
        hide_cutoff=False,
        hide_percentage=False,
        hide_binary=False,
        hide_selector=False,
        hide_popout=False,
        hide_normalize=False,
        normalize="all",
        pos_label=None,
        cutoff=0.5,
        percentage=True,
        binary=True,
        description=None,
        **kwargs,
    ):
        """Display confusion matrix component

        Args:
            explainer (Explainer): explainer object constructed with either
                        ClassifierExplainer() or RegressionExplainer()
            title (str, optional): Title of tab or page. Defaults to
                        "Confusion Matrix".
            name (str, optional): unique name to add to Component elements.
                        If None then random uuid is generated to make sure
                        it's unique. Defaults to None.
            subtitle (str): subtitle
            hide_title (bool, optional): hide title.
            hide_subtitle (bool, optional): Hide subtitle. Defaults to False.
            hide_footer (bool, optional): hide the footer at the bottom of the component
            hide_cutoff (bool, optional): Hide cutoff slider. Defaults to False.
            hide_percentage (bool, optional): Hide percentage toggle. Defaults to False.
            hide_binary (bool, optional): Hide binary toggle. Defaults to False.
            hide_selector(bool, optional): hide pos label selector. Defaults to False.
            hide_popout (bool, optional): hide popout button. Defaults to False.
            pos_label ({int, str}, optional): initial pos label. Defaults to explainer.pos_label
            cutoff (float, optional): Default cutoff. Defaults to 0.5.
            percentage (bool, optional): Display percentages instead of counts. Defaults to True.
            binary (bool, optional): Show binary instead of multiclass confusion matrix. Defaults to True.
            description (str, optional): Tooltip to display when hover over
                component title. When None default text is shown.
            normalize (str[‘true’, ‘pred’, ‘all’]): normalizes confusion matrix over the true (rows), predicted (columns) conditions or all the population.
                Defaults to all
        """
        super().__init__(explainer, title, name)

        self.cutoff_name = "confusionmatrix-cutoff-" + self.name

        if len(self.explainer.labels) <= 2:
            self.hide_binary = True

        if self.description is None:
            self.description = """
        The confusion matrix shows the number of true negatives (predicted negative, observed negative), 
        true positives (predicted positive, observed positive), 
        false negatives (predicted negative, but observed positive) and
        false positives (predicted positive, but observed negative). The amount
        of false negatives and false positives determine the costs of deploying
        and imperfect model. For different cutoffs you will get a different number
        of false positives and false negatives. This plot can help you select
        the optimal cutoff.
        """

        self.selector = PosLabelSelector(explainer, name=self.name, pos_label=pos_label)
        self.popout = GraphPopout(
            "confusionmatrix-" + self.name + "popout",
            "confusionmatrix-graph-" + self.name,
            self.title,
            self.description,
        )
        self.register_dependencies(
            "preds", "pred_probas", "pred_percentiles", "confusion_matrix"
        )

    def layout(self):
        return dbc.Card(
            [
                make_hideable(
                    dbc.CardHeader(
                        [
                            html.Div(
                                [
                                    html.H3(
                                        self.title,
                                        id="confusionmatrix-title-" + self.name,
                                    ),
                                    make_hideable(
                                        html.H6(
                                            self.subtitle, className="card-subtitle"
                                        ),
                                        hide=self.hide_subtitle,
                                    ),
                                    dbc.Tooltip(
                                        self.description,
                                        target="confusionmatrix-title-" + self.name,
                                    ),
                                ]
                            ),
                        ]
                    ),
                    hide=self.hide_title,
                ),
                dbc.CardBody(
                    [
                        dbc.Row(
                            [
                                make_hideable(
                                    dbc.Col([self.selector.layout()], width=3),
                                    hide=self.hide_selector,
                                )
                            ],
                            justify="end",
                        ),
                        dcc.Graph(
                            id="confusionmatrix-graph-" + self.name,
                            config=dict(
                                modeBarButtons=[["toImage"]], displaylogo=False
                            ),
                        ),
                        dbc.Row(
                            [
                                make_hideable(
                                    dbc.Col(
                                        [self.popout.layout()], md=2, align="start"
                                    ),
                                    hide=self.hide_popout,
                                ),
                            ],
                            justify="end",
                        ),
                    ]
                ),
                make_hideable(
                    dbc.CardFooter(
                        [
                            make_hideable(
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.Label(
                                                    "Cutoff prediction probability:"
                                                ),
                                                dcc.Slider(
                                                    id="confusionmatrix-cutoff-"
                                                    + self.name,
                                                    min=0.01,
                                                    max=0.99,
                                                    step=0.01,
                                                    value=self.cutoff,
                                                    marks={
                                                        0.01: "0.01",
                                                        0.25: "0.25",
                                                        0.50: "0.50",
                                                        0.75: "0.75",
                                                        0.99: "0.99",
                                                    },
                                                    included=False,
                                                    tooltip={"always_visible": False},
                                                    updatemode="drag",
                                                ),
                                            ],
                                            id="confusionmatrix-cutoff-div-"
                                            + self.name,
                                        ),
                                        dbc.Tooltip(
                                            f"Scores above this cutoff will be labeled positive",
                                            target="confusionmatrix-cutoff-div-"
                                            + self.name,
                                            placement="bottom",
                                        ),
                                    ],
                                    style={"margin-bottom": 25},
                                ),
                                hide=self.hide_cutoff,
                            ),
                            make_hideable(
                                html.Div(
                                    [
                                        dbc.Row(
                                            [
                                                # dbc.Label("Percentage:", id='confusionmatrix-percentage-label-'+self.name),
                                                dbc.Tooltip(
                                                    "Highlight the percentage in each cell instead of the absolute numbers",
                                                    target="confusionmatrix-percentage-"
                                                    + self.name,
                                                ),
                                                dbc.Checklist(
                                                    options=[
                                                        {
                                                            "label": "Highlight percentage",
                                                            "value": True,
                                                        }
                                                    ],
                                                    value=[True]
                                                    if self.percentage
                                                    else [],
                                                    id="confusionmatrix-percentage-"
                                                    + self.name,
                                                    inline=True,
                                                    switch=True,
                                                ),
                                            ]
                                        ),
                                    ]
                                ),
                                hide=self.hide_percentage,
                            ),
                            make_hideable(
                                html.Div(
                                    [
                                        dbc.Row(
                                            [
                                                dbc.Label("Normalisation:"),
                                                dbc.Tooltip(
                                                    "Normalize the percentages in the confusion matrix over the true observations, predicted values or overall",
                                                    target="confusionmatrix-normalize-"
                                                    + self.name,
                                                ),
                                                dbc.RadioItems(
                                                    options=[
                                                        {
                                                            "label": "Overall",
                                                            "value": "all",
                                                        },
                                                        {
                                                            "label": "Observed",
                                                            "value": "observed",
                                                        },
                                                        {
                                                            "label": "Predicted",
                                                            "value": "pred",
                                                        },
                                                    ],
                                                    value=self.normalize,
                                                    id="confusionmatrix-normalize-"
                                                    + self.name,
                                                    inline=True,
                                                ),
                                            ]
                                        ),
                                    ]
                                ),
                                hide=self.hide_normalize,
                            ),
                            make_hideable(
                                html.Div(
                                    [
                                        dbc.Row(
                                            [
                                                dbc.Label(
                                                    "Binary:",
                                                    id="confusionmatrix-binary-label-"
                                                    + self.name,
                                                ),
                                                dbc.Tooltip(
                                                    "display a binary confusion matrix of positive "
                                                    "class vs all other classes instead of a multi"
                                                    " class confusion matrix.",
                                                    target="confusionmatrix-binary-label-"
                                                    + self.name,
                                                ),
                                                dbc.Checklist(
                                                    options=[
                                                        {
                                                            "label": "Display one-vs-rest matrix",
                                                            "value": True,
                                                        }
                                                    ],
                                                    value=[True] if self.binary else [],
                                                    id="confusionmatrix-binary-"
                                                    + self.name,
                                                    inline=True,
                                                    switch=True,
                                                ),
                                            ]
                                        ),
                                    ]
                                ),
                                hide=self.hide_binary,
                            ),
                        ]
                    ),
                    hide=self.hide_footer,
                ),
            ],
            class_name="h-100",
        )

    def to_html(self, state_dict=None, add_header=True):
        args = self.get_state_args(state_dict)
        args["binary"] = bool(args["binary"])
        args["percentage"] = bool(args["percentage"])
        fig = self.explainer.plot_confusion_matrix(
            cutoff=args["cutoff"],
            percentage=args["percentage"],
            binary=args["binary"],
            normalize=args["normalize"],
            pos_label=args["pos_label"],
        )

        html = to_html.card(to_html.fig(fig), title=self.title, subtitle=self.subtitle)
        if add_header:
            return to_html.add_header(html)
        return html

    def component_callbacks(self, app):
        @app.callback(
            Output("confusionmatrix-graph-" + self.name, "figure"),
            [
                Input("confusionmatrix-cutoff-" + self.name, "value"),
                Input("confusionmatrix-percentage-" + self.name, "value"),
                Input("confusionmatrix-normalize-" + self.name, "value"),
            ],
            Input("confusionmatrix-binary-" + self.name, "value"),
            Input("pos-label-" + self.name, "value"),
        )
        def update_confusionmatrix_graph(
            cutoff, percentage, normalize, binary, pos_label
        ):
            return self.explainer.plot_confusion_matrix(
                cutoff=cutoff,
                percentage=bool(percentage),
                normalize=normalize,
                binary=bool(binary),
                pos_label=pos_label,
            )


class LiftCurveComponent(ExplainerComponent):
    _state_props = dict(
        cutoff=("liftcurve-cutoff-", "value"),
        percentage=("liftcurve-percentage-", "value"),
        wizard=("liftcurve-wizard-", "value"),
        pos_label=("pos-label-", "value"),
    )

    def __init__(
        self,
        explainer,
        title="Lift Curve",
        name=None,
        subtitle="Performance how much better than random?",
        hide_title=False,
        hide_subtitle=False,
        hide_footer=False,
        hide_cutoff=False,
        hide_percentage=False,
        hide_wizard=False,
        hide_selector=False,
        hide_popout=False,
        pos_label=None,
        cutoff=0.5,
        percentage=True,
        wizard=True,
        description=None,
        **kwargs,
    ):
        """Show liftcurve component

        Args:
            explainer (Explainer): explainer object constructed with either
                        ClassifierExplainer() or RegressionExplainer()
            title (str, optional): Title of tab or page. Defaults to
                        "Lift Curve".
            name (str, optional): unique name to add to Component elements.
                        If None then random uuid is generated to make sure
                        it's unique. Defaults to None.
            subtitle (str): subtitle
            hide_title (bool, optional): hide title.
            hide_subtitle (bool, optional): Hide subtitle. Defaults to False.
            hide_footer (bool, optional): hide the footer at the bottom of the component
            hide_cutoff (bool, optional): Hide cutoff slider. Defaults to False.
            hide_percentage (bool, optional): Hide percentage toggle. Defaults to False.
            hide_wizard (bool, optional): hide the wizard toggle. Defaults to False.
            hide_selector(bool, optional): hide pos label selector. Defaults to False.
            hide_popout (bool, optional): hide popout button. Defaults to False.
            pos_label ({int, str}, optional): initial pos label. Defaults to explainer.pos_label
            cutoff (float, optional): Cutoff for lift curve. Defaults to 0.5.
            percentage (bool, optional): Display percentages instead of counts. Defaults to True.
            wizard (bool, optional): display the wizard in the graph.
            description (str, optional): Tooltip to display when hover over
                component title. When None default text is shown.
        """
        super().__init__(explainer, title, name)

        self.cutoff_name = "liftcurve-cutoff-" + self.name

        self.selector = PosLabelSelector(explainer, name=self.name, pos_label=pos_label)

        if self.description is None:
            self.description = """
        The lift curve shows you the percentage of positive classes when you only
        select observations with a score above cutoff vs selecting observations
        randomly. This shows you how much better the model is than random (the lift).
        """

        self.popout = GraphPopout(
            "liftcurve-" + self.name + "popout",
            "liftcurve-graph-" + self.name,
            self.title,
            self.description,
        )
        self.register_dependencies("get_liftcurve_df")

    def layout(self):
        return dbc.Card(
            [
                make_hideable(
                    dbc.CardHeader(
                        [
                            html.Div(
                                [
                                    html.H3(
                                        self.title, id="liftcurve-title-" + self.name
                                    ),
                                    make_hideable(
                                        html.H6(
                                            self.subtitle, className="card-subtitle"
                                        ),
                                        hide=self.hide_subtitle,
                                    ),
                                    dbc.Tooltip(
                                        self.description,
                                        target="liftcurve-title-" + self.name,
                                    ),
                                ]
                            ),
                        ]
                    ),
                    hide=self.hide_title,
                ),
                dbc.CardBody(
                    [
                        dbc.Row(
                            [
                                make_hideable(
                                    dbc.Col([self.selector.layout()], width=3),
                                    hide=self.hide_selector,
                                )
                            ],
                            justify="end",
                        ),
                        html.Div(
                            [
                                dcc.Graph(
                                    id="liftcurve-graph-" + self.name,
                                    config=dict(
                                        modeBarButtons=[["toImage"]], displaylogo=False
                                    ),
                                ),
                            ],
                            style={"margin": 0},
                        ),
                        dbc.Row(
                            [
                                make_hideable(
                                    dbc.Col(
                                        [self.popout.layout()], md=2, align="start"
                                    ),
                                    hide=self.hide_popout,
                                ),
                            ],
                            justify="end",
                        ),
                    ]
                ),
                make_hideable(
                    dbc.CardFooter(
                        [
                            make_hideable(
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.Label(
                                                    "Cutoff prediction probability:"
                                                ),
                                                dcc.Slider(
                                                    id="liftcurve-cutoff-" + self.name,
                                                    min=0.01,
                                                    max=0.99,
                                                    step=0.01,
                                                    value=self.cutoff,
                                                    marks={
                                                        0.01: "0.01",
                                                        0.25: "0.25",
                                                        0.50: "0.50",
                                                        0.75: "0.75",
                                                        0.99: "0.99",
                                                    },
                                                    included=False,
                                                    tooltip={"always_visible": False},
                                                    updatemode="drag",
                                                ),
                                            ],
                                            id="liftcurve-cutoff-div-" + self.name,
                                        ),
                                        dbc.Tooltip(
                                            f"Scores above this cutoff will be labeled positive",
                                            target="liftcurve-cutoff-div-" + self.name,
                                            placement="bottom",
                                        ),
                                    ],
                                    style={"margin-bottom": 25},
                                ),
                                hide=self.hide_cutoff,
                            ),
                            make_hideable(
                                html.Div(
                                    [
                                        dbc.Row(
                                            [
                                                dbc.Tooltip(
                                                    "Display percentages positive and sampled"
                                                    " instead of absolute numbers",
                                                    target="liftcurve-percentage-"
                                                    + self.name,
                                                ),
                                                dbc.Checklist(
                                                    options=[
                                                        {
                                                            "label": "Display percentage",
                                                            "value": True,
                                                        }
                                                    ],
                                                    value=[True]
                                                    if self.percentage
                                                    else [],
                                                    id="liftcurve-percentage-"
                                                    + self.name,
                                                    inline=True,
                                                    switch=True,
                                                ),
                                            ]
                                        ),
                                    ]
                                ),
                                hide=self.hide_percentage,
                            ),
                            make_hideable(
                                html.Div(
                                    [
                                        dbc.Row(
                                            [
                                                dbc.Tooltip(
                                                    "Display how a perfect model would perform"
                                                    "(the so-called 'wizard')",
                                                    target="liftcurve-wizard-"
                                                    + self.name,
                                                ),
                                                dbc.Checklist(
                                                    options=[
                                                        {
                                                            "label": "Display wizard",
                                                            "value": True,
                                                        }
                                                    ],
                                                    value=[True] if self.wizard else [],
                                                    id="liftcurve-wizard-" + self.name,
                                                    inline=True,
                                                    switch=True,
                                                ),
                                            ]
                                        ),
                                    ]
                                ),
                                hide=self.hide_wizard,
                            ),
                        ]
                    ),
                    hide=self.hide_footer,
                ),
            ],
            class_name="h-100",
        )

    def to_html(self, state_dict=None, add_header=True):
        args = self.get_state_args(state_dict)
        fig = self.explainer.plot_lift_curve(
            cutoff=args["cutoff"],
            percentage=bool(args["percentage"]),
            pos_label=args["pos_label"],
            add_wizard=bool(args["wizard"]),
        )

        html = to_html.card(
            fig.to_html(include_plotlyjs="cdn", full_html=False),
            title=self.title,
            subtitle=self.subtitle,
        )
        if add_header:
            return to_html.add_header(html)
        return html

    def component_callbacks(self, app):
        @app.callback(
            Output("liftcurve-graph-" + self.name, "figure"),
            [
                Input("liftcurve-cutoff-" + self.name, "value"),
                Input("liftcurve-percentage-" + self.name, "value"),
                Input("liftcurve-wizard-" + self.name, "value"),
                Input("pos-label-" + self.name, "value"),
            ],
        )
        def update_precision_graph(cutoff, percentage, add_wizard, pos_label):
            return self.explainer.plot_lift_curve(
                cutoff=cutoff,
                percentage=bool(percentage),
                pos_label=pos_label,
                add_wizard=bool(add_wizard),
            )


class CumulativePrecisionComponent(ExplainerComponent):
    _state_props = dict(
        percentile=("cumulative-precision-percentile-", "value"),
        pos_label=("pos-label-", "value"),
    )

    def __init__(
        self,
        explainer,
        title="Cumulative Precision",
        name=None,
        subtitle="Expected distribution for highest scores",
        hide_title=False,
        hide_subtitle=False,
        hide_footer=False,
        hide_selector=False,
        hide_popout=False,
        pos_label=None,
        hide_cutoff=False,
        cutoff=None,
        hide_percentile=False,
        percentile=None,
        description=None,
        **kwargs,
    ):
        """Show cumulative precision component

        Args:
            explainer (Explainer): explainer object constructed with either
                        ClassifierExplainer() or RegressionExplainer()
            title (str, optional): Title of tab or page. Defaults to
                        "Cumulative Precision".
            name (str, optional): unique name to add to Component elements.
                        If None then random uuid is generated to make sure
                        it's unique. Defaults to None.
            subtitle (str): subtitle
            hide_title (bool, optional): hide the title.
            hide_subtitle (bool, optional): Hide subtitle. Defaults to False.
            hide_footer (bool, optional): hide the footer at the bottom of the component
            hide_selector(bool, optional): hide pos label selector. Defaults to False.
            hide_popout (bool, optional): hide popout button. Defaults to False.
            pos_label ({int, str}, optional): initial pos label. Defaults to explainer.pos_label

        """
        super().__init__(explainer, title, name)

        self.selector = PosLabelSelector(explainer, name=self.name, pos_label=pos_label)
        self.cutoff_name = "cumulative-precision-cutoff-" + self.name

        if self.description is None:
            self.description = """
        This plot shows the percentage of each label that you can expect when you
        only sample the top x% highest scores. 
        """
        self.popout = GraphPopout(
            "cumulative-precision-" + self.name + "popout",
            "cumulative-precision-graph-" + self.name,
            self.title,
            self.description,
        )
        self.register_dependencies("get_liftcurve_df")

    def layout(self):
        return dbc.Card(
            [
                make_hideable(
                    dbc.CardHeader(
                        [
                            html.Div(
                                [
                                    html.H3(
                                        self.title,
                                        id="cumulative-precision-title-" + self.name,
                                    ),
                                    make_hideable(
                                        html.H6(
                                            self.subtitle, className="card-subtitle"
                                        ),
                                        hide=self.hide_subtitle,
                                    ),
                                    dbc.Tooltip(
                                        self.description,
                                        target="cumulative-precision-title-"
                                        + self.name,
                                    ),
                                ]
                            ),
                        ]
                    ),
                    hide=self.hide_title,
                ),
                dbc.CardBody(
                    [
                        html.Div(
                            [
                                dcc.Graph(
                                    id="cumulative-precision-graph-" + self.name,
                                    config=dict(
                                        modeBarButtons=[["toImage"]], displaylogo=False
                                    ),
                                ),
                            ],
                            style={"margin": 0},
                        ),
                        dbc.Row(
                            [
                                make_hideable(
                                    dbc.Col(
                                        [self.popout.layout()], md=2, align="start"
                                    ),
                                    hide=self.hide_popout,
                                ),
                            ],
                            justify="end",
                        ),
                    ]
                ),
                make_hideable(
                    dbc.CardFooter(
                        [
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            dbc.Row(
                                                [
                                                    make_hideable(
                                                        dbc.Col(
                                                            [
                                                                html.Div(
                                                                    [
                                                                        html.Label(
                                                                            "Sample top fraction:"
                                                                        ),
                                                                        dcc.Slider(
                                                                            id="cumulative-precision-percentile-"
                                                                            + self.name,
                                                                            min=0.01,
                                                                            max=0.99,
                                                                            step=0.01,
                                                                            value=self.percentile,
                                                                            marks={
                                                                                0.01: "0.01",
                                                                                0.25: "0.25",
                                                                                0.50: "0.50",
                                                                                0.75: "0.75",
                                                                                0.99: "0.99",
                                                                            },
                                                                            included=False,
                                                                            tooltip={
                                                                                "always_visible": False
                                                                            },
                                                                            updatemode="drag",
                                                                        ),
                                                                    ],
                                                                    style={
                                                                        "margin-bottom": 15
                                                                    },
                                                                    id="cumulative-precision-percentile-div-"
                                                                    + self.name,
                                                                ),
                                                                dbc.Tooltip(
                                                                    "Draw the line where you only sample the top x% fraction of all samples",
                                                                    target="cumulative-precision-percentile-div-"
                                                                    + self.name,
                                                                ),
                                                            ]
                                                        ),
                                                        hide=self.hide_percentile,
                                                    ),
                                                ]
                                            ),
                                            dbc.Row(
                                                [
                                                    make_hideable(
                                                        dbc.Col(
                                                            [
                                                                html.Div(
                                                                    [
                                                                        html.Label(
                                                                            "Cutoff prediction probability:"
                                                                        ),
                                                                        dcc.Slider(
                                                                            id="cumulative-precision-cutoff-"
                                                                            + self.name,
                                                                            min=0.01,
                                                                            max=0.99,
                                                                            step=0.01,
                                                                            value=self.cutoff,
                                                                            marks={
                                                                                0.01: "0.01",
                                                                                0.25: "0.25",
                                                                                0.50: "0.50",
                                                                                0.75: "0.75",
                                                                                0.99: "0.99",
                                                                            },
                                                                            included=False,
                                                                            tooltip={
                                                                                "always_visible": False
                                                                            },
                                                                        ),
                                                                    ],
                                                                    style={
                                                                        "margin-bottom": 15
                                                                    },
                                                                    id="cumulative-precision-cutoff-div-"
                                                                    + self.name,
                                                                ),
                                                                dbc.Tooltip(
                                                                    f"Scores above this cutoff will be labeled positive",
                                                                    target="cumulative-precision-cutoff-div-"
                                                                    + self.name,
                                                                    placement="bottom",
                                                                ),
                                                            ]
                                                        ),
                                                        hide=self.hide_cutoff,
                                                    ),
                                                ]
                                            ),
                                        ]
                                    ),
                                    make_hideable(
                                        dbc.Col([self.selector.layout()], width=2),
                                        hide=self.hide_selector,
                                    ),
                                ]
                            )
                        ]
                    ),
                    hide=self.hide_footer,
                ),
            ],
            class_name="h-100",
        )

    def to_html(self, state_dict=None, add_header=True):
        args = self.get_state_args(state_dict)
        fig = self.explainer.plot_cumulative_precision(
            percentile=args["percentile"], pos_label=args["pos_label"]
        )

        html = to_html.card(to_html.fig(fig), title=self.title, subtitle=self.subtitle)
        if add_header:
            return to_html.add_header(html)
        return html

    def component_callbacks(self, app):
        @app.callback(
            Output("cumulative-precision-graph-" + self.name, "figure"),
            [
                Input("cumulative-precision-percentile-" + self.name, "value"),
                Input("pos-label-" + self.name, "value"),
            ],
        )
        def update_cumulative_precision_graph(percentile, pos_label):
            return self.explainer.plot_cumulative_precision(
                percentile=percentile, pos_label=pos_label
            )

        @app.callback(
            Output("cumulative-precision-percentile-" + self.name, "value"),
            [
                Input("cumulative-precision-cutoff-" + self.name, "value"),
                Input("pos-label-" + self.name, "value"),
            ],
        )
        def update_cumulative_precision_percentile(cutoff, pos_label):
            return self.explainer.percentile_from_cutoff(cutoff, pos_label)


class ClassificationComponent(ExplainerComponent):
    _state_props = dict(
        cutoff=("classification-cutoff-", "value"),
        percentage=("classification-percentage-", "value"),
        pos_label=("pos-label-", "value"),
    )

    def __init__(
        self,
        explainer,
        title="Classification Plot",
        name=None,
        subtitle="Distribution of labels above and below cutoff",
        hide_title=False,
        hide_subtitle=False,
        hide_footer=False,
        hide_cutoff=False,
        hide_percentage=False,
        hide_selector=False,
        hide_popout=False,
        pos_label=None,
        cutoff=0.5,
        percentage=True,
        description=None,
        **kwargs,
    ):
        """Shows a barchart of the number of classes above the cutoff and below
        the cutoff.

        Args:
            explainer (Explainer): explainer object constructed with either
                        ClassifierExplainer() or RegressionExplainer()
            title (str, optional): Title of tab or page. Defaults to
                        "Classification Plot".
            name (str, optional): unique name to add to Component elements.
                        If None then random uuid is generated to make sure
                        it's unique. Defaults to None.
            subtitle (str): subtitle
            hide_title (bool, optional): hide the title.
            hide_subtitle (bool, optional): Hide subtitle. Defaults to False.
            hide_footer (bool, optional): hide the footer at the bottom of the component
            hide_cutoff (bool, optional): Hide cutoff slider. Defaults to False.
            hide_percentage (bool, optional): Hide percentage toggle. Defaults to False.
            hide_selector(bool, optional): hide pos label selector. Defaults to False.
            hide_popout (bool, optional): hide popout button. Defaults to False.
            pos_label ({int, str}, optional): initial pos label. Defaults to explainer.pos_label
            cutoff (float, optional): Cutoff for prediction. Defaults to 0.5.
            percentage (bool, optional): Show percentage instead of counts. Defaults to True.
            description (str, optional): Tooltip to display when hover over
                component title. When None default text is shown.
        """
        super().__init__(explainer, title, name)

        self.cutoff_name = "classification-cutoff-" + self.name

        self.selector = PosLabelSelector(explainer, name=self.name, pos_label=pos_label)

        if self.description is None:
            self.description = """
        Plot showing the fraction of each class above and below the cutoff.
        """

        self.popout = GraphPopout(
            "classification-" + self.name + "popout",
            "classification-graph-" + self.name,
            self.title,
            self.description,
        )
        self.register_dependencies("get_classification_df")

    def layout(self):
        return dbc.Card(
            [
                make_hideable(
                    dbc.CardHeader(
                        [
                            html.Div(
                                [
                                    html.H3(
                                        self.title,
                                        id="classification-title-" + self.name,
                                    ),
                                    make_hideable(
                                        html.H6(
                                            self.subtitle, className="card-subtitle"
                                        ),
                                        hide=self.hide_subtitle,
                                    ),
                                    dbc.Tooltip(
                                        self.description,
                                        target="classification-title-" + self.name,
                                    ),
                                ]
                            ),
                        ]
                    ),
                    hide=self.hide_title,
                ),
                dbc.CardBody(
                    [
                        dbc.Row(
                            [
                                make_hideable(
                                    dbc.Col([self.selector.layout()], width=3),
                                    hide=self.hide_selector,
                                )
                            ],
                            justify="end",
                        ),
                        html.Div(
                            [
                                dcc.Graph(
                                    id="classification-graph-" + self.name,
                                    config=dict(
                                        modeBarButtons=[["toImage"]], displaylogo=False
                                    ),
                                ),
                            ],
                            style={"margin": 0},
                        ),
                        dbc.Row(
                            [
                                make_hideable(
                                    dbc.Col(
                                        [self.popout.layout()], md=2, align="start"
                                    ),
                                    hide=self.hide_popout,
                                ),
                            ],
                            justify="end",
                        ),
                    ]
                ),
                make_hideable(
                    dbc.CardFooter(
                        [
                            make_hideable(
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.Label(
                                                    "Cutoff prediction probability:"
                                                ),
                                                dcc.Slider(
                                                    id="classification-cutoff-"
                                                    + self.name,
                                                    min=0.01,
                                                    max=0.99,
                                                    step=0.01,
                                                    value=self.cutoff,
                                                    marks={
                                                        0.01: "0.01",
                                                        0.25: "0.25",
                                                        0.50: "0.50",
                                                        0.75: "0.75",
                                                        0.99: "0.99",
                                                    },
                                                    included=False,
                                                    tooltip={"always_visible": False},
                                                    updatemode="drag",
                                                ),
                                            ],
                                            id="classification-cutoff-div-" + self.name,
                                        ),
                                        dbc.Tooltip(
                                            f"Scores above this cutoff will be labeled positive",
                                            target="classification-cutoff-div-"
                                            + self.name,
                                            placement="bottom",
                                        ),
                                    ],
                                    style={"margin-bottom": 25},
                                ),
                                hide=self.hide_cutoff,
                            ),
                            make_hideable(
                                html.Div(
                                    [
                                        dbc.Row(
                                            [
                                                dbc.Tooltip(
                                                    "Do not resize the bar chart by absolute number of observations",
                                                    target="classification-percentage-"
                                                    + self.name,
                                                ),
                                                dbc.Checklist(
                                                    options=[
                                                        {
                                                            "label": "Display percentage",
                                                            "value": True,
                                                        }
                                                    ],
                                                    value=[True]
                                                    if self.percentage
                                                    else [],
                                                    id="classification-percentage-"
                                                    + self.name,
                                                    inline=True,
                                                    switch=True,
                                                ),
                                            ]
                                        ),
                                    ]
                                ),
                                hide=self.hide_percentage,
                            ),
                        ]
                    ),
                    hide=self.hide_footer,
                ),
            ],
            class_name="h-100",
        )

    def to_html(self, state_dict=None, add_header=True):
        args = self.get_state_args(state_dict)
        fig = self.explainer.plot_classification(
            cutoff=args["cutoff"],
            percentage=bool(args["percentage"]),
            pos_label=args["pos_label"],
        )

        html = to_html.card(to_html.fig(fig), title=self.title, subtitle=self.subtitle)
        if add_header:
            return to_html.add_header(html)
        return html

    def component_callbacks(self, app):
        @app.callback(
            Output("classification-graph-" + self.name, "figure"),
            [
                Input("classification-cutoff-" + self.name, "value"),
                Input("classification-percentage-" + self.name, "value"),
                Input("pos-label-" + self.name, "value"),
            ],
        )
        def update_precision_graph(cutoff, percentage, pos_label):
            return self.explainer.plot_classification(
                cutoff=cutoff, percentage=bool(percentage), pos_label=pos_label
            )


class RocAucComponent(ExplainerComponent):
    _state_props = dict(
        cutoff=("rocauc-cutoff-", "value"), pos_label=("pos-label-", "value")
    )

    def __init__(
        self,
        explainer,
        title="ROC AUC Plot",
        name=None,
        subtitle="Trade-off between False positives and false negatives",
        hide_title=False,
        hide_subtitle=False,
        hide_footer=False,
        hide_cutoff=False,
        hide_selector=False,
        hide_popout=False,
        pos_label=None,
        cutoff=0.5,
        description=None,
        **kwargs,
    ):
        """Show ROC AUC curve component

        Args:
            explainer (Explainer): explainer object constructed with either
                        ClassifierExplainer() or RegressionExplainer()
            title (str, optional): Title of tab or page. Defaults to
                        "ROC AUC Plot".
            name (str, optional): unique name to add to Component elements.
                        If None then random uuid is generated to make sure
                        it's unique. Defaults to None.
            subtitle (str): subtitle
            hide_title (bool, optional): hide title.
            hide_subtitle (bool, optional): Hide subtitle. Defaults to False.
            hide_footer (bool, optional): hide the footer at the bottom of the component
            hide_cutoff (bool, optional): Hide cutoff slider. Defaults to False.
            hide_selector(bool, optional): hide pos label selector. Defaults to False.
            hide_popout (bool, optional): hide popout button. Defaults to False.
            pos_label ({int, str}, optional): initial pos label. Defaults to explainer.pos_label
            cutoff (float, optional): default cutoff. Defaults to 0.5.

        """
        super().__init__(explainer, title, name)

        self.cutoff_name = "rocauc-cutoff-" + self.name

        self.selector = PosLabelSelector(explainer, name=self.name, pos_label=pos_label)

        if self.description is None:
            self.description = """
        """
        self.popout = GraphPopout(
            "rocauc-" + self.name + "popout",
            "rocauc-graph-" + self.name,
            self.title,
            self.description,
        )
        self.register_dependencies(
            "preds", "pred_probas", "pred_percentiles", "roc_auc_curve"
        )

    def layout(self):
        return dbc.Card(
            [
                make_hideable(
                    dbc.CardHeader(
                        [
                            html.Div(
                                [
                                    html.H3(self.title, id="rocauc-title-" + self.name),
                                    make_hideable(
                                        html.H6(
                                            self.subtitle, className="card-subtitle"
                                        ),
                                        hide=self.hide_subtitle,
                                    ),
                                    dbc.Tooltip(
                                        self.description,
                                        target="rocauc-title-" + self.name,
                                    ),
                                ]
                            ),
                        ]
                    ),
                    hide=self.hide_title,
                ),
                dbc.CardBody(
                    [
                        dbc.Row(
                            [
                                make_hideable(
                                    dbc.Col([self.selector.layout()], width=3),
                                    hide=self.hide_selector,
                                )
                            ],
                            justify="end",
                        ),
                        dcc.Graph(
                            id="rocauc-graph-" + self.name,
                            config=dict(
                                modeBarButtons=[["toImage"]], displaylogo=False
                            ),
                        ),
                        dbc.Row(
                            [
                                make_hideable(
                                    dbc.Col(
                                        [self.popout.layout()], md=2, align="start"
                                    ),
                                    hide=self.hide_popout,
                                ),
                            ],
                            justify="end",
                        ),
                    ]
                ),
                make_hideable(
                    dbc.CardFooter(
                        [
                            make_hideable(
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.Label(
                                                    "Cutoff prediction probability:"
                                                ),
                                                dcc.Slider(
                                                    id="rocauc-cutoff-" + self.name,
                                                    min=0.01,
                                                    max=0.99,
                                                    step=0.01,
                                                    value=self.cutoff,
                                                    marks={
                                                        0.01: "0.01",
                                                        0.25: "0.25",
                                                        0.50: "0.50",
                                                        0.75: "0.75",
                                                        0.99: "0.99",
                                                    },
                                                    included=False,
                                                    tooltip={"always_visible": False},
                                                    updatemode="drag",
                                                ),
                                            ],
                                            id="rocauc-cutoff-div-" + self.name,
                                        ),
                                        dbc.Tooltip(
                                            f"Scores above this cutoff will be labeled positive",
                                            target="rocauc-cutoff-div-" + self.name,
                                            placement="bottom",
                                        ),
                                    ],
                                    style={"margin-bottom": 25},
                                ),
                                hide=self.hide_cutoff,
                            ),
                        ]
                    ),
                    hide=self.hide_footer,
                ),
            ],
            class_name="h-100",
        )

    def to_html(self, state_dict=None, add_header=True):
        args = self.get_state_args(state_dict)
        fig = self.explainer.plot_roc_auc(**args)

        html = to_html.card(
            fig.to_html(include_plotlyjs="cdn", full_html=False),
            title=self.title,
            subtitle=self.subtitle,
        )
        if add_header:
            return to_html.add_header(html)
        return html

    def component_callbacks(self, app):
        @app.callback(
            Output("rocauc-graph-" + self.name, "figure"),
            [
                Input("rocauc-cutoff-" + self.name, "value"),
                Input("pos-label-" + self.name, "value"),
            ],
        )
        def update_precision_graph(cutoff, pos_label):
            return self.explainer.plot_roc_auc(cutoff=cutoff, pos_label=pos_label)


class PrAucComponent(ExplainerComponent):
    _state_props = dict(
        cutoff=("prauc-cutoff-", "value"), pos_label=("pos-label-", "value")
    )

    def __init__(
        self,
        explainer,
        title="PR AUC Plot",
        name=None,
        subtitle="Trade-off between Precision and Recall",
        hide_title=False,
        hide_subtitle=False,
        hide_footer=False,
        hide_cutoff=False,
        hide_selector=False,
        hide_popout=False,
        pos_label=None,
        cutoff=0.5,
        description=None,
        **kwargs,
    ):
        """Display PR AUC plot component

        Args:
            explainer (Explainer): explainer object constructed with either
                        ClassifierExplainer() or RegressionExplainer()
            title (str, optional): Title of tab or page. Defaults to
                        "PR AUC Plot".
            name (str, optional): unique name to add to Component elements.
                        If None then random uuid is generated to make sure
                        it's unique. Defaults to None.
            subtitle (str): subtitle
            hide_title (bool, optional): hide title.
            hide_subtitle (bool, optional): Hide subtitle. Defaults to False.
            hide_footer (bool, optional): hide the footer at the bottom of the component
            hide_cutoff (bool, optional): hide cutoff slider. Defaults to False.
            hide_selector(bool, optional): hide pos label selector. Defaults to False.
            hide_popout (bool, optional): hide popout button. Defaults to False.
            pos_label ({int, str}, optional): initial pos label. Defaults to explainer.pos_label
            cutoff (float, optional): default cutoff. Defaults to 0.5.
            description (str, optional): Tooltip to display when hover over
                component title. When None default text is shown.
        """
        super().__init__(explainer, title, name)

        self.cutoff_name = "prauc-cutoff-" + self.name

        self.selector = PosLabelSelector(explainer, name=self.name, pos_label=pos_label)

        if self.description is None:
            self.description = """
        Show the trade-off between Precision and Recall in one plot.
        """
        self.popout = GraphPopout(
            "prauc-" + self.name + "popout",
            "prauc-graph-" + self.name,
            self.title,
            self.description,
        )
        self.register_dependencies(
            "preds", "pred_probas", "pred_percentiles", "pr_auc_curve"
        )

    def layout(self):
        return dbc.Card(
            [
                make_hideable(
                    dbc.CardHeader(
                        [
                            html.Div(
                                [
                                    html.H3(self.title, id="prauc-title-" + self.name),
                                    make_hideable(
                                        html.H6(
                                            self.subtitle, className="card-subtitle"
                                        ),
                                        hide=self.hide_subtitle,
                                    ),
                                    dbc.Tooltip(
                                        self.description,
                                        target="prauc-title-" + self.name,
                                    ),
                                ]
                            ),
                        ]
                    ),
                    hide=self.hide_title,
                ),
                dbc.CardBody(
                    [
                        dbc.Row(
                            [
                                make_hideable(
                                    dbc.Col([self.selector.layout()], width=3),
                                    hide=self.hide_selector,
                                )
                            ],
                            justify="end",
                        ),
                        dcc.Graph(
                            id="prauc-graph-" + self.name,
                            config=dict(
                                modeBarButtons=[["toImage"]], displaylogo=False
                            ),
                        ),
                        dbc.Row(
                            [
                                make_hideable(
                                    dbc.Col(
                                        [self.popout.layout()], md=2, align="start"
                                    ),
                                    hide=self.hide_popout,
                                ),
                            ],
                            justify="end",
                        ),
                    ]
                ),
                make_hideable(
                    dbc.CardFooter(
                        [
                            make_hideable(
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.Label(
                                                    "Cutoff prediction probability:"
                                                ),
                                                dcc.Slider(
                                                    id="prauc-cutoff-" + self.name,
                                                    min=0.01,
                                                    max=0.99,
                                                    step=0.01,
                                                    value=self.cutoff,
                                                    marks={
                                                        0.01: "0.01",
                                                        0.25: "0.25",
                                                        0.50: "0.50",
                                                        0.75: "0.75",
                                                        0.99: "0.99",
                                                    },
                                                    included=False,
                                                    tooltip={"always_visible": False},
                                                    updatemode="drag",
                                                ),
                                            ],
                                            id="prauc-cutoff-div-" + self.name,
                                        ),
                                        dbc.Tooltip(
                                            f"Scores above this cutoff will be labeled positive",
                                            target="prauc-cutoff-div-" + self.name,
                                            placement="bottom",
                                        ),
                                    ],
                                    style={"margin-bottom": 25},
                                ),
                                hide=self.hide_cutoff,
                            ),
                        ]
                    ),
                    hide=self.hide_footer,
                ),
            ],
            class_name="h-100",
        )

    def to_html(self, state_dict=None, add_header=True):
        args = self.get_state_args(state_dict)
        fig = self.explainer.plot_pr_auc(
            cutoff=args["cutoff"], pos_label=args["pos_label"]
        )

        html = to_html.card(to_html.fig(fig), title=self.title, subtitle=self.subtitle)
        if add_header:
            return to_html.add_header(html)
        return html

    def component_callbacks(self, app):
        @app.callback(
            Output("prauc-graph-" + self.name, "figure"),
            [
                Input("prauc-cutoff-" + self.name, "value"),
                Input("pos-label-" + self.name, "value"),
            ],
        )
        def update_precision_graph(cutoff, pos_label):
            return self.explainer.plot_pr_auc(cutoff=cutoff, pos_label=pos_label)


class ClassifierModelSummaryComponent(ExplainerComponent):
    _state_props = dict(
        cutoff=("clas-model-summary-cutoff-", "value"),
        pos_label=("pos-label-", "value"),
    )

    def __init__(
        self,
        explainer,
        title="Model performance metrics",
        name=None,
        hide_title=False,
        hide_subtitle=False,
        hide_footer=False,
        hide_cutoff=False,
        hide_selector=False,
        pos_label=None,
        cutoff=0.5,
        round=3,
        show_metrics=None,
        description=None,
        **kwargs,
    ):
        """Show model summary statistics (accuracy, precision, recall,
            f1, roc_auc, pr_auc, log_loss) component.

        Args:
            explainer (Explainer): explainer object constructed with either
                        ClassifierExplainer()
            title (str, optional): Title of tab or page. Defaults to
                        "Model performance metrics".
            name (str, optional): unique name to add to Component elements.
                        If None then random uuid is generated to make sure
                        it's unique. Defaults to None.
            hide_title (bool, optional): hide title.
            hide_subtitle (bool, optional): Hide subtitle. Defaults to False.
            hide_footer (bool, optional): hide the footer at the bottom of the component
            hide_cutoff (bool, optional): hide cutoff slider. Defaults to False.
            hide_selector(bool, optional): hide pos label selector. Defaults to False.
            pos_label ({int, str}, optional): initial pos label. Defaults to explainer.pos_label
            cutoff (float, optional): default cutoff. Defaults to 0.5.
            round (int): round floats. Defaults to 3.
            show_metrics (List): list of metrics to display in order. Defaults
                to None, displaying all metrics.
            description (str, optional): Tooltip to display when hover over
                component title. When None default text is shown.
        """
        super().__init__(explainer, title, name)

        self.cutoff_name = "clas-model-summary-cutoff-" + self.name

        self.selector = PosLabelSelector(explainer, name=self.name, pos_label=pos_label)
        if self.description is None:
            self.description = """
        Shows a list of various performance metrics.
        """

        self.register_dependencies(["preds", "pred_probas"])

    def layout(self):
        return dbc.Card(
            [
                make_hideable(
                    dbc.CardHeader(
                        [
                            html.Div(
                                [
                                    html.H3(
                                        self.title,
                                        id="clas-model-summary-title-" + self.name,
                                    ),
                                    dbc.Tooltip(
                                        self.description,
                                        target="clas-model-summary-title-" + self.name,
                                    ),
                                ]
                            ),
                        ]
                    ),
                    hide=self.hide_title,
                ),
                dbc.CardBody(
                    [
                        dbc.Row(
                            [
                                make_hideable(
                                    dbc.Col([self.selector.layout()], width=3),
                                    hide=self.hide_selector,
                                )
                            ],
                            justify="end",
                        ),
                        html.Div(id="clas-model-summary-div-" + self.name),
                    ]
                ),
                make_hideable(
                    dbc.CardFooter(
                        [
                            make_hideable(
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.Label(
                                                    "Cutoff prediction probability:"
                                                ),
                                                dcc.Slider(
                                                    id="clas-model-summary-cutoff-"
                                                    + self.name,
                                                    min=0.01,
                                                    max=0.99,
                                                    step=0.01,
                                                    value=self.cutoff,
                                                    marks={
                                                        0.01: "0.01",
                                                        0.25: "0.25",
                                                        0.50: "0.50",
                                                        0.75: "0.75",
                                                        0.99: "0.99",
                                                    },
                                                    included=False,
                                                    tooltip={"always_visible": False},
                                                    updatemode="drag",
                                                ),
                                            ],
                                            id="clas-model-summary-cutoff-div-"
                                            + self.name,
                                        ),
                                        dbc.Tooltip(
                                            f"Scores above this cutoff will be labeled positive",
                                            target="clas-model-summary-cutoff-div-"
                                            + self.name,
                                            placement="bottom",
                                        ),
                                    ],
                                    style={"margin-bottom": 25},
                                ),
                                hide=self.hide_cutoff,
                            ),
                        ]
                    ),
                    hide=self.hide_footer,
                ),
            ],
            class_name="h-100",
        )

    def to_html(self, state_dict=None, add_header=True):
        args = self.get_state_args(state_dict)
        metrics_df = self._get_metrics_df(args["cutoff"], args["pos_label"])
        html = to_html.table_from_df(metrics_df)
        html = to_html.card(html, title=self.title)
        if add_header:
            return to_html.add_header(html)
        return html

    def _get_metrics_df(self, cutoff, pos_label):
        metrics_df = (
            pd.DataFrame(
                self.explainer.metrics(
                    cutoff=cutoff, pos_label=pos_label, show_metrics=self.show_metrics
                ),
                index=["Score"],
            )
            .T.rename_axis(index="metric")
            .reset_index()
            .round(self.round)
        )
        return metrics_df

    def component_callbacks(self, app):
        @app.callback(
            Output("clas-model-summary-div-" + self.name, "children"),
            [
                Input("clas-model-summary-cutoff-" + self.name, "value"),
                Input("pos-label-" + self.name, "value"),
            ],
        )
        def update_classifier_summary(cutoff, pos_label):
            metrics_df = self._get_metrics_df(cutoff, pos_label)
            metrics_table = dbc.Table.from_dataframe(
                metrics_df, striped=False, bordered=False, hover=False
            )
            metrics_dict = self.explainer.metrics_descriptions(cutoff, pos_label)
            metrics_table, tooltips = get_dbc_tooltips(
                metrics_table, metrics_dict, "clas-model-summary-div-hover", self.name
            )
            return html.Div([metrics_table, *tooltips])
