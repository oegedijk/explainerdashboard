__all__ = [
    "ShapSummaryComponent",
    "ShapDependenceComponent",
    "ShapSummaryDependenceConnector",
    "InteractionSummaryComponent",
    "InteractionDependenceComponent",
    "InteractionSummaryDependenceConnector",
    "ShapContributionsTableComponent",
    "ShapContributionsGraphComponent",
]


import dash
from dash import html, dcc, Input, Output, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

from ..dashboard_methods import *
from .. import to_html


class ShapSummaryComponent(ExplainerComponent):
    _state_props = dict(
        summary_type=("shap-summary-type-", "value"),
        depth=("shap-summary-depth-", "value"),
        index=("shap-summary-index-", "value"),
        pos_label=("pos-label-", "value"),
    )

    def __init__(
        self,
        explainer,
        title="Shap Summary",
        name=None,
        subtitle="Ordering features by shap value",
        hide_title=False,
        hide_subtitle=False,
        hide_depth=False,
        hide_type=False,
        hide_index=False,
        hide_selector=False,
        hide_popout=False,
        pos_label=None,
        depth=None,
        summary_type="aggregate",
        max_cat_colors=5,
        index=None,
        plot_sample=None,
        description=None,
        **kwargs,
    ):
        """Shows shap summary component

        Args:
            explainer (Explainer): explainer object constructed with either
                        ClassifierExplainer() or RegressionExplainer()
            title (str, optional): Title of tab or page. Defaults to
                        "Shap Dependence Summary".
            name (str, optional): unique name to add to Component elements.
                        If None then random uuid is generated to make sure
                        it's unique. Defaults to None.
            subtitle (str): subtitle
            hide_title (bool, optional): hide the title. Defaults to False.
            hide_subtitle (bool, optional): Hide subtitle. Defaults to False.
            hide_depth (bool, optional): hide the depth toggle.
                        Defaults to False.
            hide_type (bool, optional): hide the summary type toggle
                        (aggregated, detailed). Defaults to False.
            hide_popout (bool, optional): hide popout button
            hide_selector (bool, optional): hide pos label selector. Defaults to False.
            pos_label ({int, str}, optional): initial pos label.
                        Defaults to explainer.pos_label
            depth (int, optional): initial number of features to show. Defaults to None.
            summary_type (str, {'aggregate', 'detailed'}. optional): type of
                        summary graph to show. Defaults to "aggregate".
            max_cat_colors (int, optional): for categorical features, maximum number
                of categories to label with own color. Defaults to 5.
            plot_sample (int, optional): Instead of all points only plot a random
                sample of points. Defaults to None (=all points)
            description (str, optional): Tooltip to display when hover over
                component title. When None default text is shown.
        """
        super().__init__(explainer, title, name)

        if self.depth is not None:
            self.depth = min(self.depth, self.explainer.n_features)

        self.index_selector = IndexSelector(
            explainer, "shap-summary-index-" + self.name, index=index, **kwargs
        )
        self.index_name = "shap-summary-index-" + self.name
        self.selector = PosLabelSelector(explainer, name=self.name, pos_label=pos_label)
        assert self.summary_type in {"aggregate", "detailed"}
        if self.description is None:
            self.description = """
        The shap summary summarizes the shap values per feature.
        You can either select an aggregates display that shows mean absolute shap value
        per feature. Or get a more detailed look at the spread of shap values per
        feature and how they correlate the the feature value (red is high).
        """

        self.popout = GraphPopout(
            "shap-summary-" + self.name + "popout",
            "shap-summary-graph-" + self.name,
            self.title,
            self.description,
        )
        self.register_dependencies("shap_values_df")

    def layout(self):
        return dbc.Card(
            [
                make_hideable(
                    dbc.CardHeader(
                        [
                            html.Div(
                                [
                                    html.H3(
                                        self.title, id="shap-summary-title-" + self.name
                                    ),
                                    make_hideable(
                                        html.H6(
                                            self.subtitle, className="card-subtitle"
                                        ),
                                        hide=self.hide_subtitle,
                                    ),
                                    dbc.Tooltip(
                                        self.description,
                                        target="shap-summary-title-" + self.name,
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
                                            dbc.Label(
                                                "Depth:",
                                                id="shap-summary-depth-label-"
                                                + self.name,
                                            ),
                                            dbc.Tooltip(
                                                "Number of features to display",
                                                target="shap-summary-depth-label-"
                                                + self.name,
                                            ),
                                            dbc.Select(
                                                id="shap-summary-depth-" + self.name,
                                                options=[
                                                    {
                                                        "label": str(i + 1),
                                                        "value": i + 1,
                                                    }
                                                    for i in range(
                                                        self.explainer.n_features
                                                    )
                                                ],
                                                size="sm",
                                                value=self.depth,
                                            ),
                                        ],
                                        md=2,
                                    ),
                                    self.hide_depth,
                                ),
                                make_hideable(
                                    dbc.Col(
                                        [
                                            dbc.Row(
                                                [
                                                    dbc.Label(
                                                        "Summary Type",
                                                        id="shap-summary-type-label-"
                                                        + self.name,
                                                    ),
                                                    dbc.Tooltip(
                                                        "Display mean absolute SHAP value per feature (aggregate)"
                                                        " or display every single shap value per feature (detailed)",
                                                        target="shap-summary-type-label-"
                                                        + self.name,
                                                    ),
                                                    dbc.Select(
                                                        options=[
                                                            {
                                                                "label": "Aggregate",
                                                                "value": "aggregate",
                                                            },
                                                            {
                                                                "label": "Detailed",
                                                                "value": "detailed",
                                                            },
                                                        ],
                                                        value=self.summary_type,
                                                        size="sm",
                                                        id="shap-summary-type-"
                                                        + self.name,
                                                    ),
                                                ]
                                            )
                                        ]
                                    ),
                                    self.hide_type,
                                ),
                                make_hideable(
                                    dbc.Col(
                                        [
                                            html.Div(
                                                [
                                                    dbc.Label(
                                                        f"{self.explainer.index_name}:",
                                                        id="shap-summary-index-label-"
                                                        + self.name,
                                                    ),
                                                    dbc.Tooltip(
                                                        f"Select {self.explainer.index_name} to highlight in plot. "
                                                        "You can also select by clicking on a scatter point in the graph.",
                                                        target="shap-summary-index-label-"
                                                        + self.name,
                                                    ),
                                                    self.index_selector.layout(),
                                                ],
                                                id="shap-summary-index-col-"
                                                + self.name,
                                                style=dict(display="none"),
                                            ),
                                        ],
                                        md=3,
                                    ),
                                    hide=self.hide_index,
                                ),
                                make_hideable(
                                    dbc.Col([self.selector.layout()], width=2),
                                    hide=self.hide_selector,
                                ),
                            ]
                        ),
                        dcc.Loading(
                            id="loading-dependence-shap-summary-" + self.name,
                            children=[
                                dcc.Graph(
                                    id="shap-summary-graph-" + self.name,
                                    config=dict(
                                        modeBarButtons=[["toImage"]], displaylogo=False
                                    ),
                                )
                            ],
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
            ],
            class_name="h-100",
        )

    def to_html(self, state_dict=None, add_header=True):
        args = self.get_state_args(state_dict)
        summary_type = args.pop("summary_type")
        if summary_type == "aggregate":
            fig = self.explainer.plot_importances(
                kind="shap", topx=args["depth"], pos_label=args["pos_label"]
            )
        elif summary_type == "detailed":
            fig = self.explainer.plot_importances_detailed(
                topx=args["depth"],
                pos_label=args["pos_label"],
                highlight_index=args["index"],
                max_cat_colors=self.max_cat_colors,
                plot_sample=self.plot_sample,
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
            Output("shap-summary-index-" + self.name, "value"),
            [Input("shap-summary-graph-" + self.name, "clickData")],
        )
        def display_scatter_click_data(clickdata):
            if clickdata is not None and clickdata["points"][0] is not None:
                if isinstance(clickdata["points"][0]["y"], float):  # detailed
                    index = (
                        clickdata["points"][0]["text"].split("=")[1].split("<br>")[0]
                    )
                    return index
            raise PreventUpdate

        @app.callback(
            [
                Output("shap-summary-graph-" + self.name, "figure"),
                Output("shap-summary-index-col-" + self.name, "style"),
            ],
            [
                Input("shap-summary-type-" + self.name, "value"),
                Input("shap-summary-depth-" + self.name, "value"),
                Input("shap-summary-index-" + self.name, "value"),
                Input("pos-label-" + self.name, "value"),
            ],
        )
        def update_shap_summary_graph(summary_type, depth, index, pos_label):
            depth = None if depth is None else int(depth)
            if summary_type == "aggregate":
                plot = self.explainer.plot_importances(
                    kind="shap", topx=depth, pos_label=pos_label
                )
            elif summary_type == "detailed":
                plot = self.explainer.plot_importances_detailed(
                    topx=depth,
                    pos_label=pos_label,
                    highlight_index=index,
                    max_cat_colors=self.max_cat_colors,
                    plot_sample=self.plot_sample,
                )
            else:
                raise PreventUpdate

            ctx = dash.callback_context
            trigger = ctx.triggered[0]["prop_id"].split(".")[0]
            if trigger == "shap-summary-type-" + self.name:
                if summary_type == "aggregate":
                    return (plot, dict(display="none"))
                elif summary_type == "detailed":
                    return (plot, {})
            else:
                return (plot, dash.no_update)


class ShapDependenceComponent(ExplainerComponent):
    _state_props = dict(
        col=("shap-dependence-col-", "value"),
        color_col=("shap-dependence-color-col-", "value"),
        index=("shap-dependence-index-", "value"),
        cats_topx=("shap-dependence-n-categories-", "value"),
        cats_sort=("shap-dependence-categories-sort-", "value"),
        remove_outliers=("shap-dependence-outliers-", "value"),
        pos_label=("pos-label-", "value"),
    )

    def __init__(
        self,
        explainer,
        title="Shap Dependence",
        name=None,
        subtitle="Relationship between feature value and SHAP value",
        hide_title=False,
        hide_subtitle=False,
        hide_col=False,
        hide_color_col=False,
        hide_index=False,
        hide_selector=False,
        hide_outliers=False,
        hide_cats_topx=False,
        hide_cats_sort=False,
        hide_popout=False,
        hide_footer=False,
        pos_label=None,
        col=None,
        color_col=None,
        index=None,
        remove_outliers=False,
        cats_topx=10,
        cats_sort="freq",
        max_cat_colors=5,
        plot_sample=None,
        description=None,
        **kwargs,
    ):
        """Show shap dependence graph

        Args:
            explainer (Explainer): explainer object constructed with either
                        ClassifierExplainer() or RegressionExplainer()
            title (str, optional): Title of tab or page. Defaults to
                        "Shap Dependence".
            name (str, optional): unique name to add to Component elements.
                        If None then random uuid is generated to make sure
                        it's unique. Defaults to None.
            subtitle (str): subtitle
            hide_title (bool, optional): hide component title. Defaults to False.
            hide_subtitle (bool, optional): Hide subtitle. Defaults to False.
            hide_col (bool, optional): hide feature selector. Defaults to False.
            hide_color_col (bool, optional): hide color feature selector Defaults to False.
            hide_index (bool, optional): hide index selector Defaults to False.
            hide_selector (bool, optional): hide pos label selector. Defaults to False.
            hide_cats_topx (bool, optional): hide the categories topx input. Defaults to False.
            hide_cats_sort (bool, optional): hide the categories sort selector.Defaults to False.
            hide_outliers (bool, optional): Hide remove outliers toggle input. Defaults to False.
            hide_popout (bool, optional): hide popout button. Defaults to False.
            hide_footer (bool, optional): hide the footer.
            pos_label ({int, str}, optional): initial pos label.
                        Defaults to explainer.pos_label
            col (str, optional): Feature to display. Defaults to None.
            color_col (str, optional): Color plot by values of this Feature.
                        Defaults to None.
            index (int, optional): Highlight a particular index. Defaults to None.
            remove_outliers (bool, optional): remove outliers in feature and
                color feature from the plot.
            cats_topx (int, optional): maximum number of categories to display
                for categorical features. Defaults to 10.
            cats_sort (str, optional): how to sort categories: 'alphabet',
                'freq' or 'shap'. Defaults to 'freq'.
            max_cat_colors (int, optional): for categorical features, maximum number
                of categories to label with own color. Defaults to 5.
            plot_sample (int, optional): Instead of all points only plot a random
                sample of points. Defaults to None (=all points)
            description (str, optional): Tooltip to display when hover over
                component title. When None default text is shown.
        """
        super().__init__(explainer, title, name)

        if self.col is None:
            self.col = self.explainer.columns_ranked_by_shap()[0]
        if self.color_col is None:
            self.color_col = self.explainer.top_shap_interactions(self.col)[1]

        self.selector = PosLabelSelector(explainer, name=self.name, pos_label=pos_label)

        self.index_selector = IndexSelector(
            explainer, "shap-dependence-index-" + self.name, index=index, **kwargs
        )
        self.index_name = "shap-dependence-index-" + self.name

        if self.description is None:
            self.description = """
        This plot shows the relation between feature values and shap values.
        This allows you to investigate the general relationship between feature
        value and impact on the prediction. You can check whether the model
        uses features in line with your intuitions, or use the plots to learn
        about the relationships that the model has learned between the input features
        and the predicted outcome.
        """
        self.popout = GraphPopout(
            "shap-dependence-" + self.name + "popout",
            "shap-dependence-graph-" + self.name,
            self.title,
            self.description,
        )
        self.register_dependencies("shap_values_df")

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
                                        id="shap-dependence-title-" + self.name,
                                    ),
                                    make_hideable(
                                        html.H6(
                                            self.subtitle, className="card-subtitle"
                                        ),
                                        hide=self.hide_subtitle,
                                    ),
                                    dbc.Tooltip(
                                        self.description,
                                        target="shap-dependence-title-" + self.name,
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
                                    dbc.Col([self.selector.layout()], width=2),
                                    hide=self.hide_selector,
                                ),
                            ]
                        ),
                        dbc.Row(
                            [
                                make_hideable(
                                    dbc.Col(
                                        [
                                            dbc.Label(
                                                "Feature:",
                                                id="shap-dependence-col-label-"
                                                + self.name,
                                            ),
                                            dbc.Tooltip(
                                                "Select feature to display shap dependence for",
                                                target="shap-dependence-col-label-"
                                                + self.name,
                                            ),
                                            dbc.Select(
                                                id="shap-dependence-col-" + self.name,
                                                options=[
                                                    {"label": col, "value": col}
                                                    for col in self.explainer.columns_ranked_by_shap()
                                                ],
                                                value=self.col,
                                                size="sm",
                                            ),
                                        ],
                                        md=3,
                                    ),
                                    self.hide_col,
                                ),
                                make_hideable(
                                    dbc.Col(
                                        [
                                            dbc.Label(
                                                "Color feature:",
                                                id="shap-dependence-color-col-label-"
                                                + self.name,
                                            ),
                                            dbc.Tooltip(
                                                "Select feature to color the scatter markers by. This "
                                                "allows you to see interactions between various features in the graph.",
                                                target="shap-dependence-color-col-label-"
                                                + self.name,
                                            ),
                                            dbc.Select(
                                                id="shap-dependence-color-col-"
                                                + self.name,
                                                options=[
                                                    {"label": col, "value": col}
                                                    for col in self.explainer.columns_ranked_by_shap()
                                                ]
                                                + [
                                                    dict(
                                                        label="None",
                                                        value="no_color_col",
                                                    )
                                                ],
                                                value=self.color_col,
                                                size="sm",
                                            ),
                                        ],
                                        md=3,
                                    ),
                                    self.hide_color_col,
                                ),
                                make_hideable(
                                    dbc.Col(
                                        [
                                            dbc.Label(
                                                f"{self.explainer.index_name}:",
                                                id="shap-dependence-index-label-"
                                                + self.name,
                                            ),
                                            dbc.Tooltip(
                                                f"Select {self.explainer.index_name} to highlight in the plot."
                                                "You can also select by clicking on a scatter marker in the accompanying"
                                                " shap summary plot (detailed).",
                                                target="shap-dependence-index-label-"
                                                + self.name,
                                            ),
                                            self.index_selector.layout(),
                                        ],
                                        md=4,
                                    ),
                                    hide=self.hide_index,
                                ),
                            ]
                        ),
                        dcc.Loading(
                            id="loading-dependence-graph-" + self.name,
                            children=[
                                dcc.Graph(
                                    id="shap-dependence-graph-" + self.name,
                                    config=dict(
                                        modeBarButtons=[["toImage"]], displaylogo=False
                                    ),
                                )
                            ],
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
                                    make_hideable(
                                        dbc.Col(
                                            [
                                                dbc.Row(
                                                    [
                                                        dbc.Tooltip(
                                                            "Remove outliers in feature (and color feature) from plot.",
                                                            target="shap-dependence-outliers-"
                                                            + self.name,
                                                        ),
                                                        dbc.Checklist(
                                                            options=[
                                                                {
                                                                    "label": "Remove outliers",
                                                                    "value": True,
                                                                }
                                                            ],
                                                            value=[True]
                                                            if self.remove_outliers
                                                            else [],
                                                            id="shap-dependence-outliers-"
                                                            + self.name,
                                                            inline=True,
                                                            switch=True,
                                                        ),
                                                    ]
                                                ),
                                            ],
                                            md=2,
                                        ),
                                        hide=self.hide_outliers,
                                    ),
                                    make_hideable(
                                        dbc.Col(
                                            [
                                                html.Div(
                                                    [
                                                        dbc.Label(
                                                            "Categories:",
                                                            id="shap-dependence-n-categories-label-"
                                                            + self.name,
                                                        ),
                                                        dbc.Tooltip(
                                                            "Maximum number of categories to display",
                                                            target="shap-dependence-n-categories-label-"
                                                            + self.name,
                                                        ),
                                                        dbc.Input(
                                                            id="shap-dependence-n-categories-"
                                                            + self.name,
                                                            value=self.cats_topx,
                                                            type="number",
                                                            min=1,
                                                            max=50,
                                                            step=1,
                                                        ),
                                                    ],
                                                    id="shap-dependence-categories-div1-"
                                                    + self.name,
                                                    style={}
                                                    if self.col
                                                    in self.explainer.cat_cols
                                                    else dict(display="none"),
                                                )
                                            ],
                                            md=2,
                                        ),
                                        self.hide_cats_topx,
                                    ),
                                    make_hideable(
                                        dbc.Col(
                                            [
                                                html.Div(
                                                    [
                                                        html.Label(
                                                            "Sort categories:",
                                                            id="shap-dependence-categories-sort-label-"
                                                            + self.name,
                                                        ),
                                                        dbc.Tooltip(
                                                            "How to sort the categories: Alphabetically, most common "
                                                            "first (Frequency), or highest mean absolute SHAP value first (Shap impact)",
                                                            target="shap-dependence-categories-sort-label-"
                                                            + self.name,
                                                        ),
                                                        dbc.Select(
                                                            id="shap-dependence-categories-sort-"
                                                            + self.name,
                                                            options=[
                                                                {
                                                                    "label": "Alphabetically",
                                                                    "value": "alphabet",
                                                                },
                                                                {
                                                                    "label": "Frequency",
                                                                    "value": "freq",
                                                                },
                                                                {
                                                                    "label": "Shap impact",
                                                                    "value": "shap",
                                                                },
                                                            ],
                                                            value=self.cats_sort,
                                                            size="sm",
                                                        ),
                                                    ],
                                                    id="shap-dependence-categories-div2-"
                                                    + self.name,
                                                    style={}
                                                    if self.col
                                                    in self.explainer.cat_cols
                                                    else dict(display="none"),
                                                )
                                            ],
                                            md=4,
                                        ),
                                        hide=self.hide_cats_sort,
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

        if args["color_col"] == "no_color_col":
            args["color_col"], args["index"] = None, None
        fig = self.explainer.plot_dependence(
            args["col"],
            args["color_col"],
            topx=args["cats_topx"],
            sort=args["cats_sort"],
            highlight_index=args["index"],
            max_cat_colors=self.max_cat_colors,
            plot_sample=self.plot_sample,
            remove_outliers=bool(args["remove_outliers"]),
            pos_label=args["pos_label"],
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
            [
                Output("shap-dependence-color-col-" + self.name, "options"),
                Output("shap-dependence-color-col-" + self.name, "value"),
                Output("shap-dependence-categories-div1-" + self.name, "style"),
                Output("shap-dependence-categories-div2-" + self.name, "style"),
            ],
            [Input("shap-dependence-col-" + self.name, "value")],
            [State("pos-label-" + self.name, "value")],
        )
        def set_color_col_dropdown(col, pos_label):
            sorted_interact_cols = self.explainer.top_shap_interactions(
                col, pos_label=pos_label
            )
            options = [{"label": col, "value": col} for col in sorted_interact_cols] + [
                dict(label="None", value="no_color_col")
            ]
            if col in self.explainer.cat_cols:
                value = None
                style = dict()
            else:
                value = sorted_interact_cols[1]
                style = dict(display="none")
            return (options, value, style, style)

        @app.callback(
            Output("shap-dependence-graph-" + self.name, "figure"),
            [
                Input("shap-dependence-color-col-" + self.name, "value"),
                Input("shap-dependence-index-" + self.name, "value"),
                Input("shap-dependence-n-categories-" + self.name, "value"),
                Input("shap-dependence-categories-sort-" + self.name, "value"),
                Input("shap-dependence-outliers-" + self.name, "value"),
                Input("pos-label-" + self.name, "value"),
            ],
            [State("shap-dependence-col-" + self.name, "value")],
        )
        def update_dependence_graph(
            color_col, index, topx, sort, remove_outliers, pos_label, col
        ):
            if col is not None:
                if color_col == "no_color_col":
                    color_col, index = None, None
                return self.explainer.plot_dependence(
                    col,
                    color_col,
                    topx=topx,
                    sort=sort,
                    highlight_index=index,
                    max_cat_colors=self.max_cat_colors,
                    plot_sample=self.plot_sample,
                    remove_outliers=bool(remove_outliers),
                    pos_label=pos_label,
                )
            raise PreventUpdate


class ShapSummaryDependenceConnector(ExplainerComponent):
    def __init__(self, shap_summary_component, shap_dependence_component):
        """Connects a ShapSummaryComponent with a ShapDependence Component:

        - When clicking on feature in ShapSummary, then select that feature in ShapDependence

        Args:
            shap_summary_component (ShapSummaryComponent): ShapSummaryComponent
            shap_dependence_component (ShapDependenceComponent): ShapDependenceComponent
        """
        self.sum_name = shap_summary_component.name
        self.dep_name = shap_dependence_component.name

    def component_callbacks(self, app):
        @app.callback(
            [
                Output("shap-dependence-index-" + self.dep_name, "value"),
                Output("shap-dependence-col-" + self.dep_name, "value"),
            ],
            [Input("shap-summary-graph-" + self.sum_name, "clickData")],
        )
        def display_scatter_click_data(clickdata):
            if clickdata is not None and clickdata["points"][0] is not None:
                if isinstance(clickdata["points"][0]["y"], float):  # detailed
                    index = (
                        clickdata["points"][0]["text"].split("=")[1].split("<br>")[0]
                    )
                    col = clickdata["points"][0]["text"].split("=")[1].split("<br>")[1]
                    return (index, col)
                elif isinstance(clickdata["points"][0]["y"], str):  # aggregate
                    # in aggregate clickdata returns col name -> type==str
                    col = clickdata["points"][0]["y"].split(" ")[1]
                    return (dash.no_update, col)
            raise PreventUpdate


class InteractionSummaryComponent(ExplainerComponent):
    _state_props = dict(
        col=("interaction-summary-col-", "value"),
        depth=("interaction-summary-depth-", "value"),
        summary_type=("interaction-summary-type-", "value"),
        index=("interaction-summary-index-", "value"),
        pos_label=("pos-label-", "value"),
    )

    def __init__(
        self,
        explainer,
        title="Interactions Summary",
        name=None,
        subtitle="Ordering features by shap interaction value",
        hide_title=False,
        hide_subtitle=False,
        hide_col=False,
        hide_depth=False,
        hide_type=False,
        hide_index=False,
        hide_popout=False,
        hide_selector=False,
        pos_label=None,
        col=None,
        depth=None,
        summary_type="aggregate",
        max_cat_colors=5,
        index=None,
        plot_sample=None,
        description=None,
        **kwargs,
    ):
        """Show SHAP Interaciton values summary component

        Args:
            explainer (Explainer): explainer object constructed with either
                        ClassifierExplainer() or RegressionExplainer()
            title (str, optional): Title of tab or page. Defaults to
                        "Interactions Summary".
            name (str, optional): unique name to add to Component elements.
                        If None then random uuid is generated to make sure
                        it's unique. Defaults to None.
            subtitle (str): subtitle
            hide_title (bool, optional): hide the component title. Defaults to False.
            hide_subtitle (bool, optional): Hide subtitle. Defaults to False.
            hide_col (bool, optional): Hide the feature selector. Defaults to False.
            hide_depth (bool, optional): Hide depth toggle. Defaults to False.
            hide_type (bool, optional): Hide summary type toggle. Defaults to False.
            hide_index (bool, optional): Hide the index selector. Defaults to False
            hide_popout (bool, optional): hide popout button
            hide_selector (bool, optional): hide pos label selector. Defaults to False.
            pos_label ({int, str}, optional): initial pos label.
                        Defaults to explainer.pos_label
            col (str, optional): Feature to show interaction summary for.
                Defaults to None.
            depth (int, optional): Number of interaction features to display.
                Defaults to None.
            summary_type (str, {'aggregate', 'detailed'}, optional): type of
                summary graph to display. Defaults to "aggregate".
            max_cat_colors (int, optional): for categorical features, maximum number
                of categories to label with own color. Defaults to 5.
            index (str):    Default index. Defaults to None.
            plot_sample (int, optional): Instead of all points only plot a random
                sample of points. Defaults to None (=all points)
            description (str, optional): Tooltip to display when hover over
                component title. When None default text is shown.
        """
        super().__init__(explainer, title, name)

        if self.col is None:
            self.col = self.explainer.columns_ranked_by_shap()[0]
        if self.depth is not None:
            self.depth = min(self.depth, self.explainer.n_features - 1)

        self.index_selector = IndexSelector(
            explainer, "interaction-summary-index-" + self.name, index=index, **kwargs
        )
        self.index_name = "interaction-summary-index-" + self.name
        self.selector = PosLabelSelector(explainer, name=self.name, pos_label=pos_label)

        if self.description is None:
            self.description = """
        Shows shap interaction values. Each shap value can be decomposed into a direct
        effect and indirect effects. The indirect effects are due to interactions
        of the feature with other feature. For example the fact that you know
        the gender of a passenger on the titanic will have a direct effect (women
        more likely to survive then men), but may also have indirect effects through
        for example passenger class (first class women more likely to survive than
        average woman, third class women less likely).
        """
        self.popout = GraphPopout(
            "interaction-summary-" + self.name + "popout",
            "interaction-summary-graph-" + self.name,
            self.title,
            self.description,
        )
        self.register_dependencies("shap_interaction_values")

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
                                        id="interaction-summary-title-" + self.name,
                                    ),
                                    make_hideable(
                                        html.H6(
                                            self.subtitle, className="card-subtitle"
                                        ),
                                        hide=self.hide_subtitle,
                                    ),
                                    dbc.Tooltip(
                                        self.description,
                                        target="interaction-summary-title-" + self.name,
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
                                            dbc.Label(
                                                "Feature",
                                                id="interaction-summary-col-label-"
                                                + self.name,
                                            ),
                                            dbc.Tooltip(
                                                "Feature to select interactions effects for",
                                                target="interaction-summary-col-label-"
                                                + self.name,
                                            ),
                                            dbc.Select(
                                                id="interaction-summary-col-"
                                                + self.name,
                                                options=[
                                                    {"label": col, "value": col}
                                                    for col in self.explainer.columns_ranked_by_shap()
                                                ],
                                                value=self.col,
                                            ),
                                        ],
                                        md=2,
                                    ),
                                    self.hide_col,
                                ),
                                make_hideable(
                                    dbc.Col(
                                        [
                                            dbc.Label(
                                                "Depth:",
                                                id="interaction-summary-depth-label-"
                                                + self.name,
                                            ),
                                            dbc.Tooltip(
                                                "Number of interaction features to display",
                                                target="interaction-summary-depth-label-"
                                                + self.name,
                                            ),
                                            dbc.Select(
                                                id="interaction-summary-depth-"
                                                + self.name,
                                                options=[
                                                    {
                                                        "label": str(i + 1),
                                                        "value": i + 1,
                                                    }
                                                    for i in range(
                                                        self.explainer.n_features - 1
                                                    )
                                                ],
                                                value=self.depth,
                                            ),
                                        ],
                                        md=2,
                                    ),
                                    self.hide_depth,
                                ),
                                make_hideable(
                                    dbc.Col(
                                        [
                                            dbc.Row(
                                                [
                                                    dbc.Label(
                                                        "Summary Type",
                                                        id="interaction-summary-type-label-"
                                                        + self.name,
                                                    ),
                                                    dbc.Tooltip(
                                                        "Display mean absolute SHAP value per feature (aggregate)"
                                                        " or display every single shap value per feature (detailed)",
                                                        target="interaction-summary-type-label-"
                                                        + self.name,
                                                    ),
                                                    dbc.Select(
                                                        options=[
                                                            {
                                                                "label": "Aggregate",
                                                                "value": "aggregate",
                                                            },
                                                            {
                                                                "label": "Detailed",
                                                                "value": "detailed",
                                                            },
                                                        ],
                                                        value=self.summary_type,
                                                        id="interaction-summary-type-"
                                                        + self.name,
                                                    ),
                                                ]
                                            )
                                        ],
                                        md=3,
                                    ),
                                    self.hide_type,
                                ),
                                make_hideable(
                                    dbc.Col(
                                        [
                                            html.Div(
                                                [
                                                    dbc.Label(
                                                        f"{self.explainer.index_name}:",
                                                        id="interaction-summary-index-label-"
                                                        + self.name,
                                                    ),
                                                    dbc.Tooltip(
                                                        f"Select {self.explainer.index_name} to highlight in plot. "
                                                        "You can also select by clicking on a scatter point in the graph.",
                                                        target="interaction-summary-index-label-"
                                                        + self.name,
                                                    ),
                                                    self.index_selector.layout(),
                                                ],
                                                id="interaction-summary-index-col-"
                                                + self.name,
                                                style=dict(display="none"),
                                            ),
                                        ],
                                        md=3,
                                    ),
                                    hide=self.hide_index,
                                ),
                                make_hideable(
                                    dbc.Col([self.selector.layout()], width=2),
                                    hide=self.hide_selector,
                                ),
                            ]
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        dcc.Loading(
                                            id="loading-interaction-summary-graph-"
                                            + self.name,
                                            children=[
                                                dcc.Graph(
                                                    id="interaction-summary-graph-"
                                                    + self.name,
                                                    config=dict(
                                                        modeBarButtons=[["toImage"]],
                                                        displaylogo=False,
                                                    ),
                                                )
                                            ],
                                        )
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
            ],
            class_name="h-100",
        )

    def to_html(self, state_dict=None, add_header=True):
        args = self.get_state_args(state_dict)
        if args["summary_type"] == "aggregate":
            fig = self.explainer.plot_interactions_importance(
                args["col"], topx=args["depth"], pos_label=args["pos_label"]
            )
        else:
            fig = self.explainer.plot_interactions_detailed(
                args["col"],
                topx=args["depth"],
                pos_label=args["pos_label"],
                highlight_index=args["index"],
                max_cat_colors=self.max_cat_colors,
                plot_sample=self.plot_sample,
            )

        html = to_html.card(to_html.fig(fig), title=self.title, subtitle=self.subtitle)
        if add_header:
            return to_html.add_header(html)
        return html

    def component_callbacks(self, app):
        @app.callback(
            Output("interaction-summary-index-" + self.name, "value"),
            [Input("interaction-summary-graph-" + self.name, "clickData")],
        )
        def display_scatter_click_data(clickdata):
            if clickdata is not None and clickdata["points"][0] is not None:
                if isinstance(clickdata["points"][0]["y"], float):  # detailed
                    index = (
                        clickdata["points"][0]["text"].split("=")[1].split("<br>")[0]
                    )
                    return index
            raise PreventUpdate

        @app.callback(
            [
                Output("interaction-summary-graph-" + self.name, "figure"),
                Output("interaction-summary-index-col-" + self.name, "style"),
            ],
            [
                Input("interaction-summary-col-" + self.name, "value"),
                Input("interaction-summary-depth-" + self.name, "value"),
                Input("interaction-summary-type-" + self.name, "value"),
                Input("interaction-summary-index-" + self.name, "value"),
                Input("pos-label-" + self.name, "value"),
            ],
        )
        def update_interaction_scatter_graph(
            col, depth, summary_type, index, pos_label
        ):
            if col is not None:
                depth = None if depth is None else int(depth)
                if summary_type == "aggregate":
                    plot = self.explainer.plot_interactions_importance(
                        col, topx=depth, pos_label=pos_label
                    )
                    return plot, dict(display="none")
                elif summary_type == "detailed":
                    plot = self.explainer.plot_interactions_detailed(
                        col,
                        topx=depth,
                        pos_label=pos_label,
                        highlight_index=index,
                        max_cat_colors=self.max_cat_colors,
                        plot_sample=self.plot_sample,
                    )
                return plot, {}
            raise PreventUpdate


class InteractionDependenceComponent(ExplainerComponent):
    _state_props = dict(
        col=("interaction-dependence-col-", "value"),
        interact_col=("interaction-dependence-interact-col-", "value"),
        index=("interaction-dependence-index-", "value"),
        cats_topx=("interaction-dependence-top-n-categories-", "value"),
        cats_sort=("interaction-dependence-top-categories-sort-", "value"),
        remove_outliers=("interaction-dependence-top-outliers-", "value"),
        pos_label=("pos-label-", "value"),
    )

    def __init__(
        self,
        explainer,
        title="Interaction Dependence",
        name=None,
        subtitle="Relation between feature value and shap interaction value",
        hide_title=False,
        hide_subtitle=False,
        hide_col=False,
        hide_interact_col=False,
        hide_index=False,
        hide_popout=False,
        hide_selector=False,
        hide_outliers=False,
        hide_cats_topx=False,
        hide_cats_sort=False,
        hide_top=False,
        hide_bottom=False,
        pos_label=None,
        col=None,
        interact_col=None,
        remove_outliers=False,
        cats_topx=10,
        cats_sort="freq",
        max_cat_colors=5,
        plot_sample=None,
        description=None,
        index=None,
        **kwargs,
    ):
        """Interaction Dependence Component.

        Shows two graphs:
            top graph: col vs interact_col
            bottom graph: interact_col vs col

        Args:
            explainer (Explainer): explainer object constructed with either
                        ClassifierExplainer() or RegressionExplainer()
            title (str, optional): Title of tab or page. Defaults to
                        "Interactions Dependence".
            name (str, optional): unique name to add to Component elements.
                        If None then random uuid is generated to make sure
                        it's unique. Defaults to None.
            subtitle (str): subtitle
            hide_title (bool, optional): Hide component title. Defaults to False.
            hide_subtitle (bool, optional): Hide subtitle. Defaults to False.
            hide_col (bool, optional): Hide feature selector. Defaults to False.
            hide_interact_col (bool, optional): Hide interaction
                        feature selector. Defaults to False.
            hide_highlight (bool, optional): Hide highlight index selector.
                        Defaults to False.
            hide_selector (bool, optional): hide pos label selector.
                        Defaults to False.
            hide_outliers (bool, optional): Hide remove outliers toggle input. Defaults to False.
            hide_popout (bool, optional): hide popout button
            hide_cats_topx (bool, optional): hide the categories topx input.
                        Defaults to False.
            hide_cats_sort (bool, optional): hide the categories sort selector.
                        Defaults to False.
            hide_top (bool, optional): Hide the top interaction graph
                        (col vs interact_col). Defaults to False.
            hide_bottom (bool, optional): hide the bottom interaction graph
                        (interact_col vs col). Defaults to False.
            pos_label ({int, str}, optional): initial pos label.
                        Defaults to explainer.pos_label
            col (str, optional): Feature to find interactions for. Defaults to None.
            interact_col (str, optional): Feature to interact with. Defaults to None.
            highlight (int, optional): Index row to highlight Defaults to None.
            remove_outliers (bool, optional): remove outliers in feature and
                color feature from the plot.
            cats_topx (int, optional): number of categories to display for
                categorical features.
            cats_sort (str, optional): how to sort categories: 'alphabet',
                'freq' or 'shap'. Defaults to 'freq'.
            max_cat_colors (int, optional): for categorical features, maximum number
                of categories to label with own color. Defaults to 5.
            plot_sample (int, optional): Instead of all points only plot a random
                sample of points. Defaults to None (=all points)
            description (str, optional): Tooltip to display when hover over
                component title. When None default text is shown.
        """
        super().__init__(explainer, title, name)

        if self.col is None:
            self.col = explainer.columns_ranked_by_shap()[0]
        if self.interact_col is None:
            self.interact_col = explainer.top_shap_interactions(self.col)[1]

        self.index_selector = IndexSelector(
            explainer,
            "interaction-dependence-index-" + self.name,
            index=index,
            **kwargs,
        )
        self.index_name = "interaction-dependence-index-" + self.name

        self.selector = PosLabelSelector(explainer, name=self.name, pos_label=pos_label)
        self.popout_top = GraphPopout(
            self.name + "popout-top",
            "interaction-dependence-top-graph-" + self.name,
            self.title,
        )

        if self.description is None:
            self.description = """
        This plot shows the relation between feature values and shap interaction values.
        This allows you to investigate interactions between features in determining
        the prediction of the model.
        """
        self.popout_bottom = GraphPopout(
            self.name + "popout-bottom",
            "interaction-dependence-bottom-graph-" + self.name,
            self.title,
            self.description,
        )
        self.register_dependencies("shap_interaction_values")

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
                                        id="interaction-dependence-title-" + self.name,
                                    ),
                                    make_hideable(
                                        html.H6(
                                            self.subtitle, className="card-subtitle"
                                        ),
                                        hide=self.hide_subtitle,
                                    ),
                                    dbc.Tooltip(
                                        self.description,
                                        target="interaction-dependence-title-"
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
                                    dbc.Col([self.selector.layout()], width=2),
                                    hide=self.hide_selector,
                                ),
                            ]
                        ),
                        dbc.Row(
                            [
                                make_hideable(
                                    dbc.Col(
                                        [
                                            dbc.Label(
                                                "Feature:",
                                                id="interaction-dependence-col-label-"
                                                + self.name,
                                            ),
                                            dbc.Tooltip(
                                                "Select feature to display shap interactions for",
                                                target="interaction-dependence-col-label-"
                                                + self.name,
                                            ),
                                            dbc.Select(
                                                id="interaction-dependence-col-"
                                                + self.name,
                                                options=[
                                                    {"label": col, "value": col}
                                                    for col in self.explainer.columns_ranked_by_shap()
                                                ],
                                                value=self.col,
                                            ),
                                        ],
                                        md=3,
                                    ),
                                    hide=self.hide_col,
                                ),
                                make_hideable(
                                    dbc.Col(
                                        [
                                            html.Label(
                                                "Interaction:",
                                                id="interaction-dependence-interact-col-label-"
                                                + self.name,
                                            ),
                                            dbc.Tooltip(
                                                "Select feature to show interaction values for.  Two plots will be shown: "
                                                "both Feature vs Interaction Feature and Interaction Feature vs Feature.",
                                                target="interaction-dependence-interact-col-label-"
                                                + self.name,
                                            ),
                                            dbc.Select(
                                                id="interaction-dependence-interact-col-"
                                                + self.name,
                                                options=[
                                                    {"label": col, "value": col}
                                                    for col in self.explainer.top_shap_interactions(
                                                        col=self.col
                                                    )
                                                ],
                                                value=self.interact_col,
                                            ),
                                        ],
                                        md=3,
                                    ),
                                    hide=self.hide_interact_col,
                                ),
                                make_hideable(
                                    dbc.Col(
                                        [
                                            dbc.Label(
                                                f"{self.explainer.index_name}:",
                                                id="interaction-dependence-index-label-"
                                                + self.name,
                                            ),
                                            dbc.Tooltip(
                                                f"Select {self.explainer.index_name} to highlight in the plot."
                                                "You can also select by clicking on a scatter marker in the accompanying"
                                                " shap interaction summary plot (detailed).",
                                                target="interaction-dependence-index-label-"
                                                + self.name,
                                            ),
                                            self.index_selector.layout(),
                                        ],
                                        md=4,
                                    ),
                                    hide=self.hide_index,
                                ),
                            ]
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        make_hideable(
                                            dcc.Loading(
                                                id="loading-interaction-dependence-top-graph-"
                                                + self.name,
                                                children=[
                                                    dcc.Graph(
                                                        id="interaction-dependence-top-graph-"
                                                        + self.name,
                                                        config=dict(
                                                            modeBarButtons=[
                                                                ["toImage"]
                                                            ],
                                                            displaylogo=False,
                                                        ),
                                                    )
                                                ],
                                            ),
                                            hide=self.hide_top,
                                        ),
                                    ]
                                ),
                            ]
                        ),
                        dbc.Row(
                            [
                                make_hideable(
                                    dbc.Col(
                                        [self.popout_top.layout()], md=2, align="start"
                                    ),
                                    hide=self.hide_popout,
                                ),
                            ],
                            justify="end",
                        ),
                        dbc.Row(
                            [
                                make_hideable(
                                    dbc.Col(
                                        [
                                            dbc.Row(
                                                [
                                                    dbc.Tooltip(
                                                        "Remove outliers (> 1.5*IQR) in feature and interaction feature from plot.",
                                                        target="interaction-dependence-top-outliers-"
                                                        + self.name,
                                                    ),
                                                    dbc.Checklist(
                                                        options=[
                                                            {
                                                                "label": "Remove outliers",
                                                                "value": True,
                                                            }
                                                        ],
                                                        value=[True]
                                                        if self.remove_outliers
                                                        else [],
                                                        id="interaction-dependence-top-outliers-"
                                                        + self.name,
                                                        inline=True,
                                                        switch=True,
                                                    ),
                                                ]
                                            ),
                                        ],
                                        md=2,
                                    ),
                                    hide=self.hide_outliers,
                                ),
                                make_hideable(
                                    dbc.Col(
                                        [
                                            html.Div(
                                                [
                                                    dbc.Label(
                                                        "Categories:",
                                                        id="interaction-dependence-top-n-categories-label-"
                                                        + self.name,
                                                    ),
                                                    dbc.Tooltip(
                                                        "Maximum number of categories to display",
                                                        target="interaction-dependence-top-n-categories-label-"
                                                        + self.name,
                                                    ),
                                                    dbc.Input(
                                                        id="interaction-dependence-top-n-categories-"
                                                        + self.name,
                                                        value=self.cats_topx,
                                                        type="number",
                                                        min=1,
                                                        max=50,
                                                        step=1,
                                                    ),
                                                ],
                                                id="interaction-dependence-top-categories-div1-"
                                                + self.name,
                                                style={}
                                                if self.interact_col
                                                in self.explainer.cat_cols
                                                else dict(display="none"),
                                            ),
                                        ],
                                        md=2,
                                    ),
                                    self.hide_cats_topx,
                                ),
                                make_hideable(
                                    dbc.Col(
                                        [
                                            html.Div(
                                                [
                                                    html.Label(
                                                        "Sort categories:",
                                                        id="interaction-dependence-top-categories-sort-label-"
                                                        + self.name,
                                                    ),
                                                    dbc.Tooltip(
                                                        "How to sort the categories: Alphabetically, most common "
                                                        "first (Frequency), or highest mean absolute SHAP value first (Shap impact)",
                                                        target="interaction-dependence-top-categories-sort-label-"
                                                        + self.name,
                                                    ),
                                                    dbc.Select(
                                                        id="interaction-dependence-top-categories-sort-"
                                                        + self.name,
                                                        options=[
                                                            {
                                                                "label": "Alphabetically",
                                                                "value": "alphabet",
                                                            },
                                                            {
                                                                "label": "Frequency",
                                                                "value": "freq",
                                                            },
                                                            {
                                                                "label": "Shap impact",
                                                                "value": "shap",
                                                            },
                                                        ],
                                                        value=self.cats_sort,
                                                    ),
                                                ],
                                                id="interaction-dependence-top-categories-div2-"
                                                + self.name,
                                                style={}
                                                if self.interact_col
                                                in self.explainer.cat_cols
                                                else dict(display="none"),
                                            ),
                                        ],
                                        md=4,
                                    ),
                                    hide=self.hide_cats_sort,
                                ),
                            ]
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        make_hideable(
                                            dcc.Loading(
                                                id="loading-reverse-interaction-bottom-graph-"
                                                + self.name,
                                                children=[
                                                    dcc.Graph(
                                                        id="interaction-dependence-bottom-graph-"
                                                        + self.name,
                                                        config=dict(
                                                            modeBarButtons=[
                                                                ["toImage"]
                                                            ],
                                                            displaylogo=False,
                                                        ),
                                                    )
                                                ],
                                            ),
                                            hide=self.hide_bottom,
                                        ),
                                    ]
                                ),
                            ]
                        ),
                        dbc.Row(
                            [
                                make_hideable(
                                    dbc.Col(
                                        [self.popout_bottom.layout()],
                                        md=2,
                                        align="start",
                                    ),
                                    hide=self.hide_popout,
                                ),
                            ],
                            justify="end",
                        ),
                        dbc.Row(
                            [
                                make_hideable(
                                    dbc.Col(
                                        [
                                            dbc.Row(
                                                [
                                                    dbc.Tooltip(
                                                        "Remove outliers (> 1.5*IQR) in feature and interaction feature from plot.",
                                                        target="interaction-dependence-bottom-outliers-"
                                                        + self.name,
                                                    ),
                                                    dbc.Checklist(
                                                        options=[
                                                            {
                                                                "label": "Remove outliers",
                                                                "value": True,
                                                            }
                                                        ],
                                                        value=[True]
                                                        if self.remove_outliers
                                                        else [],
                                                        id="interaction-dependence-bottom-outliers-"
                                                        + self.name,
                                                        inline=True,
                                                        switch=True,
                                                    ),
                                                ]
                                            ),
                                        ],
                                        md=2,
                                    ),
                                    hide=self.hide_outliers,
                                ),
                                make_hideable(
                                    dbc.Col(
                                        [
                                            html.Div(
                                                [
                                                    dbc.Label(
                                                        "Categories:",
                                                        id="interaction-dependence-bottom-n-categories-label-"
                                                        + self.name,
                                                    ),
                                                    dbc.Tooltip(
                                                        "Maximum number of categories to display",
                                                        target="interaction-dependence-bottom-n-categories-label-"
                                                        + self.name,
                                                    ),
                                                    dbc.Input(
                                                        id="interaction-dependence-bottom-n-categories-"
                                                        + self.name,
                                                        value=self.cats_topx,
                                                        type="number",
                                                        min=1,
                                                        max=50,
                                                        step=1,
                                                    ),
                                                ],
                                                id="interaction-dependence-bottom-categories-div1-"
                                                + self.name,
                                                style={}
                                                if self.col in self.explainer.cat_cols
                                                else dict(display="none"),
                                            ),
                                        ],
                                        md=2,
                                    ),
                                    self.hide_cats_topx,
                                ),
                                make_hideable(
                                    dbc.Col(
                                        [
                                            html.Div(
                                                [
                                                    html.Label(
                                                        "Sort categories:",
                                                        id="interaction-dependence-bottom-categories-sort-label-"
                                                        + self.name,
                                                    ),
                                                    dbc.Tooltip(
                                                        "How to sort the categories: Alphabetically, most common "
                                                        "first (Frequency), or highest mean absolute SHAP value first (Shap impact)",
                                                        target="interaction-dependence-bottom-categories-sort-label-"
                                                        + self.name,
                                                    ),
                                                    dbc.Select(
                                                        id="interaction-dependence-bottom-categories-sort-"
                                                        + self.name,
                                                        options=[
                                                            {
                                                                "label": "Alphabetically",
                                                                "value": "alphabet",
                                                            },
                                                            {
                                                                "label": "Frequency",
                                                                "value": "freq",
                                                            },
                                                            {
                                                                "label": "Shap impact",
                                                                "value": "shap",
                                                            },
                                                        ],
                                                        value=self.cats_sort,
                                                    ),
                                                ],
                                                id="interaction-dependence-bottom-categories-div2-"
                                                + self.name,
                                                style={}
                                                if self.col in self.explainer.cat_cols
                                                else dict(display="none"),
                                            ),
                                        ],
                                        md=4,
                                    ),
                                    hide=self.hide_cats_sort,
                                ),
                            ]
                        ),
                    ]
                ),
            ],
            class_name="h-100",
        )

    def to_html(self, state_dict=None, add_header=True):
        args = self.get_state_args(state_dict)
        fig_top = self.explainer.plot_interaction(
            args["interact_col"],
            args["col"],
            highlight_index=args["index"],
            pos_label=args["pos_label"],
            topx=args["cats_topx"],
            sort=args["cats_sort"],
            max_cat_colors=self.max_cat_colors,
            plot_sample=self.plot_sample,
            remove_outliers=bool(args["remove_outliers"]),
        )
        fig_bottom = self.explainer.plot_interaction(
            args["col"],
            args["interact_col"],
            highlight_index=args["index"],
            pos_label=args["pos_label"],
            topx=args["cats_topx"],
            sort=args["cats_sort"],
            max_cat_colors=self.max_cat_colors,
            plot_sample=self.plot_sample,
            remove_outliers=bool(args["remove_outliers"]),
        )

        html = to_html.card(
            to_html.fig(fig_top) + to_html.fig(fig_bottom),
            title=self.title,
            subtitle=self.subtitle,
        )
        if add_header:
            return to_html.add_header(html)
        return html

    def component_callbacks(self, app):
        @app.callback(
            Output("interaction-dependence-interact-col-" + self.name, "options"),
            [
                Input("interaction-dependence-col-" + self.name, "value"),
                Input("pos-label-" + self.name, "value"),
            ],
            [State("interaction-dependence-interact-col-" + self.name, "value")],
        )
        def update_interaction_dependence_interact_col(
            col, pos_label, old_interact_col
        ):
            if col is not None:
                new_interact_cols = self.explainer.top_shap_interactions(
                    col, pos_label=pos_label
                )
                new_interact_options = [
                    {"label": col, "value": col} for col in new_interact_cols
                ]
                return new_interact_options
            raise PreventUpdate

        @app.callback(
            [
                Output("interaction-dependence-top-graph-" + self.name, "figure"),
                Output(
                    "interaction-dependence-top-categories-div1-" + self.name, "style"
                ),
                Output(
                    "interaction-dependence-top-categories-div2-" + self.name, "style"
                ),
            ],
            [
                Input("interaction-dependence-interact-col-" + self.name, "value"),
                Input("interaction-dependence-index-" + self.name, "value"),
                Input("interaction-dependence-top-n-categories-" + self.name, "value"),
                Input(
                    "interaction-dependence-top-categories-sort-" + self.name, "value"
                ),
                Input("interaction-dependence-top-outliers-" + self.name, "value"),
                Input("pos-label-" + self.name, "value"),
                Input("interaction-dependence-col-" + self.name, "value"),
            ],
        )
        def update_dependence_graph(
            interact_col, index, topx, sort, remove_outliers, pos_label, col
        ):
            if col is not None and interact_col is not None:
                style = (
                    {}
                    if interact_col in self.explainer.cat_cols
                    else dict(display="none")
                )
                return (
                    self.explainer.plot_interaction(
                        interact_col,
                        col,
                        highlight_index=index,
                        pos_label=pos_label,
                        topx=topx,
                        sort=sort,
                        max_cat_colors=self.max_cat_colors,
                        plot_sample=self.plot_sample,
                        remove_outliers=bool(remove_outliers),
                    ),
                    style,
                    style,
                )
            raise PreventUpdate

        @app.callback(
            [
                Output("interaction-dependence-bottom-graph-" + self.name, "figure"),
                Output(
                    "interaction-dependence-bottom-categories-div1-" + self.name,
                    "style",
                ),
                Output(
                    "interaction-dependence-bottom-categories-div2-" + self.name,
                    "style",
                ),
            ],
            [
                Input("interaction-dependence-interact-col-" + self.name, "value"),
                Input("interaction-dependence-index-" + self.name, "value"),
                Input(
                    "interaction-dependence-bottom-n-categories-" + self.name, "value"
                ),
                Input(
                    "interaction-dependence-bottom-categories-sort-" + self.name,
                    "value",
                ),
                Input("interaction-dependence-bottom-outliers-" + self.name, "value"),
                Input("pos-label-" + self.name, "value"),
                Input("interaction-dependence-col-" + self.name, "value"),
            ],
        )
        def update_dependence_graph(
            interact_col, index, topx, sort, remove_outliers, pos_label, col
        ):
            if col is not None and interact_col is not None:
                style = {} if col in self.explainer.cat_cols else dict(display="none")
                return (
                    self.explainer.plot_interaction(
                        col,
                        interact_col,
                        highlight_index=index,
                        pos_label=pos_label,
                        topx=topx,
                        sort=sort,
                        max_cat_colors=self.max_cat_colors,
                        plot_sample=self.plot_sample,
                        remove_outliers=bool(remove_outliers),
                    ),
                    style,
                    style,
                )
            raise PreventUpdate


class InteractionSummaryDependenceConnector(ExplainerComponent):
    def __init__(self, interaction_summary_component, interaction_dependence_component):
        """Connects a InteractionSummaryComponent with an InteractionDependenceComponent:

        - When select feature in summary, then select col in Dependence
        - When clicking on interaction feature in Summary, then select that interaction
            feature in Dependence.

        Args:
            shap_summary_component (ShapSummaryComponent): ShapSummaryComponent
            shap_dependence_component (ShapDependenceComponent): ShapDependenceComponent
        """
        self.sum_name = interaction_summary_component.name
        self.dep_name = interaction_dependence_component.name

    def component_callbacks(self, app):
        @app.callback(
            [
                Output("interaction-dependence-col-" + self.dep_name, "value"),
                Output("interaction-dependence-index-" + self.dep_name, "value"),
                Output("interaction-dependence-interact-col-" + self.dep_name, "value"),
            ],
            [
                Input("interaction-summary-col-" + self.sum_name, "value"),
                Input("interaction-summary-graph-" + self.sum_name, "clickData"),
            ],
        )
        def update_interact_col_highlight(col, clickdata):
            if clickdata is not None and clickdata["points"][0] is not None:
                if isinstance(clickdata["points"][0]["y"], float):  # detailed
                    index = (
                        clickdata["points"][0]["text"].split("=")[1].split("<br>")[0]
                    )
                    interact_col = (
                        clickdata["points"][0]["text"].split("=")[1].split("<br>")[1]
                    )
                    return (col, index, interact_col)
                elif isinstance(clickdata["points"][0]["y"], str):  # aggregate
                    # in aggregate clickdata returns col name -> type==str
                    interact_col = clickdata["points"][0]["y"].split(" ")[1]
                    return (col, dash.no_update, interact_col)
            else:
                return (col, dash.no_update, dash.no_update)
            raise PreventUpdate


class ShapContributionsGraphComponent(ExplainerComponent):
    _state_props = dict(
        index=("contributions-graph-index-", "value"),
        depth=("contributions-graph-depth-", "value"),
        sort=("contributions-graph-sorting-", "value"),
        orientation=("contributions-graph-orientation-", "value"),
        pos_label=("pos-label-", "value"),
    )

    def __init__(
        self,
        explainer,
        title="Contributions Plot",
        name=None,
        subtitle="How has each feature contributed to the prediction?",
        hide_title=False,
        hide_subtitle=False,
        hide_index=False,
        hide_depth=False,
        hide_sort=False,
        hide_orientation=True,
        hide_selector=False,
        hide_popout=False,
        feature_input_component=None,
        index_dropdown=True,
        pos_label=None,
        index=None,
        depth=None,
        sort="high-to-low",
        orientation="vertical",
        higher_is_better=True,
        description=None,
        **kwargs,
    ):
        """Display Shap contributions to prediction graph component

        Args:
            explainer (Explainer): explainer object constructed , with either
                        ClassifierExplainer() or RegressionExplainer()
            title (str, optional): Title of tab or page. Defaults to
                        "Contributions Plot".
            name (str, optional): unique name to add to Component elements.
                        If None then random uuid is generated to make sure
                        it's unique. Defaults to None.
            subtitle (str): subtitle
            hide_title (bool, optional): Hide component title. Defaults to False.
            hide_subtitle (bool, optional): Hide subtitle. Defaults to False.
            hide_index (bool, optional): Hide index selector. Defaults to False.
            hide_depth (bool, optional): Hide depth toggle. Defaults to False.
            hide_sort (bool, optional): Hide the sorting dropdown. Defaults to False.
            hide_orientation (bool, optional): Hide the orientation dropdown.
                    Defaults to True.
            hide_selector (bool, optional): hide pos label selector. Defaults to False.
            hide_popout (bool, optional): hide popout button
            feature_input_component (FeatureInputComponent): A FeatureInputComponent
                that will give the input to the graph instead of the index selector.
                If not None, hide_index=True. Defaults to None.
            index_dropdown (bool, optional): Use dropdown for index input instead
                of free text input. Defaults to True.
            pos_label ({int, str}, optional): initial pos label.
                        Defaults to explainer.pos_label
            index ({int, bool}, optional): Initial index to display. Defaults to None.
            depth (int, optional): Initial number of features to display. Defaults to None.
            sort ({'abs', 'high-to-low', 'low-to-high', 'importance'}, optional): sorting of shap values.
                        Defaults to 'high-to-low'.
            orientation ({'vertical', 'horizontal'}, optional): orientation of bar chart.
                        Defaults to 'vertical'.
            higher_is_better (bool, optional): Color positive shap values green and
                negative shap values red, or the reverse.
            description (str, optional): Tooltip to display when hover over
                component title. When None default text is shown.
        """
        super().__init__(explainer, title, name)

        self.index_name = "contributions-graph-index-" + self.name

        if self.depth is not None:
            self.depth = min(self.depth, self.explainer.n_features)
        else:
            self.depth = self.explainer.n_features

        if self.feature_input_component is not None:
            self.exclude_callbacks(self.feature_input_component)
            self.hide_index = True

        if self.description is None:
            self.description = """
        This plot shows the contribution that each individual feature has had
        on the prediction for a specific observation. The contributions (starting
        from the population average) add up to the final prediction. This allows you
        to explain exactly how each individual prediction has been built up
        from all the individual ingredients in the model.
        """

        self.selector = PosLabelSelector(explainer, name=self.name, pos_label=pos_label)
        self.index_selector = IndexSelector(
            explainer,
            "contributions-graph-index-" + self.name,
            index=index,
            index_dropdown=index_dropdown,
            **kwargs,
        )

        self.popout = GraphPopout(
            "contributions-graph-" + self.name + "popout",
            "contributions-graph-" + self.name,
            self.title,
            self.description,
        )
        self.register_dependencies("shap_values_df")

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
                                        id="contributions-graph-title-" + self.name,
                                    ),
                                    make_hideable(
                                        html.H6(
                                            self.subtitle, className="card-subtitle"
                                        ),
                                        hide=self.hide_subtitle,
                                    ),
                                    dbc.Tooltip(
                                        self.description,
                                        target="contributions-graph-title-" + self.name,
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
                            ],
                            justify="right",
                        ),
                        dbc.Row(
                            [
                                make_hideable(
                                    dbc.Col(
                                        [
                                            dbc.Label(
                                                f"{self.explainer.index_name}:",
                                                id="contributions-graph-index-label-"
                                                + self.name,
                                            ),
                                            dbc.Tooltip(
                                                f"Select the {self.explainer.index_name} to display the feature contributions for",
                                                target="contributions-graph-index-label-"
                                                + self.name,
                                            ),
                                            self.index_selector.layout(),
                                        ],
                                        md=4,
                                    ),
                                    hide=self.hide_index,
                                ),
                                make_hideable(
                                    dbc.Col(
                                        [
                                            dbc.Label(
                                                "Depth:",
                                                id="contributions-graph-depth-label-"
                                                + self.name,
                                            ),
                                            dbc.Tooltip(
                                                "Number of features to display",
                                                target="contributions-graph-depth-label-"
                                                + self.name,
                                            ),
                                            dbc.Select(
                                                id="contributions-graph-depth-"
                                                + self.name,
                                                options=[
                                                    {
                                                        "label": str(i + 1),
                                                        "value": i + 1,
                                                    }
                                                    for i in range(
                                                        self.explainer.n_features
                                                    )
                                                ],
                                                value=None
                                                if self.depth is None
                                                else str(self.depth),
                                                size="sm",
                                            ),
                                        ],
                                        md=2,
                                    ),
                                    hide=self.hide_depth,
                                ),
                                make_hideable(
                                    dbc.Col(
                                        [
                                            dbc.Label(
                                                "Sorting:",
                                                id="contributions-graph-sorting-label-"
                                                + self.name,
                                            ),
                                            dbc.Tooltip(
                                                "Sort the features either by highest absolute (positive or negative) impact (absolute), "
                                                "from most positive the most negative (high-to-low)"
                                                "from most negative to most positive (low-to-high or "
                                                "according the global feature importance ordering (importance).",
                                                target="contributions-graph-sorting-label-"
                                                + self.name,
                                            ),
                                            dbc.Select(
                                                id="contributions-graph-sorting-"
                                                + self.name,
                                                options=[
                                                    {
                                                        "label": "Absolute",
                                                        "value": "abs",
                                                    },
                                                    {
                                                        "label": "High to Low",
                                                        "value": "high-to-low",
                                                    },
                                                    {
                                                        "label": "Low to High",
                                                        "value": "low-to-high",
                                                    },
                                                    {
                                                        "label": "Importance",
                                                        "value": "importance",
                                                    },
                                                ],
                                                value=self.sort,
                                                size="sm",
                                            ),
                                        ],
                                        md=2,
                                    ),
                                    hide=self.hide_sort,
                                ),
                                make_hideable(
                                    dbc.Col(
                                        [
                                            dbc.Label(
                                                "Orientation:",
                                                id="contributions-graph-orientation-label-"
                                                + self.name,
                                            ),
                                            dbc.Tooltip(
                                                "Show vertical bars left to right or horizontal bars from top to bottom",
                                                target="contributions-graph-orientation-label-"
                                                + self.name,
                                            ),
                                            dbc.Select(
                                                id="contributions-graph-orientation-"
                                                + self.name,
                                                options=[
                                                    {
                                                        "label": "Vertical",
                                                        "value": "vertical",
                                                    },
                                                    {
                                                        "label": "Horizontal",
                                                        "value": "horizontal",
                                                    },
                                                ],
                                                value=self.orientation,
                                                size="sm",
                                            ),
                                        ],
                                        md=2,
                                    ),
                                    hide=self.hide_orientation,
                                ),
                            ]
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        dcc.Loading(
                                            id="loading-contributions-graph-"
                                            + self.name,
                                            children=[
                                                dcc.Graph(
                                                    id="contributions-graph-"
                                                    + self.name,
                                                    config=dict(
                                                        modeBarButtons=[["toImage"]],
                                                        displaylogo=False,
                                                    ),
                                                )
                                            ],
                                        ),
                                    ]
                                ),
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
            ],
            class_name="h-100",
        )

    def get_state_tuples(self):
        _state_tuples = super().get_state_tuples()
        if self.feature_input_component is not None:
            _state_tuples.extend(self.feature_input_component.get_state_tuples())
        return sorted(list(set(_state_tuples)))

    def to_html(self, state_dict=None, add_header=True):
        args = self.get_state_args(state_dict)
        args["depth"] = None if args["depth"] is None else int(args["depth"])
        if self.feature_input_component is None:
            if args["index"] is not None:
                fig = self.explainer.plot_contributions(
                    args["index"],
                    topx=args["depth"],
                    sort=args["sort"],
                    orientation=args["orientation"],
                    pos_label=args["pos_label"],
                    higher_is_better=self.higher_is_better,
                )
                html = to_html.fig(fig)
            else:
                html = "<div>no index selected</div>"
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
                shap_values = self.explainer.get_shap_row(
                    X_row=X_row, pos_label=args["pos_label"]
                )

                fig = self.explainer.plot_contributions(
                    X_row=X_row,
                    topx=args["depth"],
                    sort=args["sort"],
                    orientation=args["orientation"],
                    pos_label=args["pos_label"],
                    higher_is_better=self.higher_is_better,
                )
                html = to_html.fig(fig)
            else:
                html = f"<div>input data incorrect</div>"

        html = to_html.card(html, title=self.title, subtitle=self.subtitle)
        if add_header:
            return to_html.add_header(html)
        return html

    def component_callbacks(self, app):
        if self.feature_input_component is None:

            @app.callback(
                Output("contributions-graph-" + self.name, "figure"),
                [
                    Input("contributions-graph-index-" + self.name, "value"),
                    Input("contributions-graph-depth-" + self.name, "value"),
                    Input("contributions-graph-sorting-" + self.name, "value"),
                    Input("contributions-graph-orientation-" + self.name, "value"),
                    Input("pos-label-" + self.name, "value"),
                ],
            )
            def update_output_div(index, depth, sort, orientation, pos_label):
                if index is None or not self.explainer.index_exists(index):
                    raise PreventUpdate
                depth = None if depth is None else int(depth)
                plot = self.explainer.plot_contributions(
                    str(index),
                    topx=depth,
                    sort=sort,
                    orientation=orientation,
                    pos_label=pos_label,
                    higher_is_better=self.higher_is_better,
                )
                return plot

        else:

            @app.callback(
                Output("contributions-graph-" + self.name, "figure"),
                [
                    Input("contributions-graph-depth-" + self.name, "value"),
                    Input("contributions-graph-sorting-" + self.name, "value"),
                    Input("contributions-graph-orientation-" + self.name, "value"),
                    Input("pos-label-" + self.name, "value"),
                    *self.feature_input_component._feature_callback_inputs,
                ],
            )
            def update_output_div(depth, sort, orientation, pos_label, *inputs):
                depth = None if depth is None else int(depth)
                if not any([i is None for i in inputs]):
                    X_row = self.explainer.get_row_from_input(
                        inputs, ranked_by_shap=True
                    )
                    plot = self.explainer.plot_contributions(
                        X_row=X_row,
                        topx=depth,
                        sort=sort,
                        orientation=orientation,
                        pos_label=pos_label,
                        higher_is_better=self.higher_is_better,
                    )
                    return plot
                raise PreventUpdate


class ShapContributionsTableComponent(ExplainerComponent):
    _state_props = dict(
        index=("contributions-table-index-", "value"),
        depth=("contributions-table-depth-", "value"),
        sort=("contributions-table-sorting-", "value"),
        pos_label=("pos-label-", "value"),
    )

    def __init__(
        self,
        explainer,
        title="Contributions Table",
        name=None,
        subtitle="How has each feature contributed to the prediction?",
        hide_title=False,
        hide_subtitle=False,
        hide_index=False,
        hide_depth=False,
        hide_sort=False,
        hide_selector=False,
        feature_input_component=None,
        index_dropdown=True,
        pos_label=None,
        index=None,
        depth=None,
        sort="abs",
        description=None,
        **kwargs,
    ):
        """Show SHAP values contributions to prediction in a table component

        Args:
            explainer (Explainer): explainer object constructed with either
                        ClassifierExplainer() or RegressionExplainer()
            title (str, optional): Title of tab or page. Defaults to
                        "Contributions Table".
            name (str, optional): unique name to add to Component elements.
                        If None then random uuid is generated to make sure
                        it's unique. Defaults to None.
            subtitle (str): subtitle
            hide_title (bool, optional): Hide component title. Defaults to False.
            hide_subtitle (bool, optional): Hide subtitle. Defaults to False.
            hide_index (bool, optional): Hide index selector. Defaults to False.
            hide_depth (bool, optional): Hide depth selector. Defaults to False.
            hide_sort (bool, optional): Hide sorting dropdown. Default to False.
            hide_selector (bool, optional): hide pos label selector. Defaults to False.
            feature_input_component (FeatureInputComponent): A FeatureInputComponent
                that will give the input to the graph instead of the index selector.
                If not None, hide_index=True. Defaults to None.
            index_dropdown (bool, optional): Use dropdown for index input instead
                of free text input. Defaults to True.
            pos_label ({int, str}, optional): initial pos label.
                        Defaults to explainer.pos_label
            index ([type], optional): Initial index to display. Defaults to None.
            depth ([type], optional): Initial number of features to display. Defaults to None.
            sort ({'abs', 'high-to-low', 'low-to-high', 'importance'}, optional): sorting of shap values.
                        Defaults to 'high-to-low'.
            description (str, optional): Tooltip to display when hover over
                component title. When None default text is shown.
        """
        super().__init__(explainer, title, name)

        self.index_name = "contributions-table-index-" + self.name

        if self.depth is not None:
            self.depth = min(self.depth, self.explainer.n_features)
        else:
            self.depth = self.explainer.n_features

        if self.feature_input_component is not None:
            self.exclude_callbacks(self.feature_input_component)
            self.hide_index = True

        if self.description is None:
            self.description = """
        This tables shows the contribution that each individual feature has had
        on the prediction for a specific observation. The contributions (starting
        from the population average) add up to the final prediction. This allows you
        to explain exactly how each individual prediction has been built up
        from all the individual ingredients in the model.
        """
        self.selector = PosLabelSelector(explainer, name=self.name, pos_label=pos_label)
        self.index_selector = IndexSelector(
            explainer,
            "contributions-table-index-" + self.name,
            index=index,
            index_dropdown=index_dropdown,
            **kwargs,
        )

        self.register_dependencies("shap_values_df")

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
                                        id="contributions-table-title-" + self.name,
                                    ),
                                    make_hideable(
                                        html.H6(
                                            self.subtitle, className="card-subtitle"
                                        ),
                                        hide=self.hide_subtitle,
                                    ),
                                    dbc.Tooltip(
                                        self.description,
                                        target="contributions-table-title-" + self.name,
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
                                            dbc.Label(
                                                f"{self.explainer.index_name}:",
                                                id="contributions-table-index-label-"
                                                + self.name,
                                            ),
                                            dbc.Tooltip(
                                                f"Select the {self.explainer.index_name} to display the feature contributions for",
                                                target="contributions-table-index-label-"
                                                + self.name,
                                            ),
                                            self.index_selector.layout(),
                                        ],
                                        md=4,
                                    ),
                                    hide=self.hide_index,
                                ),
                                make_hideable(
                                    dbc.Col(
                                        [
                                            dbc.Label(
                                                "Depth:",
                                                id="contributions-table-depth-label-"
                                                + self.name,
                                            ),
                                            dbc.Tooltip(
                                                "Number of features to display",
                                                target="contributions-table-depth-label-"
                                                + self.name,
                                            ),
                                            dbc.Select(
                                                id="contributions-table-depth-"
                                                + self.name,
                                                options=[
                                                    {
                                                        "label": str(i + 1),
                                                        "value": i + 1,
                                                    }
                                                    for i in range(
                                                        self.explainer.n_features
                                                    )
                                                ],
                                                value=self.depth,
                                                size="sm",
                                            ),
                                        ],
                                        md=2,
                                    ),
                                    hide=self.hide_depth,
                                ),
                                make_hideable(
                                    dbc.Col(
                                        [
                                            dbc.Label(
                                                "Sorting:",
                                                id="contributions-table-sorting-label-"
                                                + self.name,
                                            ),
                                            dbc.Tooltip(
                                                "Sort the features either by highest absolute (positive or negative) impact (absolute), "
                                                "from most positive the most negative (high-to-low)"
                                                "from most negative to most positive (low-to-high or "
                                                "according the global feature importance ordering (importance).",
                                                target="contributions-table-sorting-label-"
                                                + self.name,
                                            ),
                                            dbc.Select(
                                                id="contributions-table-sorting-"
                                                + self.name,
                                                options=[
                                                    {
                                                        "label": "Absolute",
                                                        "value": "abs",
                                                    },
                                                    {
                                                        "label": "High to Low",
                                                        "value": "high-to-low",
                                                    },
                                                    {
                                                        "label": "Low to High",
                                                        "value": "low-to-high",
                                                    },
                                                    {
                                                        "label": "Importance",
                                                        "value": "importance",
                                                    },
                                                ],
                                                value=self.sort,
                                                size="sm",
                                            ),
                                        ],
                                        md=2,
                                    ),
                                    hide=self.hide_sort,
                                ),
                                make_hideable(
                                    dbc.Col([self.selector.layout()], width=2),
                                    hide=self.hide_selector,
                                ),
                            ]
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        dcc.Loading(
                                            id="loading-contributions-table-"
                                            + self.name,
                                            children=[
                                                html.Div(
                                                    id="contributions-table-"
                                                    + self.name
                                                )
                                            ],
                                        ),
                                    ]
                                ),
                            ]
                        ),
                    ]
                ),
            ],
            class_name="h-100",
        )

    def get_state_tuples(self):
        _state_tuples = super().get_state_tuples()
        # also get state_tuples from feature_input_component as they are needed
        # to generate html from state:
        if self.feature_input_component is not None:
            _state_tuples.extend(self.feature_input_component.get_state_tuples())
        return sorted(list(set(_state_tuples)))

    def to_html(self, state_dict=None, add_header=True):
        args = self.get_state_args(state_dict)
        args["depth"] = None if args["depth"] is None else int(args["depth"])

        if self.feature_input_component is None:
            if args["index"] is not None:
                contrib_df = self.explainer.get_contrib_summary_df(
                    args["index"],
                    topx=args["depth"],
                    sort=args["sort"],
                    pos_label=args["pos_label"],
                )
                html = to_html.table_from_df(contrib_df)
            else:
                html = "<div>no index selected</div>"
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
                contrib_df = self.explainer.get_contrib_summary_df(
                    X_row=X_row,
                    topx=args["depth"],
                    sort=args["sort"],
                    pos_label=args["pos_label"],
                )
                html = to_html.table_from_df(contrib_df)
            else:
                html = f"<div>input data incorrect</div>"

        html = to_html.card(html, title=self.title, subtitle=self.subtitle)
        if add_header:
            return to_html.add_header(html)
        return html

    def component_callbacks(self, app):
        if self.feature_input_component is None:

            @app.callback(
                Output("contributions-table-" + self.name, "children"),
                [
                    Input("contributions-table-index-" + self.name, "value"),
                    Input("contributions-table-depth-" + self.name, "value"),
                    Input("contributions-table-sorting-" + self.name, "value"),
                    Input("pos-label-" + self.name, "value"),
                ],
            )
            def update_output_div(index, depth, sort, pos_label):
                if index is None or not self.explainer.index_exists(index):
                    raise PreventUpdate
                depth = None if depth is None else int(depth)
                contributions_table = dbc.Table.from_dataframe(
                    self.explainer.get_contrib_summary_df(
                        str(index), topx=depth, sort=sort, pos_label=pos_label
                    )
                )

                tooltip_cols = {}
                for tr in contributions_table.children[1].children:
                    # insert tooltip target id's into the table html.Tr() elements:
                    tds = tr.children
                    col = tds[0].children.split(" = ")[0]
                    if self.explainer.description(col) != "":
                        tr.id = f"contributions-table-hover-{col}-" + self.name
                        tooltip_cols[col] = self.explainer.description(col)

                tooltips = [
                    dbc.Tooltip(
                        desc,
                        target=f"contributions-table-hover-{col}-" + self.name,
                        placement="top",
                    )
                    for col, desc in tooltip_cols.items()
                ]

                output_div = html.Div([contributions_table, *tooltips])
                return output_div

        else:

            @app.callback(
                Output("contributions-table-" + self.name, "children"),
                [
                    Input("contributions-table-depth-" + self.name, "value"),
                    Input("contributions-table-sorting-" + self.name, "value"),
                    Input("pos-label-" + self.name, "value"),
                    *self.feature_input_component._feature_callback_inputs,
                ],
            )
            def update_output_div(depth, sort, pos_label, *inputs):
                if not any([i is None for i in inputs]):
                    X_row = self.explainer.get_row_from_input(
                        inputs, ranked_by_shap=True
                    )
                    depth = None if depth is None else int(depth)
                    contributions_table = dbc.Table.from_dataframe(
                        self.explainer.get_contrib_summary_df(
                            X_row=X_row, topx=depth, sort=sort, pos_label=pos_label
                        )
                    )

                    tooltip_cols = {}
                    for tr in contributions_table.children[1].children:
                        # insert tooltip target id's into the table html.Tr() elements:
                        tds = tr.children
                        col = tds[0].children.split(" = ")[0]
                        if self.explainer.description(col) != "":
                            tr.id = f"contributions-table-hover-{col}-" + self.name
                            tooltip_cols[col] = self.explainer.description(col)

                    tooltips = [
                        dbc.Tooltip(
                            desc,
                            target=f"contributions-table-hover-{col}-" + self.name,
                            placement="top",
                        )
                        for col, desc in tooltip_cols.items()
                    ]

                    output_div = html.Div([contributions_table, *tooltips])
                    return output_div
                raise PreventUpdate
