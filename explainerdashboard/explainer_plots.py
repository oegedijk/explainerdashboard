__all__ = [
    "plotly_prediction_piechart",
    "plotly_contribution_plot",
    "plotly_precision_plot",
    "plotly_classification_plot",
    "plotly_lift_curve",
    "plotly_cumulative_precision_plot",
    "plotly_dependence_plot",
    "plotly_shap_violin_plot",
    "plotly_pdp",
    "plotly_importances_plot",
    "plotly_confusion_matrix",
    "plotly_roc_auc_curve",
    "plotly_pr_auc_curve",
    "plotly_shap_scatter_plot",
    "plotly_predicted_vs_actual",
    "plotly_plot_residuals",
    "plotly_residuals_vs_col",
    "plotly_actual_vs_col",
    "plotly_preds_vs_col",
    "plotly_rf_trees",
    "plotly_xgboost_trees",
]

import warnings

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

import plotly.graph_objs as go
from plotly.subplots import make_subplots

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
    roc_auc_score,
    average_precision_score,
)

from .explainer_methods import matching_cols, safe_isinstance


def plotly_prediction_piechart(predictions_df, showlegend=True, size=250):
    """Return piechart with predict_proba distributions for ClassifierExplainer

    Args:
        predictions_df (pd.DataFrame): generated with
            ClassifierExplainer.prediction_summary_df(index)
        showlegend (bool, optional): Show the legend. Defaults to True.
        size (int): width and height of the plot

    Returns:
        plotly.Fig
    """

    data = [
        go.Pie(
            labels=predictions_df.label.values,
            values=predictions_df.probability.values,
            hole=0.3,
            sort=False,
        )
    ]
    layout = dict(
        autosize=False,
        width=size,
        height=size,
        margin=dict(l=20, r=20, b=20, t=30, pad=4),
        showlegend=showlegend,
    )
    fig = go.Figure(data, layout)
    return fig


def plotly_contribution_plot(
    contrib_df,
    target="",
    model_output="raw",
    higher_is_better=True,
    include_base_value=True,
    include_prediction=True,
    orientation="vertical",
    round=2,
    units="",
):
    """Generate a shap contributions waterfall plot from a contrib_df dataframe

    Args:
        contrib_df (pd.DataFrame): contrib_df generated with get_contrib_df(...)
        target (str, optional): Target variable to be displayed. Defaults to "".
        model_output ({"raw", "logodds", "probability"}, optional): Kind of
            output of the model. Defaults to "raw".
        higher_is_better (bool, optional): Display increases in shap as green,
            decreases as red. Defaults to False.
        include_base_value (bool, optional): Include shap base value in the plot.
            Defaults to True.
        include_prediction (bool, optional): Include the final prediction in
            the plot. Defaults to True.
        orientation ({'vertical', 'horizontal'}, optional): Display the plot
            vertically or horizontally. Defaults to 'vertical'.
        round (int, optional): Round of floats. Defaults to 2.
        units (str, optional): Units of outcome variable.  Defaults to "".

    Returns:
        plotly fig:
    """

    if orientation not in ["vertical", "horizontal"]:
        raise ValueError(
            f"orientation should be in ['vertical', 'horizontal'], but you passed orientation={orientation}"
        )
    if model_output not in ["raw", "probability", "logodds"]:
        raise ValueError(
            f"model_output should be in ['raw', 'probability', 'logodds'], but you passed orientation={model_output}"
        )

    contrib_df = contrib_df.copy()
    try:
        base_value = contrib_df.query("col=='_BASE'")["contribution"].item()
    except:
        base_value = None

    if not include_base_value:
        contrib_df = contrib_df[contrib_df.col != "_BASE"]
    if not include_prediction:
        contrib_df = contrib_df[contrib_df.col != "_PREDICTION"]
    contrib_df = contrib_df.replace(
        {
            "_BASE": "Population<br>average",
            "_REST": "Other features combined",
            "_PREDICTION": "Final Prediction",
        }
    )

    multiplier = 100 if model_output == "probability" else 1
    contrib_df["base"] = np.round(multiplier * contrib_df["base"].astype(float), round)
    contrib_df["cumulative"] = np.round(
        multiplier * contrib_df["cumulative"].astype(float), round
    )
    contrib_df["contribution"] = np.round(
        multiplier * contrib_df["contribution"].astype(float), round
    )

    if not include_base_value:
        contrib_df = contrib_df[contrib_df.col != "_BASE"]

    longest_feature_name = contrib_df["col"].str.len().max()

    # prediction is the sum of all contributions:
    prediction = contrib_df["cumulative"].values[-1]
    cols = contrib_df["col"].values.tolist()
    values = contrib_df.value.tolist()
    bases = contrib_df.base.tolist()
    contribs = contrib_df.contribution.tolist()

    if "value" in contrib_df.columns:
        hover_text = [
            f"{col}={value}<BR>{'+' if contrib>0 else ''}{contrib} {units}"
            for col, value, contrib in zip(cols, values, contribs)
        ]
    else:
        hover_text = [
            f"{col}=?<BR>{'+' if contrib>0 else ''}{contrib} {units}"
            for col, contrib in zip(cols, contribs)
        ]

    green_fill, green_line = "rgba(50, 200, 50, 1.0)", "rgba(40, 160, 50, 1.0)"
    yellow_fill, yellow_line = "rgba(230, 230, 30, 1.0)", "rgba(190, 190, 30, 1.0)"
    blue_fill, blue_line = "rgba(55, 128, 191, 0.7)", "rgba(55, 128, 191, 1.0)"
    red_fill, red_line = "rgba(219, 64, 82, 0.7)", "rgba(219, 64, 82, 1.0)"

    fill_color_up = green_fill if higher_is_better else red_fill
    fill_color_down = red_fill if higher_is_better else green_fill
    line_color_up = green_line if higher_is_better else red_line
    line_color_down = red_line if higher_is_better else green_line

    fill_colors = [fill_color_up if y > 0 else fill_color_down for y in contribs]
    line_colors = [line_color_up if y > 0 else line_color_down for y in contribs]
    if include_base_value:
        fill_colors[0] = yellow_fill
        line_colors[0] = yellow_line
    if include_prediction:
        fill_colors[-1] = blue_fill
        line_colors[-1] = blue_line

    if orientation == "horizontal":
        cols = cols[::-1]
        values = values[::-1]
        contribs = contribs[::-1]
        bases = bases[::-1]
        fill_colors = fill_colors[::-1]
        line_colors = line_colors[::-1]
        hover_text = hover_text[::-1]

    # Base of each bar
    trace0 = go.Bar(
        x=bases if orientation == "horizontal" else cols,
        y=cols if orientation == "horizontal" else bases,
        hoverinfo="skip",
        name="",
        marker=dict(
            color="rgba(1,1,1, 0.0)",
        ),
        orientation="h" if orientation == "horizontal" else None,
    )

    # top of each bar (base + contribution)
    trace1 = go.Bar(
        x=contribs if orientation == "horizontal" else cols,
        y=cols if orientation == "horizontal" else contribs,
        text=hover_text,
        name="contribution",
        hoverinfo="text",
        marker=dict(
            # blue if positive contribution, red if negative
            color=fill_colors,
            line=dict(
                color=line_colors,
                width=2,
            ),
        ),
        orientation="h" if orientation == "horizontal" else None,
    )

    if model_output == "probability":
        title = f"Contribution to prediction probability = {prediction}%"
    elif model_output == "logodds":
        title = f"Contribution to prediction logodds = {prediction}"
    else:
        title = f"Contribution to prediction {target} = {prediction} {units}"

    data = [trace0, trace1]
    layout = go.Layout(
        height=600 if orientation == "vertical" else 100 + 35 * len(cols),
        title=title,
        barmode="stack",
        plot_bgcolor="#fff",
        showlegend=False,
    )

    fig = go.Figure(data=data, layout=layout)
    if (
        model_output == "probability"
        and base_value is not None
        and base_value > 0.3
        and base_value < 0.7
    ):
        # stretch out probability axis to entire probability range (0-100)
        if orientation == "vertical":
            fig.update_yaxes(range=[0, 100])
        elif orientation == "horizontal":
            fig.update_xaxes(range=[0, 100])

    fig.update_layout(
        margin=go.layout.Margin(
            l=longest_feature_name * 7 if orientation == "horizontal" else 40,
            r=40,
            b=40 if orientation == "horizontal" else longest_feature_name * 6,
            t=40,
            pad=4,
        ),
        title_x=0.5,
    )

    if orientation == "vertical":
        fig.update_yaxes(
            title_text="Predicted " + ("%" if model_output == "probability" else units)
        )
    else:
        fig.update_xaxes(
            title_text="Predicted " + ("%" if model_output == "probability" else units)
        )
    return fig


def plotly_precision_plot(precision_df, cutoff=None, labels=None, pos_label=None):
    """Returns a plotly figure with average predicted probability and
    percentage positive per probability bin.

    Args:
        precision_df (pd.DataFrame): generated with get_precision_df(..)
        cutoff (float, optional): Model cutoff to display in graph. Defaults to None.
        labels (List[str], optional): Labels for prediction classes. Defaults to None.
        pos_label (int, optional): For multiclass classifiers: which class to treat
        as positive class. Defaults to None.

    Returns:
        Plotly fig
    """

    label = (
        labels[pos_label]
        if labels is not None and pos_label is not None
        else "positive"
    )

    precision_df = precision_df.copy()

    spacing = 0.1 / len(precision_df)
    bin_widths = precision_df["bin_width"] - spacing
    bin_widths[bin_widths < 0.005] = 0.005

    trace1 = go.Bar(
        x=(0.5 * (precision_df["p_min"] + precision_df["p_max"])).values,
        y=precision_df["count"].values,
        width=bin_widths.values.astype("float"),
        name="counts",
    )

    data = [trace1]

    if "precision_0" in precision_df.columns.tolist():
        # if a pred_proba with probability for every class gets passed
        # to get_precision_df, it generates a precision for every class
        # in every bin as well.
        precision_cols = [
            col for col in precision_df.columns.tolist() if col.startswith("precision_")
        ]
        if labels is None:
            labels = ["class " + str(i) for i in range(len(precision_cols))]
        if pos_label is not None:
            # add the positive class first with thick line
            trace = go.Scatter(
                x=precision_df["p_avg"].values.tolist(),
                y=precision_df["precision_" + str(pos_label)].values.tolist(),
                name=labels[pos_label] + "(positive class)",
                line=dict(width=4),
                yaxis="y2",
            )
            data.append(trace)

        for i, precision_col in enumerate(precision_cols):
            # add the rest of the classes with thin lines
            if pos_label is None or i != pos_label:
                trace = go.Scatter(
                    x=precision_df["p_avg"].values.tolist(),
                    y=precision_df[precision_col].values.tolist(),
                    name=labels[i],
                    line=dict(width=2),
                    yaxis="y2",
                )
                data.append(trace)
    else:
        trace2 = go.Scatter(
            x=precision_df["p_avg"].values.tolist(),
            y=precision_df["precision"].values.tolist(),
            name="percentage " + label,
            yaxis="y2",
        )
        data = [trace1, trace2]

    layout = go.Layout(
        title=f"percentage {label} vs predicted probability",
        yaxis=dict(title="counts"),
        yaxis2=dict(
            title="percentage",
            titlefont=dict(color="rgb(148, 103, 189)"),
            tickfont=dict(color="rgb(148, 103, 189)"),
            overlaying="y",
            side="right",
            rangemode="tozero",
        ),
        xaxis=dict(title="predicted probability"),
        plot_bgcolor="#fff",
    )

    if cutoff is not None:
        layout["shapes"] = [
            dict(
                type="line",
                xref="x",
                yref="y2",
                x0=cutoff,
                x1=cutoff,
                y0=0,
                y1=1.0,
            )
        ]

    fig = go.Figure(data=data, layout=layout)
    fig.update_layout(
        legend=dict(orientation="h", xanchor="center", y=-0.2, x=0.5),
        margin=dict(t=40, b=40, l=40, r=40),
    )
    if cutoff is not None:
        fig.update_layout(
            annotations=[
                go.layout.Annotation(
                    x=cutoff, y=0.1, yref="y2", text=f"cutoff={cutoff}"
                )
            ]
        )
    return fig


def plotly_classification_plot(
    classification_df: pd.DataFrame, round: int = 2, percentage: bool = False
):
    """Displays bar plots showing label distributions above and below cutoff
    value.

    Args:
        classification_df (pd.DataFrame): generated with explainer.get_classification_df()
        round (int, optional): round float by round number of digits. Defaults to 2.
        percentage (bool, optional): Display percentage instead of absolute
            numbers. Defaults to False.

    Returns:
        Plotly fig
    """

    x = ["below cutoff", "above cutoff", "total"]

    fig = go.Figure()
    col_sums = classification_df.sum(axis=0)

    for label, below, above, total in classification_df.itertuples():
        below_perc = 100 * below / col_sums["below"]
        above_perc = 100 * above / col_sums["above"]
        total_perc = 100 * total / col_sums["total"]
        if percentage:
            fig.add_trace(
                go.Bar(
                    x=x,
                    y=[below_perc, above_perc, total_perc],
                    text=[
                        f"<b>{below}</b> ({below_perc:.{round}f}%)",
                        f"<b>{above}</b> ({above_perc:.{round}f}%)",
                        f"<b>{total}</b> ({total_perc:.{round}f}%)",
                    ],
                    textposition="auto",
                    hoverinfo="text",
                    name=label,
                )
            )
        else:
            fig.add_trace(
                go.Bar(
                    x=x,
                    y=[below, above, total],
                    text=[
                        f"<b>{below}</b> ({below_perc:.{round}f}%)",
                        f"<b>{above}</b> ({above_perc:.{round}f}%)",
                        f"<b>{total}</b> ({total_perc:.{round}f}%)",
                    ],
                    textposition="auto",
                    hoverinfo="text",
                    name=label,
                )
            )
    if percentage:
        fig.update_layout(title="Percentage above and below cutoff")
    else:
        fig.update_layout(title="Total above and below cutoff")

    fig.update_layout(barmode="stack", margin=dict(t=40, b=40, l=40, r=40))

    fig.update_layout(legend=dict(orientation="h", xanchor="center", y=-0.2, x=0.5))
    return fig


def plotly_lift_curve(
    lift_curve_df, cutoff=None, percentage=False, add_wizard=True, round=2
):
    """returns a lift plot for values

    Args:
        lift_curve_df (pd.DataFrame): generated with get_liftcurve_df(pred_proba, y)
        cutoff (float, optional): cutoff above which samples get classified as
            positive. Defaults to None.
        percentage (bool, optional): Display percentages instead of absolute
            numbers along axis. Defaults to False.
        add_wizard (bool, optional): Add a line indicating how a perfect model
            would perform ("the wizard"). Defaults to True
        round (int, optional): Rounding to apply to floats. Defaults to 2.

    Returns:
        Plotly fig
    """

    if percentage:
        model_text = [
            f"model selected {pos:.{round}f}% of all positives in first {i:.{round}f}% sampled<br>"
            + f"precision={precision:.{round}f}% positives in sample<br>"
            + f"lift={pos/exp:.{round}f}"
            for (i, pos, exp, precision) in zip(
                lift_curve_df.index_percentage,
                lift_curve_df.cumulative_percentage_pos,
                lift_curve_df.random_cumulative_percentage_pos,
                lift_curve_df.precision,
            )
        ]

        random_text = [
            f"random selected {exp:.{round}f}% of all positives in first {i:.{round}f}% sampled<br>"
            + f"precision={precision:.{round}f}% positives in sample"
            for (i, pos, exp, precision) in zip(
                lift_curve_df.index_percentage,
                lift_curve_df.cumulative_percentage_pos,
                lift_curve_df.random_cumulative_percentage_pos,
                lift_curve_df.random_precision,
            )
        ]
    else:
        model_text = [
            f"model selected {pos} positives out of {i}<br>"
            + f"precision={precision:.{round}f}<br>"
            + f"lift={pos/exp:.{round}f}"
            for (i, pos, exp, precision) in zip(
                lift_curve_df["index"],
                lift_curve_df.positives,
                lift_curve_df.random_pos,
                lift_curve_df.precision,
            )
        ]
        random_text = [
            f"random selected {int(exp)} positives out of {i}<br>"
            + f"precision={precision:.{round}f}"
            for (i, pos, exp, precision) in zip(
                lift_curve_df["index"],
                lift_curve_df.positives,
                lift_curve_df.random_pos,
                lift_curve_df.random_precision,
            )
        ]

    trace0 = go.Scatter(
        x=lift_curve_df["index_percentage"].values
        if percentage
        else lift_curve_df["index"],
        y=np.round(lift_curve_df.cumulative_percentage_pos.values, round)
        if percentage
        else np.round(lift_curve_df.positives.values, round),
        name="model",
        text=model_text,
        hoverinfo="text",
    )

    trace1 = go.Scatter(
        x=lift_curve_df["index_percentage"].values
        if percentage
        else lift_curve_df["index"],
        y=np.round(lift_curve_df.random_cumulative_percentage_pos.values, round)
        if percentage
        else np.round(lift_curve_df.random_pos.values, round),
        name="random",
        text=random_text,
        hoverinfo="text",
    )
    if add_wizard:
        if percentage:
            trace2 = go.Scatter(
                x=[0.0, lift_curve_df.random_precision[0], 100],
                y=[0.0, 100, 100],
                text=[
                    "0%, 0%",
                    f"{lift_curve_df.random_precision[0]:.2f}%, 100%",
                    "100, 100%",
                ],
                name="perfect",
                hoverinfo="text",
            )
        else:
            trace2 = go.Scatter(
                x=[0.0, lift_curve_df["positives"].max(), lift_curve_df["index"].max()],
                y=[
                    0.0,
                    lift_curve_df["positives"].max(),
                    lift_curve_df["positives"].max(),
                ],
                name="perfect",
            )

        data = [trace2, trace0, trace1]
    else:
        data = [trace0, trace1]

    fig = go.Figure(data)

    fig.update_layout(
        title=dict(text="Lift curve", x=0.5, font=dict(size=18)),
        xaxis_title="Percentage sampled" if percentage else "Number sampled",
        yaxis_title="Percentage of positive" if percentage else "Number of positives",
        xaxis=dict(spikemode="across"),
        hovermode="x",
        plot_bgcolor="#fff",
    )

    fig.update_layout(legend=dict(xanchor="center", y=0.9, x=0.1))
    if percentage:
        fig.update_layout(xaxis=dict(range=[0, 100]))
    else:
        fig.update_layout(xaxis=dict(range=[0, lift_curve_df["index"].max()]))

    if cutoff is not None:
        # cutoff_idx = max(0, (np.abs(lift_curve_df.pred_proba - cutoff)).argmin() - 1)
        cutoff_idx = max(0, len(lift_curve_df[lift_curve_df.pred_proba >= cutoff]) - 1)
        if percentage:
            cutoff_x = lift_curve_df["index_percentage"].iloc[cutoff_idx]
        else:
            cutoff_x = lift_curve_df["index"].iloc[cutoff_idx]

        cutoff_n = lift_curve_df["index"].iloc[cutoff_idx]
        cutoff_pos = lift_curve_df["positives"].iloc[cutoff_idx]
        cutoff_random_pos = int(lift_curve_df["random_pos"].iloc[cutoff_idx])
        cutoff_lift = np.round(
            lift_curve_df["positives"].iloc[cutoff_idx]
            / lift_curve_df.random_pos.iloc[cutoff_idx],
            1,
        )
        cutoff_precision = np.round(lift_curve_df["precision"].iloc[cutoff_idx], 2)
        cutoff_random_precision = np.round(
            lift_curve_df["random_precision"].iloc[cutoff_idx], 2
        )

        fig.update_layout(
            shapes=[
                dict(
                    type="line",
                    xref="x",
                    yref="y",
                    x0=cutoff_x,
                    x1=cutoff_x,
                    y0=0,
                    y1=100.0 if percentage else lift_curve_df.positives.max(),
                )
            ]
        )
        fig.update_layout(
            annotations=[
                go.layout.Annotation(
                    x=cutoff_x, y=5, yref="y", text=f"cutoff={cutoff:.{round}f}"
                ),
                go.layout.Annotation(
                    x=0.5,
                    y=0.4,
                    text=f"Model: {cutoff_pos} out {cutoff_n} ({cutoff_precision:.{round}f}%)",
                    showarrow=False,
                    align="right",
                    xref="paper",
                    yref="paper",
                    xanchor="left",
                    yanchor="top",
                ),
                go.layout.Annotation(
                    x=0.5,
                    y=0.33,
                    text=f"Random: {cutoff_random_pos} out {cutoff_n} ({cutoff_random_precision:.{round}f}%)",
                    showarrow=False,
                    align="right",
                    xref="paper",
                    yref="paper",
                    xanchor="left",
                    yanchor="top",
                ),
                go.layout.Annotation(
                    x=0.5,
                    y=0.26,
                    text=f"Lift: {cutoff_lift}",
                    showarrow=False,
                    align="right",
                    xref="paper",
                    yref="paper",
                    xanchor="left",
                    yanchor="top",
                ),
            ]
        )
    fig.update_layout(margin=dict(t=40, b=40, l=40, r=40))
    return fig


def plotly_cumulative_precision_plot(
    lift_curve_df, labels=None, percentile=None, round=2, pos_label=1
):
    """Return cumulative precision plot showing the expected label distribution
    if you cumulatively sample a more and more of the highest predicted samples.

    Args:
        lift_curve_df (pd.DataFrame): generated with get_liftcurve_df(...)
        labels (List[str], optional): list of labels for classes. Defaults to None.
        percentile (float, optional): draw line at percentile, defaults to None
        round (int, optional): round floats to digits. Defaults to 2.
        pos_label (int, optional): Positive class label. Defaults to 1.

    Returns:
        Plotly fig
    """
    if labels is None:
        labels = ["category " + str(i) for i in range(lift_curve_df.y.max() + 1)]
    fig = go.Figure()
    text = [
        f"percentage sampled = top {idx_perc:.{round}f}%"
        for idx_perc in lift_curve_df["index_percentage"].values
    ]
    fig = fig.add_trace(
        go.Scatter(
            x=lift_curve_df.index_percentage,
            y=np.zeros(len(lift_curve_df)),
            showlegend=False,
            text=text,
            hoverinfo="text",
        )
    )

    text = [
        f"percentage {labels[pos_label]}={perc:.{round}f}%"
        for perc in lift_curve_df["precision_" + str(pos_label)].values
    ]
    fig = fig.add_trace(
        go.Scatter(
            x=lift_curve_df.index_percentage,
            y=lift_curve_df["precision_" + str(pos_label)].values,
            fill="tozeroy",
            name=labels[pos_label],
            text=text,
            hoverinfo="text",
        )
    )

    cumulative_y = lift_curve_df["precision_" + str(pos_label)].values
    for y_label in range(pos_label, lift_curve_df.y.max() + 1):
        if y_label != pos_label:
            cumulative_y = (
                cumulative_y + lift_curve_df["precision_" + str(y_label)].values
            )
            text = [
                f"percentage {labels[y_label]}={perc:.{round}f}%"
                for perc in lift_curve_df["precision_" + str(y_label)].values
            ]
            fig = fig.add_trace(
                go.Scatter(
                    x=lift_curve_df.index_percentage,
                    y=cumulative_y,
                    fill="tonexty",
                    name=labels[y_label],
                    text=text,
                    hoverinfo="text",
                )
            )

    for y_label in range(0, pos_label):
        if y_label != pos_label:
            cumulative_y = (
                cumulative_y + lift_curve_df["precision_" + str(y_label)].values
            )
            text = [
                f"percentage {labels[y_label]}={perc:.{round}f}%"
                for perc in lift_curve_df["precision_" + str(y_label)].values
            ]
            fig = fig.add_trace(
                go.Scatter(
                    x=lift_curve_df.index_percentage,
                    y=cumulative_y,
                    fill="tonexty",
                    name=labels[y_label],
                    text=text,
                    hoverinfo="text",
                )
            )

    fig.update_layout(
        title=dict(
            text="Cumulative percentage per category<br>when sampling top X%",
            x=0.5,
            font=dict(size=18),
        ),
        yaxis=dict(title="Cumulative precision per category"),
        xaxis=dict(title="Top X% model scores", spikemode="across", range=[0, 100]),
        hovermode="x",
        plot_bgcolor="#fff",
    )

    if percentile is not None:
        fig.update_layout(
            shapes=[
                dict(
                    type="line",
                    xref="x",
                    yref="y",
                    x0=100 * percentile,
                    x1=100 * percentile,
                    y0=0,
                    y1=100.0,
                )
            ]
        )
        fig.update_layout(
            annotations=[
                go.layout.Annotation(
                    x=100 * percentile,
                    y=20,
                    yref="y",
                    ax=60,
                    text=f"percentile={100*percentile:.{round}f}",
                )
            ]
        )
    fig.update_xaxes(nticks=10)
    fig.update_layout(margin=dict(t=40, b=40, l=40, r=40))
    return fig


def plotly_dependence_plot(
    X_col,
    shap_values,
    interact_col=None,
    interaction=False,
    na_fill=-999,
    round=3,
    units="",
    highlight_index=None,
    idxs=None,
):
    """Returns a dependence plot showing the relationship between feature col_name
    and shap values for col_name. Do higher values of col_name increase prediction
    or decrease them? Or some kind of U-shape or other?

    Args:
        X_col (pd.Series): pd.Series with column values.
        shap_values (np.ndarray): shap values generated for X_col
        interact_col (pd.Series): pd.Series with column marker values. Defaults to None.
        interaction (bool, optional): Is this a plot of shap interaction values?
            Defaults to False.
        na_fill (int, optional): value used for filling missing values.
            Defaults to -999.
        round (int, optional): Rounding to apply to floats. Defaults to 2.
        units (str, optional): Units of the target variable. Defaults to "".
        highlight_index (str, int, optional): index row of X to highlight in
            the plot. Defaults to None.
        idxs (pd.Index, optional): list of descriptors of the index, e.g.
            names or other identifiers. Defaults to None.

    Returns:
        Plotly fig
    """
    assert len(X_col) == len(
        shap_values
    ), f"Column(len={len(X_col)}) and Shap values(len={len(shap_values)}) and should have the same length!"
    if idxs is not None:
        assert len(idxs) == X_col.shape[0]
        idxs = pd.Index(idxs).astype(str)
    else:
        idxs = X_col.index.astype(str)

    if highlight_index is not None:
        if isinstance(highlight_index, int):
            highlight_idx = highlight_index
            highlight_name = idxs[highlight_idx]
        elif isinstance(highlight_index, str):
            assert (
                highlight_index in idxs
            ), f"highlight_index should be int or in idxs, {highlight_index} is neither!"
            highlight_idx = idxs.get_loc(highlight_index)
            highlight_name = highlight_index

    col_name = X_col.name

    if interact_col is not None:
        text = np.array(
            [
                f"{idxs.name}={index}<br>{X_col.name}={col_val}<br>{interact_col.name}={col_col_val}<br>SHAP={shap_val:.{round}f}"
                for index, col_val, col_col_val, shap_val in zip(
                    idxs, X_col, interact_col, shap_values
                )
            ]
        )
    else:
        text = np.array(
            [
                f"{idxs.name}={index}<br>{X_col.name}={col_val}<br>SHAP={shap_val:.{round}f}"
                for index, col_val, shap_val in zip(idxs, X_col, shap_values)
            ]
        )

    data = []

    X_col = X_col.copy().replace({na_fill: np.nan})
    y = shap_values
    if interact_col is not None and not is_numeric_dtype(interact_col):
        for onehot_col in interact_col.unique().tolist():
            data.append(
                go.Scattergl(
                    x=X_col[interact_col == onehot_col].replace({na_fill: np.nan}),
                    y=shap_values[interact_col == onehot_col],
                    mode="markers",
                    marker=dict(size=7, showscale=False, opacity=0.6),
                    showlegend=True,
                    opacity=0.8,
                    hoverinfo="text",
                    name=onehot_col,
                    text=[
                        f"{idxs.name}={index}<br>{X_col.name}={col_val}<br>{interact_col.name}={interact_val}<br>SHAP={shap_val:.{round}f}"
                        for index, col_val, interact_val, shap_val in zip(
                            idxs,
                            X_col[interact_col == onehot_col],
                            interact_col[interact_col == onehot_col],
                            shap_values[interact_col == onehot_col],
                        )
                    ],
                )
            )
    elif interact_col is not None and is_numeric_dtype(interact_col):
        if na_fill in interact_col:
            data.append(
                go.Scattergl(
                    x=X_col[interact_col != na_fill],
                    y=shap_values[interact_col != na_fill],
                    mode="markers",
                    text=text[interact_col != na_fill],
                    hoverinfo="text",
                    marker=dict(
                        size=7,
                        opacity=0.6,
                        color=interact_col[interact_col != na_fill],
                        colorscale="Bluered",
                        colorbar=dict(title=interact_col.name),
                        showscale=True,
                    ),
                )
            )
            data.append(
                go.Scattergl(
                    x=X_col[interact_col == na_fill],
                    y=shap_values[interact_col == na_fill],
                    mode="markers",
                    text=text[interact_col == na_fill],
                    hoverinfo="text",
                    marker=dict(size=7, opacity=0.35, color="grey"),
                )
            )
        else:
            data.append(
                go.Scattergl(
                    x=X_col,
                    y=shap_values,
                    mode="markers",
                    text=text,
                    hoverinfo="text",
                    marker=dict(
                        size=7,
                        opacity=0.6,
                        color=interact_col,
                        colorscale="Bluered",
                        colorbar=dict(title=interact_col.name),
                        showscale=True,
                    ),
                )
            )

    else:
        data.append(
            go.Scattergl(
                x=X_col,
                y=shap_values,
                mode="markers",
                text=text,
                hoverinfo="text",
                marker=dict(size=7, opacity=0.6),
            )
        )

    if interaction:
        title = f"Interaction plot for {X_col.name} and {interact_col.name}"
    else:
        title = f"Dependence plot for {X_col.name}"

    layout = go.Layout(
        title=title,
        paper_bgcolor="#fff",
        plot_bgcolor="#fff",
        showlegend=False,
        hovermode="closest",
        xaxis=dict(title=col_name),
        yaxis=dict(title=f"SHAP value ({units})" if units != "" else "SHAP value"),
    )

    fig = go.Figure(data, layout)

    if interact_col is not None and not is_numeric_dtype(interact_col):
        fig.update_layout(showlegend=True)

    if highlight_index is not None:
        fig.add_trace(
            go.Scattergl(
                x=[X_col.iloc[highlight_idx]],
                y=[shap_values[highlight_idx]],
                mode="markers",
                marker=dict(
                    color="LightSkyBlue",
                    size=25,
                    opacity=0.5,
                    line=dict(color="MediumPurple", width=4),
                ),
                name=f"{idxs.name} {highlight_name}",
                text=f"{idxs.name} {highlight_name}",
                hoverinfo="text",
                showlegend=False,
            ),
        )
    fig.update_traces(selector=dict(mode="markers"))
    fig.update_layout(margin=dict(t=40, b=40, l=40, r=40))
    return fig


def plotly_shap_violin_plot(
    X_col,
    shap_values,
    X_color_col=None,
    points=False,
    interaction=False,
    units="",
    highlight_index=None,
    idxs=None,
    round=3,
    cats_order=None,
    max_cat_colors=5,
):  # sourcery no-metrics
    """Generates a violin plot for displaying shap value distributions for
    categorical features.

    Args:
        X_col (pd.DataFrame): dataframe of input rows
        shap_values (np.ndarray): shap values generated for X
        col_name (str): Column of X to display violin plot for
        color_col (str, optional): Column of X to color plot markers by.
            Defaults to None.
        points (bool, optional): display point cloud next to violin plot.
            Defaults to False.
        interaction (bool, optional): Is this a plot for shap_interaction_values?
            Defaults to False.
        units (str, optional): Units of target variable. Defaults to "".
        highlight_index (int, str, optional): Row index to highligh. Defaults to None.
        idxs (List[str], optional): List of identifiers for each row in X, e.g.
            names or id's. Defaults to None.
        cats_order (list, optional): list of categories to display. If None
            defaults to X_col.unique().tolist() so displays all categories.
        max_cat_colors (int, optional): maximum number of X_color_col categories
            to colorize in scatter plot next to violin plot. Defaults to 5.

    Returns:
        Plotly fig
    """

    assert not is_numeric_dtype(
        X_col
    ), f"{X_col.name} is not categorical! Can only plot violin plots for categorical features!"

    if cats_order is None:
        cats_order = sorted(X_col.unique().tolist())

    n_cats = len(cats_order)

    if idxs is not None:
        assert len(idxs) == X_col.shape[0] == len(shap_values)
        idxs = pd.Index(idxs).astype(str)
    else:
        idxs = X_col.index.astype(str)

    if highlight_index is not None:
        if isinstance(highlight_index, int):
            highlight_idx = highlight_index
            highlight_name = idxs[highlight_idx]
        elif isinstance(highlight_index, str):
            assert (
                highlight_index in idxs
            ), f"highlight_index should be int or in idxs, {highlight_index} is neither!"
            highlight_idx = idxs.get_loc(highlight_index)
            highlight_name = highlight_index

    if points or X_color_col is not None:
        fig = make_subplots(
            rows=1, cols=2 * n_cats, column_widths=[3, 1] * n_cats, shared_yaxes=True
        )
        showscale = True
    else:
        fig = make_subplots(rows=1, cols=n_cats, shared_yaxes=True)

    shap_range = shap_values.max() - shap_values.min()
    fig.update_yaxes(
        range=[
            shap_values.min() - 0.1 * shap_range,
            shap_values.max() + 0.1 * shap_range,
        ]
    )

    if X_color_col is not None:
        color_cats = list(X_color_col.value_counts().index[:max_cat_colors])
        n_color_cats = len(color_cats)
        colors = [
            "#636EFA",
            "#EF553B",
            "#00CC96",
            "#AB63FA",
            "#FFA15A",
            "#19D3F3",
            "#FF6692",
            "#B6E880",
            "#FF97FF",
            "#FECB52",
        ]
        colors = colors * (1 + int(n_color_cats / len(colors)))
        colors = colors[:n_color_cats]
        show_legend = set(color_cats + ["Category_Other"])

    for i, cat in enumerate(cats_order):
        col = 1 + i * 2 if points or X_color_col is not None else 1 + i
        if cat.startswith(X_col.name + "_"):
            cat_name = cat[len(X_col.name) + 1 :]
        else:
            cat_name = cat
        fig.add_trace(
            go.Violin(
                x=np.repeat(cat_name, len(X_col[X_col == cat])),
                y=shap_values[X_col == cat],
                name=cat_name,
                box_visible=True,
                meanline_visible=True,
                showlegend=False,
            ),
            row=1,
            col=col,
        )
        if X_color_col is not None:
            if is_numeric_dtype(X_color_col):
                fig.add_trace(
                    go.Scattergl(
                        x=np.random.randn((X_col == cat).sum()),
                        y=shap_values[X_col == cat],
                        name=X_color_col.name,
                        mode="markers",
                        showlegend=False,
                        hoverinfo="text",
                        text=[
                            f"{idxs.name}: {index}<br>shap: {shap:.{round}f}<br>{X_color_col.name}: {col}"
                            for index, shap, col in zip(
                                idxs[X_col == cat],
                                shap_values[X_col == cat],
                                X_color_col[X_col == cat],
                            )
                        ],
                        marker=dict(
                            size=7,
                            opacity=0.3,
                            cmin=X_color_col.min(),
                            cmax=X_color_col.max(),
                            color=X_color_col[X_col == cat],
                            colorscale="Bluered",
                            showscale=showscale,
                            colorbar=dict(title=X_color_col.name),
                        ),
                    ),
                    row=1,
                    col=col + 1,
                )
            else:
                for color_cat, color in zip(color_cats, colors):
                    if color_cat.startswith(X_color_col.name + "_"):
                        color_cat_name = color_cat[len(X_color_col.name) + 1 :]
                    else:
                        color_cat_name = color_cat

                    fig.add_trace(
                        go.Scattergl(
                            x=np.random.randn(
                                ((X_col == cat) & (X_color_col == color_cat)).sum()
                            ),
                            y=shap_values[(X_col == cat) & (X_color_col == color_cat)],
                            name=color_cat_name,
                            mode="markers",
                            showlegend=color_cat in show_legend,
                            hoverinfo="text",
                            text=[
                                f"{idxs.name}: {index}<br>shap: {shap:.{round}f}<br>{X_color_col.name}: {color_cat_name}"
                                for index, shap in zip(
                                    idxs[(X_col == cat) & (X_color_col == color_cat)],
                                    shap_values[
                                        (X_col == cat) & (X_color_col == color_cat)
                                    ],
                                )
                            ],
                            marker=dict(size=7, opacity=0.3, color=color),
                        ),
                        row=1,
                        col=col + 1,
                    )
                    if color_cat in X_color_col[X_col == cat].unique():
                        show_legend = show_legend - {color_cat}
                if X_color_col.nunique() > max_cat_colors:
                    fig.add_trace(
                        go.Scattergl(
                            x=np.random.randn(
                                ((X_col == cat) & (~X_color_col.isin(color_cats))).sum()
                            ),
                            y=shap_values[
                                (X_col == cat) & (~X_color_col.isin(color_cats))
                            ],
                            name="Other",
                            mode="markers",
                            showlegend="Category_Other" in show_legend,
                            hoverinfo="text",
                            text=[
                                f"{idxs.name}: {index}<br>shap: {shap:.{round}f}<br>{X_color_col.name}: {col}"
                                for index, shap, col in zip(
                                    idxs[
                                        (X_col == cat) & (~X_color_col.isin(color_cats))
                                    ],
                                    shap_values[
                                        (X_col == cat) & (~X_color_col.isin(color_cats))
                                    ],
                                    X_color_col[
                                        (X_col == cat) & (~X_color_col.isin(color_cats))
                                    ],
                                )
                            ],
                            marker=dict(size=7, opacity=0.3, color="grey"),
                        ),
                        row=1,
                        col=col + 1,
                    )
                    show_legend = show_legend - {"Category_Other"}

            showscale = False
        elif points:
            fig.add_trace(
                go.Scattergl(
                    x=np.random.randn((X_col == cat).sum()),
                    y=shap_values[X_col == cat],
                    mode="markers",
                    showlegend=False,
                    hoverinfo="text",
                    text=[
                        f"{idxs.name}: {index}<br>shap: {shap}"
                        for index, shap in zip(
                            idxs[(X_col == cat)], shap_values[X_col == cat]
                        )
                    ],
                    marker=dict(size=7, opacity=0.6, color="blue"),
                ),
                row=1,
                col=col + 1,
            )
        if (
            highlight_index is not None
            and (points or X_color_col is not None)
            and X_col[highlight_idx] == cat
        ):
            fig.add_trace(
                go.Scattergl(
                    x=[0],
                    y=[shap_values[highlight_idx]],
                    mode="markers",
                    marker=dict(
                        color="LightSkyBlue",
                        size=25,
                        opacity=0.5,
                        line=dict(color="MediumPurple", width=4),
                    ),
                    name=f"{idxs.name} {highlight_name}",
                    text=f"{idxs.name} {highlight_name}",
                    hoverinfo="text",
                    showlegend=False,
                ),
                row=1,
                col=col + 1,
            )

    if points or X_color_col is not None:
        for i in range(n_cats):
            fig.update_xaxes(
                showgrid=False, zeroline=False, visible=False, row=1, col=2 + i * 2
            )
            fig.update_yaxes(showgrid=False, zeroline=False, row=1, col=2 + i * 2)

    fig.update_layout(
        yaxis=dict(title=f"SHAP value ({units})" if units != "" else "SHAP value"),
        hovermode="closest",
    )

    if X_color_col is not None and interaction:
        fig.update_layout(
            title=f"Interaction plot for {X_col.name} and {X_color_col.name}"
        )
    elif X_color_col is not None:
        fig.update_layout(
            title=f"Shap values for {X_col.name}<br>(colored by {X_color_col.name})"
        )
    else:
        fig.update_layout(title=f"Shap values for {X_col.name}")
    fig.update_layout(margin=dict(t=40, b=40, l=40, r=40))
    return fig


def plotly_pdp(
    pdp_df,
    display_index=None,
    index_feature_value=None,
    index_prediction=None,
    absolute=True,
    plot_lines=True,
    num_grid_lines=100,
    feature_name=None,
    round=2,
    target="",
    units="",
    index_name="index",
):
    """Display partial-dependence plot (pdp)

    Args:
        pdp_df (pd.DataFrame): Generated from get_pdp_df()
        display_index (int, str, optional): Index to highligh in plot.
            Defaults to None.
        index_feature_value (str, float, optional): value of feature for index.
            Defaults to None.
        index_prediction (float, optional): Baseline prediction for index.
            Defaults to None.
        absolute (bool, optional): Display absolute pdp lines. If false then
            display relative to base. Defaults to True.
        plot_lines (bool, optional): Display selection of individual pdp lines.
            Defaults to True.
        num_grid_lines (int, optional): Number of sample gridlines to display.
            Defaults to 100.
        feature_name (str, optional): Name of the feature. Defaults to None.
        round (int, optional): Rounding to apply to floats. Defaults to 2.
        target (str, optional): Name of target variables. Defaults to "".
        units (str, optional): Units of target variable. Defaults to "".
        index_name (str): identifier for idxs. Defaults to "index".

    Returns:
        Plotly fig
    """
    if absolute:
        pdp_mean = pdp_df.mean().round(round).values
    else:
        pdp_mean = (
            pdp_df.mean().round(round).values - pdp_df.mean().round(round).values[0]
        )

    trace0 = go.Scatter(
        x=pdp_df.columns.values,
        y=pdp_mean,
        mode="lines+markers",
        line=dict(color="grey", width=4),
        name=f"average prediction <br>for different values of <br>{feature_name}",
    )
    data = [trace0]

    if display_index is not None:
        trace1 = go.Scatter(
            x=pdp_df.columns.values,
            y=pdp_df.iloc[[display_index]].round(round).values[0]
            if absolute
            else pdp_df.iloc[[display_index]].round(round).values[0]
            - pdp_df.iloc[[display_index]].values[0, 0],
            mode="lines+markers",
            line=dict(color="blue", width=4),
            name=f"prediction for {index_name} {display_index} <br>for different values of <br>{feature_name}",
        )
        data.append(trace1)
    if plot_lines:
        x = pdp_df.columns.values
        pdp_sample = pdp_df.sample(min(num_grid_lines, len(pdp_df)))
        ice_lines = (
            pdp_sample.values
            if absolute
            else pdp_sample.values
            - np.expand_dims(pdp_sample.iloc[:, 0].values, axis=1)
        )

        for row in pdp_sample.itertuples(index=False):
            data.append(
                go.Scatter(
                    x=x,
                    y=tuple(row),
                    mode="lines",
                    hoverinfo="skip",
                    line=dict(color="grey"),
                    opacity=0.1,
                    showlegend=False,
                )
            )

    layout = go.Layout(
        title=f"pdp plot for {feature_name}",
        plot_bgcolor="#fff",
        yaxis=dict(title=f"Predicted {target}{f' ({units})' if units else ''}"),
        xaxis=dict(title=feature_name),
    )

    fig = go.Figure(data=data, layout=layout)
    shapes = []
    annotations = []

    if index_feature_value is not None:
        if not isinstance(index_feature_value, str):
            index_feature_value = np.round(index_feature_value, 2)

        shapes.append(
            dict(
                type="line",
                xref="x",
                yref="y",
                x0=index_feature_value,
                x1=index_feature_value,
                y0=pdp_sample.min().min() if plot_lines else pdp_mean.min(),
                y1=pdp_sample.max().max() if plot_lines else pdp_mean.max(),
                line=dict(
                    color="MediumPurple",
                    width=4,
                    dash="dot",
                ),
            )
        )
        annotations.append(
            go.layout.Annotation(
                x=index_feature_value,
                y=pdp_sample.min().min() if plot_lines else pdp_mean.min(),
                text=f"baseline value = {index_feature_value}",
            )
        )

    if index_prediction is not None:
        shapes.append(
            dict(
                type="line",
                xref="x",
                yref="y",
                x0=pdp_df.columns.values[0],
                x1=pdp_df.columns.values[-1],
                y0=index_prediction,
                y1=index_prediction,
                line=dict(
                    color="MediumPurple",
                    width=4,
                    dash="dot",
                ),
            )
        )

        annotations.append(
            go.layout.Annotation(
                x=pdp_df.columns[int(0.5 * len(pdp_df.columns))],
                y=index_prediction,
                text=f"baseline pred = {index_prediction:.{round}f}",
            )
        )

    fig.update_layout(annotations=annotations)
    fig.update_layout(shapes=shapes)
    fig.update_layout(showlegend=False)
    fig.update_layout(margin=dict(t=40, b=40, l=40, r=40))
    return fig


def plotly_importances_plot(
    importance_df,
    descriptions=None,
    round=3,
    target="target",
    units="",
    title=None,
    xaxis_title=None,
):
    """Return feature importance plot

    Args:
        importance_df (pd.DataFrame): generate with get_importance_df(...)
        descriptions (dict, optional): dict of descriptions of each feature.
        round (int, optional): Rounding to apply to floats. Defaults to 3.
        target (str, optional): Name of target variable. Defaults to "target".
        units (str, optional): Units of target variable. Defaults to "".
        title (str, optional): Title for graph. Defaults to None.
        xaxis_title (str, optional): Title for x-axis Defaults to None.

    Returns:
        Plotly fig
    """

    importance_name = importance_df.columns[
        1
    ]  # can be "MEAN_ABS_SHAP", "Permutation Importance", etc
    if title is None:
        title = importance_name
    longest_feature_name = importance_df["Feature"].str.len().max()

    imp = importance_df.sort_values(importance_name)

    feature_names = [
        str(len(imp) - i) + ". " + col
        for i, col in enumerate(imp.iloc[:, 0].astype(str).values.tolist())
    ]

    importance_values = imp.iloc[:, 1]

    data = [
        go.Bar(
            y=feature_names,
            x=importance_values,
            # text=importance_values.round(round),
            text=descriptions[::-1]
            if descriptions is not None
            else None,  # don't know why, but order needs to be reversed
            # textposition='inside',
            # insidetextanchor='end',
            hoverinfo="text",
            orientation="h",
        )
    ]

    layout = go.Layout(title=title, plot_bgcolor="#fff", showlegend=False)
    fig = go.Figure(data=data, layout=layout)
    fig.update_yaxes(automargin=True)
    if xaxis_title is None:
        xaxis_title = units
    fig.update_xaxes(automargin=True, title=xaxis_title)

    left_margin = longest_feature_name * 7
    if np.isnan(left_margin):
        left_margin = 100

    fig.update_layout(
        height=200 + len(importance_df) * 20,
        margin=go.layout.Margin(l=left_margin, r=40, b=40, t=40, pad=4),
    )
    return fig


def plotly_confusion_matrix(cm, labels=None, percentage=True, normalize="all"):
    """Generates Plotly fig confusion matrix

    Args:
        cm (np.ndarray): generated by sklearn.metrics.confusion_matrix(y_true, preds)
        labels (List[str], optional): List of labels for classes. Defaults to None.
        percentage (bool, optional): Display percentages on top of the counts.
            Defaults to True.
        normalize ({‘observed’, ‘pred’, ‘all’}): normalizes confusion matrix over
            the true (rows), predicted (columns) conditions or all the population.
            Defaults to 'all'.

    Returns:
        Plotly fig
    """

    if normalize not in ["observed", "pred", "all"]:
        raise ValueError(
            "Error! parameters normalize must be one of {'observed', 'pred', 'all'} !"
        )

    with np.errstate(all="ignore"):
        if normalize == "all":
            cm_normalized = np.round(100 * cm / cm.sum(), 1)
        elif normalize == "observed":
            cm_normalized = np.round(100 * cm / cm.sum(axis=1, keepdims=True), 1)
        elif normalize == "pred":
            cm_normalized = np.round(100 * cm / cm.sum(axis=0, keepdims=True), 1)

        cm_normalized = np.nan_to_num(cm_normalized)

    if labels is None:
        labels = [str(i) for i in range(cm.shape[0])]

    zmax = 130  # to keep the text readable at 100% accuracy

    data = [
        go.Heatmap(
            z=cm_normalized,
            x=[f" {lab}" for lab in labels],
            y=[f" {lab}" for lab in labels],
            hoverinfo="skip",
            zmin=0,
            zmax=zmax,
            colorscale="Blues",
            showscale=False,
        )
    ]

    layout = go.Layout(
        title="Confusion Matrix",
        xaxis=dict(
            title="predicted",
            constrain="domain",
            tickmode="array",
            showgrid=False,
            tickvals=[f" {lab}" for lab in labels],
            ticktext=[f" {lab}" for lab in labels],
        ),
        yaxis=dict(
            title=dict(text="observed", standoff=20),
            autorange="reversed",
            side="left",
            scaleanchor="x",
            scaleratio=1,
            showgrid=False,
            tickmode="array",
            tickvals=[f" {lab}" for lab in labels],
            ticktext=[f" {lab}" for lab in labels],
        ),
        plot_bgcolor="#fff",
    )
    fig = go.Figure(data, layout)
    annotations = []
    for x in range(cm.shape[0]):
        for y in range(cm.shape[1]):
            top_text = f"{cm_normalized[x, y]}%" if percentage else f"{cm[x, y]}"
            bottom_text = f"{cm_normalized[x, y]}%" if not percentage else f"{cm[x, y]}"
            annotations.extend(
                [
                    go.layout.Annotation(
                        x=fig.data[0].x[y],
                        y=fig.data[0].y[x],
                        text=top_text,
                        showarrow=False,
                        font=dict(size=20),
                    ),
                    go.layout.Annotation(
                        x=fig.data[0].x[y],
                        y=fig.data[0].y[x],
                        text=f" <br> <br> <br>({bottom_text})",
                        showarrow=False,
                        font=dict(size=12),
                    ),
                ]
            )
    longest_label = max([len(label) for label in labels])
    fig.update_layout(annotations=annotations)
    fig.update_layout(margin=dict(t=40, b=40, l=longest_label * 7, r=40))
    return fig


def plotly_roc_auc_curve(fpr, tpr, thresholds, score, cutoff=None, round=2):
    """Plot ROC AUC curve

    Args:
        fpr
        tpr
        thresholds
        cutoff (float, optional): Cutoff proba to display. Defaults to None.
        round (int, optional): rounding of floats. Defaults to 2.

    Returns:
        Plotly Fig:
    """

    trace0 = go.Scatter(
        x=fpr,
        y=tpr,
        mode="lines",
        name="ROC AUC CURVE",
        text=[
            f"threshold: {th:.{round}f} <br> FP: {fp:.{round}f} <br> TP: {tp:.{round}f}"
            for fp, tp, th in zip(fpr, tpr, thresholds)
        ],
        hoverinfo="text",
    )
    data = [trace0]
    layout = go.Layout(
        title="ROC AUC CURVE",
        #    width=450,
        #    height=450,
        xaxis=dict(title="False Positive Rate", range=[0, 1], constrain="domain"),
        yaxis=dict(
            title="True Positive Rate",
            range=[0, 1],
            constrain="domain",
            scaleanchor="x",
            scaleratio=1,
        ),
        hovermode="closest",
        plot_bgcolor="#fff",
    )
    fig = go.Figure(data, layout)
    shapes = [
        dict(
            type="line",
            xref="x",
            yref="y",
            x0=0,
            x1=1,
            y0=0,
            y1=1,
            line=dict(color="darkslategray", width=4, dash="dot"),
        )
    ]

    if cutoff is not None:
        threshold_idx = np.argmin(np.abs(thresholds - cutoff))
        cutoff_tpr = tpr[threshold_idx]
        cutoff_fpr = fpr[threshold_idx]

        shapes.append(
            dict(
                type="line",
                xref="x",
                yref="y",
                x0=0,
                x1=1,
                y0=cutoff_tpr,
                y1=cutoff_tpr,
                line=dict(color="lightslategray", width=1),
            )
        )
        shapes.append(
            dict(
                type="line",
                xref="x",
                yref="y",
                x0=cutoff_fpr,
                x1=cutoff_fpr,
                y0=0,
                y1=1,
                line=dict(color="lightslategray", width=1),
            )
        )

        annotations = [
            go.layout.Annotation(
                x=0.6,
                y=0.4,
                text=f"roc-auc-score: {score:.{round}f}",
                showarrow=False,
                align="right",
                xanchor="left",
                yanchor="top",
            ),
            go.layout.Annotation(
                x=0.6,
                y=0.35,
                text=f"cutoff: {cutoff:.{round}f}",
                showarrow=False,
                align="right",
                xanchor="left",
                yanchor="top",
            ),
            go.layout.Annotation(
                x=0.6,
                y=0.3,
                text=f"TPR: {cutoff_tpr:.{round}f}",
                showarrow=False,
                align="right",
                xanchor="left",
                yanchor="top",
            ),
            go.layout.Annotation(
                x=0.6,
                y=0.24,
                text=f"FPR: {cutoff_fpr:.{round}f}",
                showarrow=False,
                align="right",
                xanchor="left",
                yanchor="top",
            ),
        ]
        fig.update_layout(annotations=annotations)

    fig.update_layout(shapes=shapes)
    fig.update_layout(margin=dict(t=40, b=40, l=40, r=40))
    return fig


def plotly_pr_auc_curve(precision, recall, thresholds, score, cutoff=None, round=2):
    """Generate Precision-Recall Area Under Curve plot

    Args:
        true_y (np.ndarray): array of tru labels
        pred_probas (np.ndarray): array of predicted probabilities
        cutoff (float, optional): model cutoff to display in graph. Defaults to None.
        round (int, optional): rounding to apply to floats. Defaults to 2.

    Returns:
        Plotly fig:
    """

    trace0 = go.Scatter(
        x=precision,
        y=recall,
        mode="lines",
        name="PR AUC CURVE",
        text=[
            f"threshold: {th:.{round}f} <br>"
            + f"precision: {p:.{round}f} <br>"
            + f"recall: {r:.{round}f}"
            for p, r, th in zip(precision, recall, thresholds)
        ],
        hoverinfo="text",
    )
    data = [trace0]
    layout = go.Layout(
        title="PR AUC CURVE",
        #    width=450,
        #    height=450,
        xaxis=dict(title="Precision", range=[0, 1], constrain="domain"),
        yaxis=dict(
            title="Recall",
            range=[0, 1],
            constrain="domain",
            scaleanchor="x",
            scaleratio=1,
        ),
        hovermode="closest",
        plot_bgcolor="#fff",
    )
    fig = go.Figure(data, layout)
    shapes = []

    if cutoff is not None:
        threshold_idx = np.argmin(np.abs(thresholds - cutoff))
        cutoff_recall = recall[threshold_idx]
        cutoff_precision = precision[threshold_idx]
        shapes.append(
            dict(
                type="line",
                xref="x",
                yref="y",
                x0=0,
                x1=1,
                y0=cutoff_recall,
                y1=cutoff_recall,
                line=dict(color="lightslategray", width=1),
            )
        )
        shapes.append(
            dict(
                type="line",
                xref="x",
                yref="y",
                x0=cutoff_precision,
                x1=cutoff_precision,
                y0=0,
                y1=1,
                line=dict(color="lightslategray", width=1),
            )
        )

        annotations = [
            go.layout.Annotation(
                x=0.15,
                y=0.40,
                text=f"pr-auc-score: {score:.{round}f}",
                showarrow=False,
                align="right",
                xanchor="left",
                yanchor="top",
            ),
            go.layout.Annotation(
                x=0.15,
                y=0.35,
                text=f"cutoff: {cutoff:.{round}f}",
                showarrow=False,
                align="right",
                xanchor="left",
                yanchor="top",
            ),
            go.layout.Annotation(
                x=0.15,
                y=0.3,
                text=f"precision: {cutoff_precision:.{round}f}",
                showarrow=False,
                align="right",
                xanchor="left",
                yanchor="top",
            ),
            go.layout.Annotation(
                x=0.15,
                y=0.25,
                text=f"recall: {cutoff_recall:.{round}f}",
                showarrow=False,
                align="right",
                xanchor="left",
                yanchor="top",
            ),
        ]
        fig.update_layout(annotations=annotations)

    fig.update_layout(shapes=shapes)
    fig.update_layout(margin=dict(t=40, b=40, l=40, r=40))
    return fig


def plotly_shap_scatter_plot(
    X,
    shap_values_df,
    display_columns=None,
    title="Shap values",
    idxs=None,
    highlight_index=None,
    na_fill=-999,
    round=3,
    max_cat_colors=5,
):
    """Generate a shap values summary plot where features are ranked from
    highest mean absolute shap value to lowest, with point clouds shown
    for each feature.

    Args:

        X (pd.DataFrame): dataframe of input features
        shap_values_df (pd.DataFrame): dataframe shap_values with same columns as X
        display_columns (List[str]): list of feature to be displayed. If None
            default to all columns in X.
        title (str, optional): Title to display above graph.
            Defaults to "Shap values".
        idxs (List[str], optional): List of identifiers for each row in X.
            Defaults to None.
        highlight_index ({str, int}, optional): Index to highlight in graph.
            Defaults to None.
        na_fill (int, optional): Fill value used to fill missing values,
            will be colored grey in the graph.. Defaults to -999.
        round (int, optional): rounding to apply to floats. Defaults to 3.
        index_name (str): identifier for idxs. Defaults to "index".

    Returns:
        Plotly fig
    """
    assert matching_cols(
        X, shap_values_df
    ), "X and shap_values_df should have matching columns!"
    if display_columns is None:
        display_columns = X.columns.tolist()
    if idxs is not None:
        assert len(idxs) == X.shape[0]
        idxs = pd.Index(idxs).astype(str)
    else:
        idxs = X.index.astype(str)
    index_name = idxs.name

    length = len(X)
    if highlight_index is not None:
        if isinstance(highlight_index, int):
            assert highlight_index >= 0 and highlight_index < len(
                X
            ), "if highlight_index is int, then should be between 0 and {len(X)}!"
            highlight_idx = highlight_index
            highlight_index = idxs[highlight_idx]
        elif isinstance(highlight_index, str):
            assert str(highlight_index) in idxs, f"{highlight_index} not found in idxs!"
            highlight_idx = np.where(idxs == str(highlight_index))[0].item()
        else:
            raise ValueError("Please pass either int or str highlight_index!")

    # make sure that columns are actually in X:
    display_columns = [col for col in display_columns if col in X.columns]
    min_shap = shap_values_df.min().min()
    max_shap = shap_values_df.max().max()
    shap_range = max_shap - min_shap
    min_shap = min_shap - 0.01 * shap_range
    max_shap = max_shap + 0.01 * shap_range

    fig = make_subplots(
        rows=len(display_columns),
        cols=1,
        subplot_titles=display_columns,
        shared_xaxes=True,
    )

    for i, col in enumerate(display_columns):
        if is_numeric_dtype(X[col]):
            # numerical feature get a single bluered plot
            fig.add_trace(
                go.Scattergl(
                    x=shap_values_df[col],
                    y=np.random.rand(length),
                    mode="markers",
                    marker=dict(
                        size=5,
                        color=X[col].replace({na_fill: np.nan}),
                        colorscale="Bluered",
                        showscale=True,
                        opacity=0.3,
                        colorbar=dict(
                            title="feature value <br> (red is high)",
                            tickfont=dict(color="rgba(0, 0, 0, 0)"),
                        ),
                    ),
                    name=col,
                    showlegend=False,
                    opacity=0.8,
                    hoverinfo="text",
                    text=[
                        f"{index_name}={i}<br>{col}={value}<br>shap={shap:.{round}f}"
                        for i, shap, value in zip(
                            idxs, shap_values_df[col], X[col].replace({na_fill: np.nan})
                        )
                    ],
                ),
                row=i + 1,
                col=1,
            )
        else:
            color_cats = list(X[col].value_counts().index[:max_cat_colors])
            n_color_cats = len(color_cats)
            colors = [
                "#636EFA",
                "#EF553B",
                "#00CC96",
                "#AB63FA",
                "#FFA15A",
                "#19D3F3",
                "#FF6692",
                "#B6E880",
                "#FF97FF",
                "#FECB52",
            ]
            colors = colors * (1 + int(n_color_cats / len(colors)))
            colors = colors[:n_color_cats]
            for cat, color in zip(color_cats, colors):
                fig.add_trace(
                    go.Scattergl(
                        x=shap_values_df[col][X[col] == cat],
                        y=np.random.rand((X[col] == cat).sum()),
                        mode="markers",
                        marker=dict(
                            size=5,
                            showscale=False,
                            opacity=0.3,
                            color=color,
                        ),
                        name=cat,
                        showlegend=False,
                        opacity=0.8,
                        hoverinfo="text",
                        text=[
                            f"{index_name}={i}<br>{col}={cat}<br>shap={shap:.{round}f}"
                            for i, shap in zip(
                                idxs[X[col] == cat], shap_values_df[col][X[col] == cat]
                            )
                        ],
                    ),
                    row=i + 1,
                    col=1,
                )
            if X[col].nunique() > max_cat_colors:
                fig.add_trace(
                    go.Scattergl(
                        x=shap_values_df[col][~X[col].isin(color_cats)],
                        y=np.random.rand((~X[col].isin(color_cats)).sum()),
                        mode="markers",
                        marker=dict(
                            size=5,
                            showscale=False,
                            opacity=0.3,
                            color="grey",
                        ),
                        name="Other",
                        showlegend=False,
                        opacity=0.8,
                        hoverinfo="text",
                        text=[
                            f"{index_name}={i}<br>{col}={col_val}<br>shap={shap:.{round}f}"
                            for i, shap, col_val in zip(
                                idxs[~X[col].isin(color_cats)],
                                shap_values_df[col][~X[col].isin(color_cats)],
                                X[col][~X[col].isin(color_cats)],
                            )
                        ],
                    ),
                    row=i + 1,
                    col=1,
                )

        if highlight_index is not None:
            fig.add_trace(
                go.Scattergl(
                    x=[shap_values_df[col].iloc[highlight_idx]],
                    y=[0],
                    mode="markers",
                    marker=dict(
                        color="LightSkyBlue",
                        size=20,
                        opacity=0.5,
                        line=dict(color="MediumPurple", width=4),
                    ),
                    name=f"{index_name} {highlight_index}",
                    text=f"index={highlight_index}<br>{col}={X[col].iloc[highlight_idx]}<br>shap={shap_values_df[col].iloc[highlight_idx]:.{round}f}",
                    hoverinfo="text",
                    showlegend=False,
                ),
                row=i + 1,
                col=1,
            )
        fig.update_xaxes(
            showgrid=False, zeroline=False, range=[min_shap, max_shap], row=i + 1, col=1
        )
        fig.update_yaxes(
            showgrid=False, zeroline=False, showticklabels=False, row=i + 1, col=1
        )

    fig.update_layout(
        title=title + "<br>",
        height=100 + len(display_columns) * 50,
        margin=go.layout.Margin(l=40, r=40, b=40, t=100, pad=4),
        hovermode="closest",
        plot_bgcolor="#fff",
    )
    return fig


def plotly_predicted_vs_actual(
    y,
    preds,
    target="",
    units="",
    round=2,
    logs=False,
    log_x=False,
    log_y=False,
    idxs=None,
    index_name="index",
):
    """Generate graph showing predicted values from a regressor model vs actual
    values.

    Args:
        y (np.ndarray): Actual values
        preds (np.ndarray): Predicted values
        target (str, optional): Label for target. Defaults to "".
        units (str, optional): Units of target. Defaults to "".
        round (int, optional): Rounding to apply to floats. Defaults to 2.
        logs (bool, optional): Log both axis. Defaults to False.
        log_x (bool, optional): Log x axis. Defaults to False.
        log_y (bool, optional): Log y axis. Defaults to False.
        idxs (List[str], optional): list of identifiers for each observation. Defaults to None.
        index_name (str): identifier for idxs. Defaults to "index".

    Returns:
        Plotly fig
    """

    if idxs is not None:
        assert len(idxs) == len(preds)
        idxs = [str(idx) for idx in idxs]
    else:
        idxs = [str(i) for i in range(len(preds))]

    marker_text = [
        f"{index_name}: {idx}<br>Observed: {actual:.{round}f}<br>Prediction: {pred:.{round}f}"
        for idx, actual, pred in zip(idxs, y, preds)
    ]

    trace0 = go.Scattergl(
        x=y,
        y=preds,
        mode="markers",
        name=f"predicted {target}" + f" ({units})" if units else "",
        text=marker_text,
        hoverinfo="text",
    )

    sorted_y = np.sort(y)
    trace1 = go.Scattergl(
        x=sorted_y,
        y=sorted_y,
        mode="lines",
        name=f"observed {target}" + f" ({units})" if units else "",
        hoverinfo="none",
    )

    data = [trace0, trace1]

    layout = go.Layout(
        title=f"Predicted {target} vs Observed {target}",
        yaxis=dict(title=f"Predicted {target}" + (f" ({units})" if units else "")),
        xaxis=dict(title=f"Observed {target}" + (f" ({units})" if units else "")),
        plot_bgcolor="#fff",
        hovermode="closest",
    )

    fig = go.Figure(data, layout)
    if logs:
        fig.update_layout(xaxis_type="log", yaxis_type="log")
    if log_x:
        fig.update_layout(xaxis_type="log")
    if log_y:
        fig.update_layout(yaxis_type="log")
    fig.update_layout(margin=dict(t=40, b=40, l=40, r=40))
    return fig


def plotly_plot_residuals(
    y,
    preds,
    vs_actual=False,
    target="",
    units="",
    residuals="difference",
    round=2,
    idxs=None,
    index_name="index",
):
    """generates a residual plot

    Args:
        y (np.array, pd.Series): Actual values
        preds (np.array, pd.Series): Predictions
        vs_actual (bool, optional): Put actual values (y) on the x-axis.
                    Defaults to False (i.e. preds on the x-axis)
        target (str, optional): name of the target variable. Defaults to ""
        units (str, optional): units of the axis. Defaults to "".
        residuals (str, {'difference', 'ratio', 'log-ratio'} optional):
                    How to calcualte residuals. Defaults to 'difference'.
        round (int, optional): [description]. Defaults to 2.
        idxs ([type], optional): [description]. Defaults to None.
        index_name (str): identifier for idxs. Defaults to "index".

    Returns:
        [type]: [description]
    """
    if idxs is not None:
        assert len(idxs) == len(preds)
        idxs = [str(idx) for idx in idxs]
    else:
        idxs = [str(i) for i in range(len(preds))]

    res = y - preds
    res_ratio = y / preds

    if residuals == "log-ratio":
        residuals_display = np.log(res_ratio)
        residuals_name = "residuals log ratio<br>(log(y/preds))"
    elif residuals == "ratio":
        residuals_display = res_ratio
        residuals_name = "residuals ratio<br>(y/preds)"
    elif residuals == "difference":
        residuals_display = res
        residuals_name = "residuals (y-preds)"
    else:
        raise ValueError(
            f"parameter residuals should be in ['difference', "
            f"'ratio', 'log-ratio'] but is equal to {residuals}!"
        )

    residuals_text = [
        f"{index_name}: {idx}<br>Observed: {actual:.{round}f}<br>Prediction: {pred:.{round}f}<br>Residual: {residual:.{round}f}"
        for idx, actual, pred, residual in zip(idxs, y, preds, res)
    ]
    trace0 = go.Scattergl(
        x=y if vs_actual else preds,
        y=residuals_display,
        mode="markers",
        name=residuals_name,
        text=residuals_text,
        hoverinfo="text",
    )

    trace1 = go.Scattergl(
        x=y if vs_actual else preds,
        y=np.ones(len(preds)) if residuals == "ratio" else np.zeros(len(preds)),
        mode="lines",
        name=(f"Observed {target}" + f" ({units})" if units else "")
        if vs_actual
        else (f"Predicted {target}" + f" ({units})" if units else ""),
        hoverinfo="none",
    )

    data = [trace0, trace1]

    layout = go.Layout(
        title=f"Residuals vs {'observed' if vs_actual else 'predicted'} {target}",
        yaxis=dict(title=residuals_name),
        xaxis=dict(
            title=(f"Observed {target}" + f" ({units})" if units else "")
            if vs_actual
            else (f"Predicted {target}" + f" ({units})" if units else "")
        ),
        plot_bgcolor="#fff",
        hovermode="closest",
    )

    fig = go.Figure(data, layout)
    fig.update_layout(margin=dict(t=40, b=40, l=40, r=40))
    return fig


def plotly_residuals_vs_col(
    y,
    preds,
    col,
    col_name=None,
    residuals="difference",
    idxs=None,
    round=2,
    points=True,
    winsor=0,
    na_fill=-999,
    index_name="index",
    cats_order=None,
):
    """Generates a residuals plot vs a particular feature column.

    Args:
        y (np.ndarray): array of actual target values
        preds (np.ndarray): array of predicted values
        col (pd.Series): series of values to be used as x-axis
        col_name (str, optional): feature name to display.
            Defaults to None, in which case col.name gets used.
        residuals ({'log-ratio', 'ratio', 'difference'}, optional):
            type of residuals to display. Defaults to 'difference'.
        idxs (List[str], optional): str identifiers for each sample.
            Defaults to None.
        round (int, optional): Rounding to apply to floats. Defaults to 2.
        points (bool, optional): For categorical features display point cloud
            next to violin plots. Defaults to True.
        winsor (int, optional): Winsorize the outliers. Remove the top `winsor`
            percent highest and lowest values. Defaults to 0.
        na_fill (int, optional): Value used to fill missing values. Defaults to -999.
        index_name (str): identifier for idxs. Defaults to "index".
        cats_order (list, optional): list of categories to display. If None
            defaults to X_col.unique().tolist() so displays all categories.


    Returns:
        Plotly fig
    """
    if col_name is None:
        try:
            col_name = col.name
        except:
            col_name = "Feature"

    if idxs is not None:
        assert len(idxs) == len(preds)
        idxs = [str(idx) for idx in idxs]
    else:
        idxs = [str(i) for i in range(len(preds))]

    res = y - preds
    res_ratio = y / preds

    if residuals == "log-ratio":
        residuals_display = np.log(res_ratio)
        residuals_name = "residuals log ratio<br>(log(y/preds))"
    elif residuals == "ratio":
        residuals_display = res_ratio
        residuals_name = "residuals ratio<br>(y/preds)"
    elif residuals == "difference":
        residuals_display = res
        residuals_name = "residuals (y-preds)"
    else:
        raise ValueError(
            f"parameter residuals should be in ['difference', "
            f"'ratio', 'log-ratio'] but is equal to {residuals}!"
        )

    residuals_text = [
        f"{index_name}: {idx}<br>Actual: {actual:.{round}f}<br>Prediction: {pred:.{round}f}<br>Residual: {residual:.{round}f}"
        for idx, actual, pred, residual in zip(idxs, y, preds, res)
    ]

    if not is_numeric_dtype(col):
        if cats_order is None:
            cats_order = sorted(col.unique().tolist())
        n_cats = len(cats_order)

        if points:
            fig = make_subplots(
                rows=1,
                cols=2 * n_cats,
                column_widths=[3, 1] * n_cats,
                shared_yaxes=True,
            )
            showscale = True
        else:
            fig = make_subplots(rows=1, cols=n_cats, shared_yaxes=True)

        fig.update_yaxes(
            range=[
                np.percentile(residuals_display, winsor),
                np.percentile(residuals_display, 100 - winsor),
            ]
        )

        for i, cat in enumerate(cats_order):
            column = 1 + i * 2 if points else 1 + i
            fig.add_trace(
                go.Violin(
                    x=col[col == cat],
                    y=residuals_display[col == cat],
                    name=cat,
                    box_visible=True,
                    meanline_visible=True,
                    showlegend=False,
                ),
                row=1,
                col=column,
            )
            if points:
                fig.add_trace(
                    go.Scattergl(
                        x=np.random.randn(len(col[col == cat])),
                        y=residuals_display[col == cat],
                        mode="markers",
                        showlegend=False,
                        text=[t for t, b in zip(residuals_text, col == cat) if b],
                        hoverinfo="text",
                        marker=dict(size=7, opacity=0.3, color="blue"),
                    ),
                    row=1,
                    col=column + 1,
                )

        if points:
            for i in range(n_cats):
                fig.update_xaxes(
                    showgrid=False, zeroline=False, visible=False, row=1, col=2 + i * 2
                )
                fig.update_yaxes(showgrid=False, zeroline=False, row=1, col=2 + i * 2)

        fig.update_layout(title=f"Residuals vs {col_name}", hovermode="closest")
        fig.update_layout(margin=dict(t=40, b=40, l=40, r=40))
        return fig

    else:
        col[col == na_fill] = np.nan

        trace0 = go.Scattergl(
            x=col,
            y=residuals_display,
            mode="markers",
            name=residuals_name,
            text=residuals_text,
            hoverinfo="text",
        )

        trace1 = go.Scattergl(
            x=col,
            y=np.ones(len(preds)) if residuals == "ratio" else np.zeros(len(preds)),
            mode="lines",
            name=col_name,
            hoverinfo="none",
        )

        data = [trace0, trace1]

        layout = go.Layout(
            title=f"Residuals vs {col_name}",
            yaxis=dict(title=residuals_name),
            xaxis=dict(title=f"{col_name} value"),
            plot_bgcolor="#fff",
            hovermode="closest",
        )

        fig = go.Figure(data, layout)
        fig.update_yaxes(
            range=[
                np.percentile(residuals_display, winsor),
                np.percentile(residuals_display, 100 - winsor),
            ]
        )
        fig.update_layout(margin=dict(t=40, b=40, l=40, r=40))
        return fig


def plotly_actual_vs_col(
    y,
    preds,
    col,
    col_name=None,
    idxs=None,
    round=2,
    points=True,
    winsor=0,
    na_fill=-999,
    units="",
    target="",
    index_name="index",
    cats_order=None,
):
    """Generates a residuals plot vs a particular feature column.

    Args:
        y (np.ndarray): array of actual target values
        preds (np.ndarray): array of predicted values
        col (pd.Series): series of values to be used as x-axis
        col_name (str, optional): feature name to display.
            Defaults to None, in which case col.name gets used.
        idxs (List[str], optional): str identifiers for each sample.
            Defaults to None.
        round (int, optional): Rounding to apply to floats. Defaults to 2.
        points (bool, optional): For categorical features display point cloud
            next to violin plots. Defaults to True.
        winsor (int, optional): Winsorize the outliers. Remove the top `winsor`
            percent highest and lowest values. Defaults to 0.
        na_fill (int, optional): Value used to fill missing values. Defaults to -999.
        index_name (str): identifier for idxs. Defaults to "index".
        cats_order (list, optional): list of categories to display. If None
            defaults to X_col.unique().tolist() so displays all categories.


    Returns:
        Plotly fig
    """
    if col_name is None:
        try:
            col_name = col.name
        except:
            col_name = "Feature"

    if idxs is not None:
        assert len(idxs) == len(preds)
        idxs = [str(idx) for idx in idxs]
    else:
        idxs = [str(i) for i in range(len(preds))]

    y_text = [
        f"{index_name}: {idx}<br>Observed {target}: {actual:.{round}f}<br>Prediction: {pred:.{round}f}"
        for idx, actual, pred in zip(idxs, y, preds)
    ]

    if not is_numeric_dtype(col):
        if cats_order is None:
            cats_order = sorted(col.unique().tolist())
        n_cats = len(cats_order)

        if points:
            fig = make_subplots(
                rows=1,
                cols=2 * n_cats,
                column_widths=[3, 1] * n_cats,
                shared_yaxes=True,
            )
            showscale = True
        else:
            fig = make_subplots(rows=1, cols=n_cats, shared_yaxes=True)

        fig.update_yaxes(
            range=[np.percentile(y, winsor), np.percentile(y, 100 - winsor)]
        )

        for i, cat in enumerate(cats_order):
            column = 1 + i * 2 if points else 1 + i
            fig.add_trace(
                go.Violin(
                    x=col[col == cat],
                    y=y[col == cat],
                    name=cat,
                    box_visible=True,
                    meanline_visible=True,
                    showlegend=False,
                ),
                row=1,
                col=column,
            )
            if points:
                fig.add_trace(
                    go.Scattergl(
                        x=np.random.randn(len(col[col == cat])),
                        y=y[col == cat],
                        mode="markers",
                        showlegend=False,
                        text=[t for t, b in zip(y_text, col == cat) if b],
                        hoverinfo="text",
                        marker=dict(size=7, opacity=0.6, color="blue"),
                    ),
                    row=1,
                    col=column + 1,
                )

        if points:
            for i in range(n_cats):
                fig.update_xaxes(
                    showgrid=False, zeroline=False, visible=False, row=1, col=2 + i * 2
                )
                fig.update_yaxes(showgrid=False, zeroline=False, row=1, col=2 + i * 2)

        fig.update_layout(
            title=f"Observed {target} vs {col_name}",
            yaxis=dict(
                title=f"Observed {target} ({units})" if units else f"Observed {target}"
            ),
            hovermode="closest",
        )
        fig.update_layout(margin=dict(t=40, b=40, l=40, r=40))
        return fig

    else:
        col[col == na_fill] = np.nan

        trace0 = go.Scattergl(
            x=col,
            y=y,
            mode="markers",
            name="Observed",
            text=y_text,
            hoverinfo="text",
        )

        data = [trace0]

        layout = go.Layout(
            title=f"Observed {target} vs {col_name}",
            yaxis=dict(
                title=f"Observed {target} ({units})" if units else f"Observed {target}"
            ),
            xaxis=dict(title=f"{col_name} value"),
            plot_bgcolor="#fff",
            hovermode="closest",
        )

        fig = go.Figure(data, layout)
        fig.update_yaxes(
            range=[np.percentile(y, winsor), np.percentile(y, 100 - winsor)]
        )
        fig.update_layout(margin=dict(t=40, b=40, l=40, r=40))
        return fig


def plotly_preds_vs_col(
    y,
    preds,
    col,
    col_name=None,
    idxs=None,
    round=2,
    points=True,
    winsor=0,
    na_fill=-999,
    units="",
    target="",
    index_name="index",
    cats_order=None,
):
    """Generates plot of predictions vs a particular feature column.

    Args:
        y (np.ndarray): array of actual target values
        preds (np.ndarray): array of predicted values
        col (pd.Series): series of values to be used as x-axis
        col_name (str, optional): feature name to display.
            Defaults to None, in which case col.name gets used.
        idxs (List[str], optional): str identifiers for each sample.
            Defaults to None.
        round (int, optional): Rounding to apply to floats. Defaults to 2.
        points (bool, optional): For categorical features display point cloud
            next to violin plots. Defaults to True.
        winsor (int, optional): Winsorize the outliers. Remove the top `winsor`
            percent highest and lowest values. Defaults to 0.
        na_fill (int, optional): Value used to fill missing values. Defaults to -999.
        index_name (str): identifier for idxs. Defaults to "index".
        cats_order (list, optional): list of categories to display. If None
            defaults to X_col.unique().tolist() so displays all categories.

    Returns:
        Plotly fig
    """
    if col_name is None:
        try:
            col_name = col.name
        except:
            col_name = "Feature"

    if idxs is not None:
        assert len(idxs) == len(preds)
        idxs = [str(idx) for idx in idxs]
    else:
        idxs = [str(i) for i in range(len(preds))]

    preds_text = [
        f"{index_name}: {idx}<br>Predicted {target}: {pred:.{round}f}{units}<br>Observed {target}: {actual:.{round}f}{units}"
        for idx, actual, pred in zip(idxs, y, preds)
    ]

    if not is_numeric_dtype(col):
        if cats_order is None:
            cats_order = sorted(col.unique().tolist())
        n_cats = len(cats_order)

        if points:
            fig = make_subplots(
                rows=1,
                cols=2 * n_cats,
                column_widths=[3, 1] * n_cats,
                shared_yaxes=True,
            )
            showscale = True
        else:
            fig = make_subplots(rows=1, cols=n_cats, shared_yaxes=True)

        fig.update_yaxes(
            range=[np.percentile(preds, winsor), np.percentile(preds, 100 - winsor)]
        )

        for i, cat in enumerate(cats_order):
            column = 1 + i * 2 if points else 1 + i
            fig.add_trace(
                go.Violin(
                    x=col[col == cat],
                    y=preds[col == cat],
                    name=cat,
                    box_visible=True,
                    meanline_visible=True,
                    showlegend=False,
                ),
                row=1,
                col=column,
            )
            if points:
                fig.add_trace(
                    go.Scattergl(
                        x=np.random.randn(len(col[col == cat])),
                        y=preds[col == cat],
                        mode="markers",
                        showlegend=False,
                        text=[t for t, b in zip(preds_text, col == cat) if b],
                        hoverinfo="text",
                        marker=dict(size=7, opacity=0.6, color="blue"),
                    ),
                    row=1,
                    col=column + 1,
                )

        if points:
            for i in range(n_cats):
                fig.update_xaxes(
                    showgrid=False, zeroline=False, visible=False, row=1, col=2 + i * 2
                )
                fig.update_yaxes(showgrid=False, zeroline=False, row=1, col=2 + i * 2)

        fig.update_layout(
            title=f"Predicted {target} vs {col_name}",
            yaxis=dict(
                title=f"Predicted {target} ({units})"
                if units
                else f"Predicted {target}"
            ),
            hovermode="closest",
        )
        fig.update_layout(margin=dict(t=40, b=40, l=40, r=40))
        return fig

    else:
        col[col == na_fill] = np.nan

        trace0 = go.Scattergl(
            x=col,
            y=preds,
            mode="markers",
            name="Predicted",
            text=preds_text,
            hoverinfo="text",
        )

        data = [trace0]

        layout = go.Layout(
            title=f"Predicted {target} vs {col_name}",
            yaxis=dict(
                title=f"Predicted {target} ({units})"
                if units
                else f"Predicted {target}"
            ),
            xaxis=dict(title=f"{col_name} value"),
            plot_bgcolor="#fff",
            hovermode="closest",
        )

        fig = go.Figure(data, layout)
        fig.update_yaxes(
            range=[np.percentile(preds, winsor), np.percentile(preds, 100 - winsor)]
        )
        fig.update_layout(margin=dict(t=40, b=40, l=40, r=40))
        return fig


def plotly_rf_trees(
    model,
    observation,
    y=None,
    highlight_tree=None,
    round=2,
    pos_label=1,
    target="",
    units="",
):
    """Generate a plot showing the prediction of every single tree inside a RandomForest model

    Args:
        model ({RandomForestClassifier, RandomForestRegressor}): model to display trees for
        observation (pd.DataFrame): row of input data, e.g. X.iloc[[0]]
        y (np.ndarray, optional): Target values. Defaults to None.
        highlight_tree (int, optional): DecisionTree to highlight in graph. Defaults to None.
        round (int, optional): Apply rounding to floats. Defaults to 2.
        pos_label (int, optional): For RandomForestClassifier: Class label
            to generate graph for. Defaults to 1.
        target (str, optional): Description of target variable. Defaults to "".
        units (str, optional): Units of target variable. Defaults to "".

    Returns:
        Plotly fig
    """
    assert safe_isinstance(
        model,
        "RandomForestClassifier",
        "RandomForestRegressor",
        "ExtraTreesClassifier",
        "ExtraTreesRegressor",
    ), (
        f"model is of type {type(model)}, but plot_rf_trees() only accepts RandomForestClassifier, "
        "RandomForestRegressor, ExtraTreesClassifier or ExtraTreesRegressor!"
    )

    colors = ["blue"] * len(model.estimators_)
    if highlight_tree is not None:
        assert highlight_tree >= 0 and highlight_tree <= len(
            model.estimators_
        ), f"{highlight_tree} is out of range (0, {len(model.estimators_)})"
        colors[highlight_tree] = "red"
    warnings.filterwarnings("ignore", category=UserWarning)
    if safe_isinstance(model, "RandomForestClassifier", "ExtraTreesClassifier"):
        preds_df = (
            pd.DataFrame(
                {
                    "model": range(len(model.estimators_)),
                    "prediction": [
                        np.round(
                            100 * m.predict_proba(observation)[0, pos_label], round
                        )
                        for m in model.estimators_
                    ],
                    "color": colors,
                }
            )
            .sort_values("prediction")
            .reset_index(drop=True)
        )
    else:
        preds_df = (
            pd.DataFrame(
                {
                    "model": range(len(model.estimators_)),
                    "prediction": [
                        np.round(m.predict(observation)[0], round)
                        for m in model.estimators_
                    ],
                    "color": colors,
                }
            )
            .sort_values("prediction")
            .reset_index(drop=True)
        )
    warnings.filterwarnings("default", category=UserWarning)

    trace0 = go.Bar(
        x=preds_df.index,
        y=preds_df.prediction,
        marker_color=preds_df.color,
        text=[
            f"tree no {t}:<br> prediction={p}<br> click for detailed info"
            for (t, p) in zip(preds_df.model.values, preds_df.prediction.values)
        ],
        hoverinfo="text",
    )

    if target:
        title = f"Individual decision trees predicting {target}"
        yaxis_title = f"Predicted {target} {f'({units})' if units else ''}"
    else:
        title = f"Individual decision trees"
        yaxis_title = f"Predicted outcome ({units})" if units else "Predicted outcome"

    layout = go.Layout(
        title=title,
        plot_bgcolor="#fff",
        yaxis=dict(title=yaxis_title),
        xaxis=dict(title="decision trees (sorted by prediction"),
    )
    fig = go.Figure(data=[trace0], layout=layout)
    shapes = [
        dict(
            type="line",
            xref="x",
            yref="y",
            x0=0,
            x1=preds_df.model.max(),
            y0=preds_df.prediction.mean(),
            y1=preds_df.prediction.mean(),
            line=dict(color="lightgray", width=4, dash="dot"),
        )
    ]

    annotations = [
        go.layout.Annotation(
            x=1.2 * preds_df.model.mean(),
            y=preds_df.prediction.mean(),
            text=f"Average prediction = {preds_df.prediction.mean():.{round}f}",
            bgcolor="lightgrey",
            arrowcolor="lightgrey",
            startstandoff=0,
        )
    ]

    if y is not None:
        shapes.append(
            dict(
                type="line",
                xref="x",
                yref="y",
                x0=0,
                x1=preds_df.model.max(),
                y0=y,
                y1=y,
                line=dict(color="red", width=4, dash="dashdot"),
            )
        )
        annotations.append(
            go.layout.Annotation(
                x=0.8 * preds_df.model.mean(),
                y=y,
                text=f"observed={y}",
                bgcolor="red",
                arrowcolor="red",
            )
        )

    fig.update_layout(shapes=shapes)
    fig.update_layout(annotations=annotations)
    fig.update_layout(margin=dict(t=40, b=40, l=40, r=40))
    return fig


def plotly_xgboost_trees(
    xgboost_preds_df,
    highlight_tree=None,
    y=None,
    round=2,
    pos_label=1,
    target="",
    units="",
    higher_is_better=True,
):
    """Generate a plot showing the prediction of every single tree inside an XGBoost model

    Args:
        xgboost_preds_df (pd.DataFrame): generated with get_xgboost_preds_df(...)
        highlight_tree (int, optional): DecisionTree to highlight in graph. Defaults to None.
        y (np.ndarray, optional): Target values. Defaults to None.
        round (int, optional): Apply rounding to floats. Defaults to 2.
        pos_label (int, optional): For RandomForestClassifier: Class label
            to generate graph for. Defaults to 1.
        target (str, optional): Description of target variable. Defaults to "".
        units (str, optional): Units of target variable. Defaults to "".
        higher_is_better (bool, optional): up is green, down is red. If False then
            flip the colors.

    Returns:
        Plotly fig
    """

    xgboost_preds_df["color"] = "blue"
    xgboost_preds_df.loc[0, "color"] = "yellow"
    if highlight_tree is not None:
        xgboost_preds_df.loc[highlight_tree + 1, "color"] = "red"

    trees = xgboost_preds_df.tree.values[1:]
    colors = xgboost_preds_df.color.values[1:]

    is_classifier = True if "pred_proba" in xgboost_preds_df.columns else False

    colors = xgboost_preds_df.color.values
    if is_classifier:
        final_prediction = xgboost_preds_df.pred_proba.values[-1]
        base_prediction = xgboost_preds_df.pred_proba.values[0]
        preds = xgboost_preds_df.pred_proba.values[1:]
        bases = xgboost_preds_df.pred_proba.values[:-1]
        diffs = xgboost_preds_df.pred_proba_diff.values[1:]

        texts = [
            f"tree no {t}:<br>change = {100*d:.{round}f}%<br> click for detailed info"
            for (t, d) in zip(trees, diffs)
        ]
        texts.insert(
            0, f"Base prediction: <br>proba = {100*base_prediction:.{round}f}%"
        )
        texts.append(f"Final Prediction: <br>proba = {100*final_prediction:.{round}f}%")
    else:
        final_prediction = xgboost_preds_df.pred.values[-1]
        base_prediction = xgboost_preds_df.pred.values[0]
        preds = xgboost_preds_df.pred.values[1:]
        bases = xgboost_preds_df.pred.values[:-1]
        diffs = xgboost_preds_df.pred_diff.values[1:]

        texts = [
            f"tree no {t}:<br>change = {d:.{round}f}<br> click for detailed info"
            for (t, d) in zip(trees, diffs)
        ]
        texts.insert(0, f"Base prediction: <br>pred = {base_prediction:.{round}f}")
        texts.append(f"Final Prediction: <br>pred = {final_prediction:.{round}f}")

    green_fill, green_line = "rgba(50, 200, 50, 1.0)", "rgba(40, 160, 50, 1.0)"
    yellow_fill, yellow_line = "rgba(230, 230, 30, 1.0)", "rgba(190, 190, 30, 1.0)"
    blue_fill, blue_line = "rgba(55, 128, 191, 0.7)", "rgba(55, 128, 191, 1.0)"
    red_fill, red_line = "rgba(219, 64, 82, 0.7)", "rgba(219, 64, 82, 1.0)"

    if higher_is_better:
        fill_color_up, line_color_up = green_fill, green_line
        fill_color_down, line_color_down = red_fill, red_line
    else:
        fill_color_up, line_color_up = red_fill, red_line
        fill_color_down, line_color_down = green_fill, green_line

    fill_colors = [fill_color_up if diff > 0 else fill_color_down for diff in diffs]
    line_colors = [line_color_up if diff > 0 else line_color_down for diff in diffs]

    fill_colors.insert(0, yellow_fill)
    line_colors.insert(0, yellow_line)

    fill_colors.append(blue_fill)
    line_colors.append(blue_line)

    trees = np.append(trees, len(trees))
    trees = np.insert(trees, 0, -1)
    bases = np.insert(bases, 0, 0)
    bases = np.append(bases, 0)
    diffs = np.insert(diffs, 0, base_prediction)
    diffs = np.append(diffs, final_prediction)

    trace0 = go.Bar(
        x=trees,
        y=bases,
        hoverinfo="skip",
        name="",
        showlegend=False,
        marker=dict(color="rgba(1,1,1, 0.0)"),
    )

    trace1 = go.Bar(
        x=trees,
        y=diffs,
        text=texts,
        name="",
        hoverinfo="text",
        showlegend=False,
        marker=dict(
            color=fill_colors,
            line=dict(
                color=line_colors,
                width=2,
            ),
        ),
    )

    if target:
        title = f"Individual xgboost decision trees predicting {target}"
        yaxis_title = f"Predicted {target} {f'({units})' if units else ''}"
    else:
        title = f"Individual xgboost decision trees"
        yaxis_title = f"Predicted outcome ({units})" if units else "Predicted outcome"

    layout = go.Layout(
        title=title,
        barmode="stack",
        plot_bgcolor="#fff",
        yaxis=dict(title=yaxis_title),
        xaxis=dict(title="decision trees"),
    )

    fig = go.Figure(data=[trace0, trace1], layout=layout)

    shapes = []
    annotations = []

    if y is not None:
        shapes.append(
            dict(
                type="line",
                xref="x",
                yref="y",
                x0=trees.min(),
                x1=trees.max(),
                y0=y,
                y1=y,
                line=dict(color="black", width=4, dash="dashdot"),
            )
        )
        annotations.append(
            go.layout.Annotation(
                x=0.75 * trees.max(), y=y, text=f"Observed={y}", bgcolor="white"
            )
        )

    fig.update_layout(shapes=shapes)
    fig.update_layout(annotations=annotations)
    fig.update_layout(margin=dict(t=40, b=40, l=40, r=40))
    return fig
