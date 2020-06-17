import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_string_dtype

import plotly.graph_objs as go
from plotly.subplots import make_subplots

from sklearn.metrics import (classification_report, confusion_matrix,
                             precision_recall_curve, roc_curve, 
                             roc_auc_score, average_precision_score)


def plotly_contribution_plot(contrib_df, target="target", 
                         model_output="raw", higher_is_better=False,
                         include_base_value=True, round=2):
    """
    Takes in a DataFrame contrib_df with columns
    'col' -- name of columns
    'contribution' -- contribution of that column to the final predictions
    'cumulatief' -- the summed contributions from the top to the current row
    'base' -- the summed contributions from the top excluding the current row

    Outputs a bar chart displaying the contribution of each individual
    column to the final prediction. 

    :param model_out 'raw' or 'probability' or 'logodds'
    """ 
    contrib_df = contrib_df.copy()
    multiplier = 100 if model_output=='probability' else 1
    contrib_df['base'] = np.round(multiplier * contrib_df['base'], round)
    contrib_df['cumulative'] = np.round(multiplier * contrib_df['cumulative'], round)
    contrib_df['contribution'] = np.round(multiplier * contrib_df['contribution'], round)


    if not include_base_value:
        contrib_df = contrib_df[contrib_df.col != 'base_value']
        
    longest_feature_name = contrib_df['col'].str.len().max()

    # prediction is the sum of all contributions:
    prediction = contrib_df['cumulative'].values[-1]

    # Base of each bar
    trace0 = go.Bar(
        x=contrib_df['col'].values,
        y=contrib_df['base'].values,
        hoverinfo='skip',
        name="",
        marker=dict(
            color='rgba(1,1,1, 0.0)',
        )
    )
    if 'value' in contrib_df.columns:
        hover_text=[f"{col}={value}<BR>{'+' if contrib>0 else ''}{contrib}" 
                  for col, value, contrib in zip(
                      contrib_df.col, contrib_df.value, contrib_df.contribution)]
    else:
        hover_text=[f"{col}=?<BR>{'+' if contrib>0 else ''}{contrib}"  
                  for col, contrib in zip(contrib_df.col, contrib_df.contribution)]

    fill_colour_up='rgba(55, 128, 191, 0.7)' if higher_is_better else 'rgba(219, 64, 82, 0.7)'
    fill_colour_down='rgba(219, 64, 82, 0.7)' if higher_is_better else 'rgba(55, 128, 191, 0.7)'
    line_colour_up='rgba(55, 128, 191, 1.0)' if higher_is_better else 'rgba(219, 64, 82, 1.0)'
    line_colour_down='rgba(219, 64, 82, 1.0)' if higher_is_better else 'rgba(55, 128, 191, 1.0)'
    
    # top of each bar (base + contribution)
    trace1 = go.Bar(
        x=contrib_df['col'].values.tolist(),
        y=contrib_df['contribution'].values,
        text=hover_text,
        name="contribution",
        hoverinfo="text",
        marker=dict(
            # blue if positive contribution, red if negative
            color=[fill_colour_up if y > 0 else fill_colour_down 
                           for y in contrib_df['contribution'].values.tolist()],
            line=dict(
                color=[line_colour_up if y > 0 else line_colour_down
                           for y in contrib_df['contribution'].values.tolist()],
                width=2,
            )
        )
    )
    
    if model_output == "probability":
        title = f'Contribution to prediction probability = {prediction}%'
    elif model_output == "logodds":
        title = f'Contribution to prediction logodds = {prediction}'
    else:
        title = f'Contribution to prediction {target} = {prediction}'

    data = [trace0, trace1]
    layout = go.Layout(
        title=title,
        barmode='stack',
        plot_bgcolor = '#fff',
        showlegend=False
    )

    fig = go.Figure(data=data, layout=layout)
    #fig.update_xaxes(automargin=True)
    fig.update_layout(margin=go.layout.Margin(
                                l=50,
                                r=100,
                                b=longest_feature_name*6,
                                t=50,
                                pad=4
                            ))
    if model_output=="probability":
        fig.update_yaxes(title_text='Prediction %')
    else:
        fig.update_yaxes(title_text='Prediction')
    return fig



def plotly_precision_plot(precision_df, cutoff=None, labels=None, pos_label=None):
    """
    returns a plotly figure with bar plots for counts of observations for a 
    certain pred_proba bin,
    and a line trace for the actual fraction of positive labels for that bin.
    
    precision_df generated from get_precision_df(predictions_df)
    predictions_df generated from get_predictions_df(rf_model, X, y)
    """
    label = labels[pos_label] if labels is not None and pos_label is not None else 'positive'
    
    precision_df = precision_df.copy()

    spacing = 0.1 /  len(precision_df)
    bin_widths = precision_df['bin_width'] - spacing
    bin_widths[bin_widths<0.005] = 0.005

    trace1 = go.Bar(
        x=(0.5*(precision_df['p_min']+precision_df['p_max'])).values,
        y=precision_df['count'].values,
        width=bin_widths,
        name='counts'
    )
    
    data = [trace1]
    
    if 'precision_0' in precision_df.columns.tolist():
        # if a pred_proba with probability for every class gets passed
        # to get_precision_df, it generates a precision for every class
        # in every bin as well.
        precision_cols = [col for col in precision_df.columns.tolist() 
                                if col.startswith('precision_')]
        if labels is None: labels = ['class ' + str(i) for i in range(len(precision_cols))]
        if pos_label is not None:
            # add the positive class first with thick line
            trace = go.Scatter(
                x=precision_df['p_avg'].values.tolist(),
                y=precision_df['precision_'+str(pos_label)].values.tolist(),
                name=labels[pos_label] + '(positive class)',
                line = dict(width=4),
                yaxis='y2')
            data.append(trace)

        for i, precision_col in enumerate(precision_cols):
            # add the rest of the classes with thin lines
            if pos_label is None or i != pos_label:
                trace = go.Scatter(
                    x=precision_df['p_avg'].values.tolist(),
                    y=precision_df[precision_col].values.tolist(),
                    name=labels[i],
                    line = dict(width=2),
                    yaxis='y2')
                data.append(trace)
    else: 
        
        trace2 = go.Scatter(
            x=precision_df['p_avg'].values.tolist(),
            y=precision_df['precision'].values.tolist(),
            name='percentage ' + label,
            yaxis='y2'
        )
        data = [trace1, trace2]
        
    layout = go.Layout(
        title=f'percentage {label} vs predicted probability',
        yaxis=dict(
            title='counts'
        ),
        yaxis2=dict(
            title='percentage',
            titlefont=dict(
                color='rgb(148, 103, 189)'
            ),
            tickfont=dict(
                color='rgb(148, 103, 189)'
            ),
            overlaying='y',
            side='right',
            rangemode='tozero'
        ),
        xaxis=dict(
            title='predicted probability'
        ),
        plot_bgcolor = '#fff',
    )
    
    if cutoff is not None:
        layout['shapes'] = [dict(
                    type='line',
                    xref='x',
                    yref='y2',
                    x0=cutoff,
                    x1=cutoff,
                    y0=0,
                    y1=1.0,
                 )]
        
    fig = go.Figure(data=data, layout=layout)
    fig.update_layout(legend=dict(orientation="h",
                                    xanchor="center",
                                    y=-0.2,
                                    x=0.5))
    if cutoff is not None:
        fig.update_layout(annotations=[
            go.layout.Annotation(x=cutoff, y=0.1, yref='y2',
                                    text=f"cutoff={cutoff}")])
    return fig


def plotly_classification_plot(pred_probas, targets, labels=None, cutoff=0.5, 
                                pos_label=1, percentage=False):
    if len(pred_probas.shape) == 2:
        below = (pred_probas[:, pos_label] < cutoff)
    else:
        below = pred_probas < cutoff

    below_threshold = (pred_probas[below], targets[below])
    above_threshold = (pred_probas[~below], targets[~below])
    x = ['below cutoff', 'above cutoff', 'all']
    
    fig = go.Figure()
    for i, label in enumerate(labels):
        text = [f"<b>{sum(below_threshold[1]==i)}</b> ({np.round(100*np.mean(below_threshold[1]==i), 2)}%)",
                f"<b>{sum(above_threshold[1]==i)}</b> ({np.round(100*np.mean(above_threshold[1]==i), 2)}%)", 
                f"<b>{sum(targets==i)}</b> ({np.round(100*np.mean(targets==i), 2)}%)"]
        if percentage:
            fig.add_trace(go.Bar(
                x=x, 
                y=[100*np.mean(below_threshold[1]==i),
                    100*np.mean(above_threshold[1]==i), 
                    100*np.mean(targets==i)], 
                # text=[str(np.round(100*np.mean(below_threshold[1]==i), 2)) + '%',
                #       str(np.round(100*np.mean(above_threshold[1]==i), 2)) + '%', 
                #       str(np.round(100*np.mean(targets==i), 2)) + '%'],
                text=text,
                textposition='auto',
                hoverinfo="text",
                name=label))
            fig.update_layout(title='Percentage above and below cutoff')
        else:
            fig.add_trace(go.Bar(
                x=x, 
                y=[sum(below_threshold[1]==i),
                    sum(above_threshold[1]==i), 
                    sum(targets==i)], 
                # text = [sum(below_threshold[1]==i),
                #     sum(above_threshold[1]==i), 
                #     sum(targets==i)], 
                text=text,
                textposition='auto',
                hoverinfo="text",
                name=label))
            fig.update_layout(title='Total above and below cutoff')

    fig.update_layout(barmode='stack')

    fig.update_layout(legend=dict(orientation="h",
                                    xanchor="center",
                                    y=-0.2,
                                    x=0.5))

    return fig


def plotly_lift_curve(lift_curve_df, cutoff=None, percentage=False, round=2):
    """
    returns a lift plot for values generated with get_lift_curve_df(pred_proba, y)
    """
    
    if percentage:
        model_text=[f"model selected {np.round(pos, round)}% of all positives in first {np.round(i, round)}% sampled<br>" \
                    + f"precision={np.round(precision, 2)}% positives in sample<br>" \
                    + f"lift={np.round(pos/exp, 2)}" 
                  for (i, pos, exp, precision) in zip(lift_curve_df.index_percentage, 
                                                      lift_curve_df.cumulative_percentage_pos,
                                                      lift_curve_df.random_cumulative_percentage_pos, 
                                                      lift_curve_df.precision)]
        
        random_text=[f"random selected {np.round(exp, round)}% of all positives in first {np.round(i, round)}% sampled<br>" \
                     + f"precision={np.round(precision, 2)}% positives in sample"
                  for (i, pos, exp, precision) in zip(lift_curve_df.index_percentage,
                                                      lift_curve_df.cumulative_percentage_pos, 
                                                      lift_curve_df.random_cumulative_percentage_pos, 
                                                      lift_curve_df.random_precision)]
    else:
        model_text=[f"model selected {pos} positives out of {i}<br>" \
                    + f"precision={np.round(precision, 2)}<br>" \
                    + f"lift={np.round(pos/exp, 2)}" 
                  for (i, pos, exp, precision) in zip(lift_curve_df['index'], 
                                                      lift_curve_df.positives,
                                                      lift_curve_df.random_pos, 
                                                      lift_curve_df.precision)]
        random_text=[f"random selected {np.round(exp).astype(int)} positives out of {i}<br>" \
                     + f"precision={np.round(precision, 2)}"
                  for (i, pos, exp, precision) in zip(lift_curve_df['index'], 
                                                      lift_curve_df.positives, 
                                                      lift_curve_df.random_pos, 
                                                      lift_curve_df.random_precision)]
        
    
    trace0 = go.Scatter(
        x=lift_curve_df['index_percentage'].values if percentage else lift_curve_df['index'],
        y=np.round(lift_curve_df.cumulative_percentage_pos.values, round) if percentage \
                    else np.round(lift_curve_df.positives.values, round),
        name='model',
        text=model_text,
        hoverinfo="text",
    )

    trace1 = go.Scatter(
        x=lift_curve_df['index_percentage'].values if percentage else lift_curve_df['index'],
        y=np.round(lift_curve_df.random_cumulative_percentage_pos.values, round) if percentage \
                    else np.round(lift_curve_df.random_pos.values, round),
        name='random',
        text=random_text,
        hoverinfo="text",
    )

    data = [trace0, trace1]

    fig = go.Figure(data)

    fig.update_layout(title=dict(text='Lift curve', 
                                 x=0.5, 
                                 font=dict(size=18)),
                        xaxis_title= 'Percentage sampled' if percentage else 'Number sampled',
                        yaxis_title='Percentage of positive' if percentage else 'Number of positives',
                        xaxis=dict(spikemode="across"),
                        hovermode="x",
                        plot_bgcolor = '#fff')

    fig.update_layout(legend=dict(xanchor="center", y=0.9, x=0.1))
    
    if cutoff is not None:
        #cutoff_idx = max(0, (np.abs(lift_curve_df.pred_proba - cutoff)).argmin() - 1)
        cutoff_idx = max(0, len(lift_curve_df[lift_curve_df.pred_proba >= cutoff])-1)
        if percentage:
            cutoff_x = lift_curve_df['index_percentage'].iloc[cutoff_idx]
        else:
            cutoff_x = lift_curve_df['index'].iloc[cutoff_idx] 

        cutoff_n = lift_curve_df['index'].iloc[cutoff_idx] 
        cutoff_pos = lift_curve_df['positives'].iloc[cutoff_idx]
        cutoff_random_pos = int(lift_curve_df['random_pos'].iloc[cutoff_idx])
        cutoff_lift = np.round(lift_curve_df['positives'].iloc[cutoff_idx] / lift_curve_df.random_pos.iloc[cutoff_idx], 1)
        cutoff_precision = np.round(lift_curve_df['precision'].iloc[cutoff_idx], 2)
        cutoff_random_precision = np.round(lift_curve_df['random_precision'].iloc[cutoff_idx], 2)

        fig.update_layout(shapes = [dict(
                                        type='line',
                                        xref='x',
                                        yref='y',
                                        x0=cutoff_x,
                                        x1=cutoff_x,
                                        y0=0,
                                        y1=100.0 if percentage else lift_curve_df.positives.max(),
                                     )]
        )
        fig.update_layout(annotations=[
                                    go.layout.Annotation(
                                        x=cutoff_x, 
                                        y=5, 
                                        yref='y',
                                        text=f"cutoff={np.round(cutoff,3)}"),
                                    go.layout.Annotation(x=0.5, y=0.4, 
                                        text=f"Model: {cutoff_pos} out {cutoff_n} ({cutoff_precision}%)",
                                        showarrow=False, align="right", 
                                        xref='paper', yref='paper',
                                        xanchor='left', yanchor='top'
                                        ),
                                    go.layout.Annotation(x=0.5, y=0.33, 
                                        text=f"Random: {cutoff_random_pos} out {cutoff_n} ({cutoff_random_precision}%)",
                                        showarrow=False, align="right", 
                                        xref='paper', yref='paper',
                                        xanchor='left', yanchor='top'
                                        ),
                                    go.layout.Annotation(x=0.5, y=0.26, 
                                        text=f"Lift: {cutoff_lift}",
                                        showarrow=False, align="right", 
                                        xref='paper', yref='paper',
                                        xanchor='left', yanchor='top'
                                        )
        ])
    return fig


def plotly_cumulative_precision_plot(lift_curve_df, labels=None, pos_label=1):
    if labels is None:
        labels = ['category ' + str(i) for i in range(lift_curve_df.y.max()+1)]
    fig = go.Figure()
    text = [f"percentage sampled = top {round(idx_perc,2)}%"
                for idx_perc in lift_curve_df['index_percentage'].values]
    fig = fig.add_trace(go.Scatter(x=lift_curve_df.index_percentage, 
                                   y=np.zeros(len(lift_curve_df)),
                                   showlegend=False,
                                   text=text,
                                   hoverinfo="text")) 

    text = [f"percentage {labels[pos_label]}={round(perc, 2)}%" 
                for perc in lift_curve_df['precision_' +str(pos_label)].values]
    fig = fig.add_trace(go.Scatter(x=lift_curve_df.index_percentage, 
                                   y=lift_curve_df['precision_' +str(pos_label)].values, 
                                   fill='tozeroy', 
                                   name=labels[pos_label]+"(positive label)",
                                   text=text,
                                   hoverinfo="text")) 

    cumulative_y = lift_curve_df['precision_' +str(pos_label)].values
    for y_label in range(pos_label, lift_curve_df.y.max()+1):
        
        if y_label != pos_label:
            cumulative_y = cumulative_y + lift_curve_df['precision_' +str(y_label)].values
            text = [f"percentage {labels[y_label]}={round(perc, 2)}%" 
                for perc in lift_curve_df['precision_' +str(y_label)].values]
            fig=fig.add_trace(go.Scatter(x=lift_curve_df.index_percentage, 
                                         y=cumulative_y, 
                                         fill='tonexty', 
                                         name=labels[y_label],
                                         text=text,
                                         hoverinfo="text")) 

    for y_label in range(0, pos_label): 
        if y_label != pos_label:
            cumulative_y = cumulative_y + lift_curve_df['precision_' +str(y_label)].values
            text = [f"percentage {labels[y_label]}={round(perc, 2)}%" 
                for perc in lift_curve_df['precision_' +str(y_label)].values]
            fig=fig.add_trace(go.Scatter(x=lift_curve_df.index_percentage, 
                                         y=cumulative_y, 
                                         fill='tonexty', 
                                         name=labels[y_label],
                                         text=text,
                                         hoverinfo="text")) 
    fig.update_layout(title=dict(text='Cumulative percentage per category when sampling top X%', 
                                 x=0.5, 
                                 font=dict(size=18)),
                     yaxis=dict(title='Cumulative precision per category'),
                     xaxis=dict(title='Top X% model scores', spikemode="across"),
                     hovermode="x",
                     plot_bgcolor = '#fff')
    fig.update_xaxes(nticks=10)

    return fig


def plotly_dependence_plot(X, shap_values, col_name, interact_col_name=None, 
                            highlight_idx=None, interaction=False, na_fill=-999, round=2):
    """
    Returns a partial dependence plot based on shap values.

    Observations are colored according to the values in column interact_col_name
    """
    assert col_name in X.columns.tolist(), f'{col_name} not in X.columns'
    assert (interact_col_name is None and not interaction) or interact_col_name in X.columns.tolist(),\
            f'{interact_col_name} not in X.columns'
    
    x = X[col_name].replace({-999:np.nan})
    if len(shap_values.shape)==2:
        y = shap_values[:, X.columns.get_loc(col_name)]
    elif len(shap_values.shape)==3 and interact_col_name is not None:
        y = shap_values[:, X.columns.get_loc(col_name), X.columns.get_loc(interact_col_name)]
    else:
        raise Exception('Either provide shap_values or shap_interaction_values with an interact_col_name')
    
    if interact_col_name is not None:
        text = np.array([f'{col_name}={col_val}<br>{interact_col_name}={col_col_val}<br>SHAP={shap_val}' 
                    for col_val, col_col_val, shap_val in zip(x, X[interact_col_name], np.round(y, round))])
    else:
        text = np.array([f'{col_name}={col_val}<br>SHAP={shap_val}' 
                    for col_val, shap_val in zip(x, np.round(y, round))])  
        
    data = []
    
    if interact_col_name is not None and is_string_dtype(X[interact_col_name]):
        for onehot_col in X[interact_col_name].unique().tolist():
                data.append(go.Scatter(
                                x=X[X[interact_col_name]==onehot_col][col_name].replace({-999:np.nan}),
                                y=shap_values[X[interact_col_name]==onehot_col, X.columns.get_loc(col_name)],
                                mode='markers',
                                marker=dict(
                                      size=7,
                                      showscale=False,
                                      opacity=0.6,
                                  ),
                                
                                showlegend=True,
                                opacity=0.8,
                                hoverinfo="text",
                                name=onehot_col,
                                text=[f'{col_name}={col_val}<br>{interact_col_name}={col_col_val}<br>SHAP={shap_val}' 
                                        for col_val, col_col_val, shap_val in zip(
                                            X[X[interact_col_name]==onehot_col][col_name], 
                                            X[X[interact_col_name]==onehot_col][interact_col_name], 
                                            np.round(shap_values[X[interact_col_name]==onehot_col, X.columns.get_loc(col_name)], round))],
                                ))
                
    elif interact_col_name is not None and is_numeric_dtype(X[interact_col_name]):
        data.append(go.Scatter(
                        x=x[X[interact_col_name]!=na_fill],
                        y=y[X[interact_col_name]!=na_fill], 
                        mode='markers',
                        text=text[X[interact_col_name]!=na_fill],
                        hoverinfo="text",
                        marker=dict(size=7, 
                                    opacity=0.6,
                                    color=X[interact_col_name][X[interact_col_name]!=na_fill],
                                    colorscale='Bluered',
                                    colorbar=dict(
                                        title=interact_col_name
                                        ),
                                    showscale=True),    
                ))
        data.append(go.Scatter(
                        x=x[X[interact_col_name]==na_fill],
                        y=y[X[interact_col_name]==na_fill], 
                        mode='markers',
                        text=text[X[interact_col_name]==na_fill],
                        hoverinfo="text",
                        marker=dict(size=7, 
                                    opacity=0.35,
                                    color='grey'),
                ))
    else:
        data.append(go.Scatter(
                        x=x, 
                        y=y, 
                        mode='markers',
                        text=text,
                        hoverinfo="text",
                        marker=dict(size=7, 
                                    opacity=0.6)  ,                    
                ))
    if interaction:
        title = f'Interaction plot for {col_name} and {interact_col_name}'
    else:
        title = f'Dependence plot for {col_name}'

    layout = go.Layout(
            title=title,
            paper_bgcolor='#fff',
            plot_bgcolor = '#fff',
            showlegend=False,
            hovermode='closest',
            xaxis=dict(title=col_name),
            yaxis=dict(title='SHAP value')
        )
        
    fig = go.Figure(data, layout)
    
    if interact_col_name is not None and is_string_dtype(X[interact_col_name]):
        fig.update_layout(showlegend=True)
                                                      
    if isinstance(highlight_idx, int) and highlight_idx > 0 and highlight_idx < len(x):
        fig.add_trace(go.Scatter(
                x=[x[highlight_idx]], 
                y=[y[highlight_idx]], 
                mode='markers',
                marker=dict(
                    color='LightSkyBlue',
                    size=25,
                    opacity=0.5,
                    line=dict(
                        color='MediumPurple',
                        width=4
                    )
                ),
                name = f"index {highlight_idx}",
                text=f"index {highlight_idx}",
                hoverinfo="text"))
    fig.update_traces(selector = dict(mode='markers'))
    return fig


def plotly_shap_violin_plot(X, shap_values, col_name, color_col=None, points=False, interaction=False):
    """
    Returns a violin plot for categorical values. 

    if points=True or color_col is not None, a scatterplot of points is plotted next to the violin plots.
    If color_col is given, scatter is colored by color_col.
    """
    assert is_string_dtype(X[col_name]), \
        f'{col_name} is not categorical! Can only plot violin plots for categorical features!'
        
    x = X[col_name]
    shaps = shap_values[:, X.columns.get_loc(col_name)]
    n_cats = X[col_name].nunique()
    
    if points or color_col is not None:
        fig = make_subplots(rows=1, cols=2*n_cats, column_widths=[3, 1]*n_cats, shared_yaxes=True)
        showscale = True
    else:
        fig = make_subplots(rows=1, cols=n_cats, shared_yaxes=True)

    fig.update_yaxes(range=[shaps.min()*1.3, shaps.max()*1.3])  

    for i, cat in enumerate(X[col_name].unique()):
        col = 1+i*2 if points or color_col is not None else 1+i
        fig.add_trace(go.Violin(
                            x=x[x == cat],
                            y=shaps[x == cat],
                            name=cat,
                            box_visible=True,
                            meanline_visible=True,  
                            showlegend=False,
                               ),
                     row=1, col=col)
        if color_col is not None:
            if is_numeric_dtype(X[color_col]):
                fig.add_trace(go.Scatter(
                                x=np.random.randn(len(x[x == cat])),
                                y=shaps[x == cat],
                                name=color_col,
                                mode='markers',
                                showlegend=False,
                                hoverinfo="text",
                                hovertemplate = 
                                "<i>shap</i>: %{y:.2f}<BR>" +
                                f"<i>{color_col}" + ": %{marker.color}",
                                text = [f"shap: {shap}<>{color_col}: {col}" for shap, col in zip(shaps[x == cat], X[color_col][x==cat])],
                                marker=dict(size=7, 
                                        opacity=0.6,
                                        cmin=X[color_col].min(),
                                        cmax=X[color_col].max(),
                                        color=X[color_col][x==cat],
                                        colorscale='Bluered',
                                        showscale=showscale,
                                        colorbar=dict(title=color_col)),              
                                ),
                         row=1, col=col+1)
            else:
                n_color_cats = X[color_col].nunique()
                colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']
                colors = colors * (1+int(n_color_cats / len(colors)))
                colors = colors[:n_color_cats]
                for color_cat, color in zip(X[color_col].unique(), colors):
                    fig.add_trace(go.Scatter(
                                    x=np.random.randn(len(x[(x == cat) & (X[color_col] == color_cat)])),
                                    y=shaps[(x == cat) & (X[color_col] == color_cat)],
                                    name=color_cat,
                                   
                                    mode='markers',
                                    showlegend=showscale,
                                    hoverinfo="text",
                                    hovertemplate = 
                                    "<i>shap</i>: %{y:.2f}<BR>" +
                                    f"<i>{color_col}: {color_cat}",
                                    marker=dict(size=7, 
                                                opacity=0.8,
                                                color=color)           
                                    ),
                             row=1, col=col+1)
                
            showscale = False
        elif points:
            fig.add_trace(go.Scatter(
                            x=np.random.randn(len(x[x == cat])),
                            y=shaps[x == cat],
                            mode='markers',
                            showlegend=False,
                            hovertemplate = 
                            "<i>shap</i>: %{y:.2f}",
                            marker=dict(size=7, 
                                    opacity=0.6,
                                       color='blue'),
                        ), row=1, col=col+1)

    if points or color_col is not None:
        for i in range(n_cats):
            fig.update_xaxes(showgrid=False, zeroline=False, visible=False, row=1, col=2+i*2)
            fig.update_yaxes(showgrid=False, zeroline=False, row=1, col=2+i*2)
        if color_col is not None:
            fig.update_layout(title=f'Shap {"interaction" if interaction else None} values for {col_name}<br>(colored by {color_col})', hovermode='closest')
        else:
            fig.update_layout(title=f'Shap {"interaction" if interaction else None} values for {col_name}', hovermode='closest')
    else:
        fig.update_layout(title=f'Shap {"interaction" if interaction else None} values for {col_name}')
    
    return fig


def plotly_pdp(pdp_result, 
               display_index=None, index_feature_value=None, index_prediction=None,
               absolute=True, plot_lines=True, num_grid_lines=100, feature_name=None):
    """
    display_index: display the pdp of particular index
    index_feature_value: the actual feature value of the index to be highlighted
    index_prediction: the actual prediction of the index to be highlighted
    absolute: pdp ice lines absolute prediction or relative to base
    plot_lines: add ice lines or not
    num_grid_lines number of icelines to display
    feature_name: name of the feature that is being displayed, defaults to pdp_result.feature
    """ 
    if feature_name is None: feature_name = pdp_result.feature

    trace0 = go.Scatter(
            x = pdp_result.feature_grids,
            y = pdp_result.pdp.round(2) if absolute else (
                    pdp_result.pdp - pdp_result.pdp[0]).round(2),
            mode = 'lines+markers',
            line = dict(color='grey', width = 4),
            name = f'average prediction <br>for different values of <br>{pdp_result.feature}'
        )
    data = [trace0]

    if display_index is not None:
        # pdp_result.ice_lines.index = X.index
        trace1 = go.Scatter(
            x = pdp_result.feature_grids,
            y = pdp_result.ice_lines.iloc[display_index].values.round(2) if absolute else \
                pdp_result.ice_lines.iloc[display_index].values - pdp_result.ice_lines.iloc[display_index].values[0],
            mode = 'lines+markers',
            line = dict(color='blue', width = 4),
            name = f'prediction for index {display_index} <br>for different values of <br>{pdp_result.feature}'
        )
        data.append(trace1)

    if plot_lines:
        x = pdp_result.feature_grids
        ice_lines = pdp_result.ice_lines.sample(num_grid_lines)
        ice_lines = ice_lines.values if absolute else\
                    ice_lines.values - np.expand_dims(ice_lines.iloc[:, 0].transpose().values, axis=1)

        for y in ice_lines:
            data.append(
                go.Scatter(
                    x = x,
                    y = y,
                    mode='lines',
                    hoverinfo='skip',
                    line=dict(color='grey'),
                    opacity=0.1,
                    showlegend=False             
                )
            )

    layout = go.Layout(title = f'pdp plot for {feature_name}',
                        plot_bgcolor = '#fff',)

    fig = go.Figure(data=data, layout=layout)

    shapes = []
    annotations = []

    if index_feature_value is not None:
        if not isinstance(index_feature_value, str):
            index_feature_value = np.round(index_feature_value, 2)

        shapes.append(
                    dict(
                        type='line',
                        xref='x',
                        yref='y',
                        x0=index_feature_value,
                        x1=index_feature_value,
                        y0=np.min(ice_lines) if plot_lines else \
                            np.min(pdp_result.pdp),
                        y1=np.max(ice_lines) if plot_lines \
                            else np.max(pdp_result.pdp),
                        line=dict(
                            color="MediumPurple",
                            width=4,
                            dash="dot",
                        ),
                         ))
        annotations.append(
            go.layout.Annotation(x=index_feature_value, 
                                 y=np.min(ice_lines) if plot_lines else \
                                    np.min(pdp_result.pdp),
                                 text=f"baseline value = {index_feature_value}"))

    if index_prediction is not None:
        shapes.append(
                    dict(
                        type='line',
                        xref='x',
                        yref='y',
                        x0=pdp_result.feature_grids[0],
                        x1=pdp_result.feature_grids[-1],
                        y0=index_prediction,
                        y1=index_prediction,
                        line=dict(
                            color="MediumPurple",
                            width=4,
                            dash="dot",
                        ),
                         ))
        annotations.append(
            go.layout.Annotation(
                x=pdp_result.feature_grids[
                            round(0.5*len(pdp_result.feature_grids))], 
                y=index_prediction, 
                text=f"baseline pred = {np.round(index_prediction,2)}"))

    fig.update_layout(annotations=annotations)
    fig.update_layout(shapes=shapes)
    fig.update_layout(showlegend=False)
    return fig


def plotly_importances_plot(importance_df, descriptions=None, round=3):
    
    importance_name = importance_df.columns[1] # can be "MEAN_ABS_SHAP", "Permutation Importance", etc
    longest_feature_name = importance_df['Feature'].str.len().max()

    imp = importance_df.sort_values(importance_name)

    feature_names = [str(len(imp)-i)+". "+col 
            for i, col in enumerate(imp.iloc[:, 0].astype(str).values.tolist())]

    importance_values = imp.iloc[:,1]

    data = [go.Bar(
                y=feature_names,
                x=importance_values,
                #text=importance_values.round(round),
                text=descriptions[::-1] if descriptions is not None else None, #don't know why, but order needs to be reversed
                #textposition='inside',
                #insidetextanchor='end',
                hoverinfo="text",
                orientation='h')]

    layout = go.Layout(
        title=importance_name,
        plot_bgcolor = '#fff',
        showlegend=False
    )
    fig = go.Figure(data=data, layout=layout)
    fig.update_yaxes(automargin=True)
    fig.update_xaxes(automargin=True)

    left_margin = longest_feature_name*7
    if np.isnan(left_margin):
        left_margin = 100

    fig.update_layout(height=200+len(importance_df)*20,
                      margin=go.layout.Margin(
                                l=left_margin,
                                r=50,
                                b=50,
                                t=50,
                                pad=4
                            ))
    return fig


def plotly_tree_predictions(model, observation, highlight_tree=None, round=2, pos_label=1):
    """
    returns a plot with all the individual predictions of the 
    DecisionTrees that make up the RandomForest.
    """
    assert (str(type(model)).endswith("RandomForestClassifier'>") 
            or str(type(model)).endswith("RandomForestRegressor'>")), \
        f"model is of type {type(model)}, but should be either RandomForestClassifier or RandomForestRegressor"
    
    colors = ['blue'] * len(model.estimators_) 
    if highlight_tree is not None:
        assert highlight_tree >= 0 and highlight_tree <= len(model.estimators_), \
            f"{highlight_tree} is out of range (0, {len(model.estimators_)})"
        colors[highlight_tree] = 'red'
        
    if model.estimators_[0].classes_[0] is not None: #if classifier
        preds_df = (
            pd.DataFrame({
                'model' : range(len(model.estimators_)), 
                'prediction' : [
                        np.round(100*m.predict_proba(observation)[0, pos_label], round) 
                                    for m in model.estimators_],
                'color' : colors
            })
            .sort_values('prediction')\
            .reset_index(drop=True))
    else:
        preds_df = (
            pd.DataFrame({
                'model' : range(len(model.estimators_)), 
                'prediction' : [np.round(m.predict(observation)[0] , round)
                                    for m in model.estimators_],
                'color' : colors
            })
            .sort_values('prediction')\
            .reset_index(drop=True))
      
    trace0 = go.Bar(x=preds_df.index, 
                    y=preds_df.prediction, 
                    marker_color=preds_df.color,
                    text=[f"tree no {t}:<br> prediction={p}<br> click for detailed info"
                             for (t, p) in zip(preds_df.model.values, preds_df.prediction.values)],
                    hoverinfo="text")
    
    layout = go.Layout(
                title='individual predictions trees',
                plot_bgcolor = '#fff',
            )
    fig = go.Figure(data = [trace0], layout=layout)
    shapes = [dict(
                type='line',
                xref='x',
                yref='y',
                x0=0,
                x1=preds_df.model.max(),
                y0=preds_df.prediction.mean(),
                y1=preds_df.prediction.mean(),
                line=dict(
                    color="darkslategray",
                    width=4,
                    dash="dot"),
                )]
    
    annotations = [go.layout.Annotation(x=preds_df.model.mean(), 
                                         y=preds_df.prediction.mean(),
                                         text=f"Average prediction = {np.round(preds_df.prediction.mean(),2)}")]

    fig.update_layout(annotations=annotations)
    fig.update_layout(shapes=shapes)
    return fig 



def plotly_confusion_matrix(y_true, y_preds, labels = None, normalized=True):

    cm = confusion_matrix(y_true, y_preds)

    if labels is None:
        labels = [str(i) for i in range(cm.shape[0])] 

    if normalized:
        cm = np.round(100*cm / cm.sum(),1)
        zmax = 100
    else:
        zmax = len(y_true)
        
    data=[go.Heatmap(
                        z=cm,
                        x=[f'predicted {lab}' if len(lab) < 5 else f'predicted<br>{lab}' for lab in labels],
                        y=[f'actual {lab}' if len(lab) < 5 else f'actual<br>{lab}' for lab in labels],
                        hoverinfo="skip",
                        zmin=0, zmax=zmax, colorscale='Blues',
                        showscale=False,
                    )]
    layout = go.Layout(
            title="Confusion Matrix",
            xaxis=dict(side='top', constrain="domain"),
            yaxis=dict(autorange="reversed", side='left',
                        scaleanchor='x', scaleratio=1),
            plot_bgcolor = '#fff',
        )
    fig = go.Figure(data, layout)
    annotations = []
    for x in range(cm.shape[0]):
        for y in range(cm.shape[1]):
            text= str(cm[x,y]) + '%' if normalized else str(cm[x,y])
            annotations.append(
                go.layout.Annotation(
                    x=fig.data[0].x[y], 
                    y=fig.data[0].y[x], 
                    text=text, 
                    showarrow=False,
                    font=dict(size=20)
                )
            )

    fig.update_layout(annotations=annotations)  
    return fig


def plotly_roc_auc_curve(true_y, pred_probas, cutoff=None):
    fpr, tpr, thresholds = roc_curve(true_y, pred_probas)
    roc_auc = roc_auc_score(true_y, pred_probas)
    trace0 = go.Scatter(x=fpr, y=tpr,
                    mode='lines',
                    name='ROC AUC CURVE',
                    text=[f"threshold: {np.round(th,2)} <br> FP: {np.round(fp,2)} <br> TP: {np.round(tp,2)}" 
                              for fp, tp, th in zip(fpr, tpr, thresholds)],
                    hoverinfo="text"
                )
    data = [trace0]
    layout = go.Layout(title='ROC AUC CURVE', 
                    #    width=450,
                    #    height=450,
                       xaxis= dict(title='False Positive Rate', range=[0,1], constrain="domain"),
                       yaxis = dict(title='True Positive Rate', range=[0,1], constrain="domain", 
                                    scaleanchor='x', scaleratio=1),
                       hovermode='closest',
                       plot_bgcolor = '#fff',)
    fig = go.Figure(data, layout)
    shapes = [dict(
                            type='line',
                            xref='x',
                            yref='y',
                            x0=0,
                            x1=1,
                            y0=0,
                            y1=1,
                            line=dict(
                                color="darkslategray",
                                width=4,
                                dash="dot"),
                            )]
    
    if cutoff is not None:
        threshold_idx = np.argmin(np.abs(thresholds-cutoff))
        shapes.append(
            dict(type='line', xref='x', yref='y',
                x0=0, x1=1, y0=tpr[threshold_idx], y1=tpr[threshold_idx],
                line=dict(color="lightslategray",width=1)))
        shapes.append(
            dict(type='line', xref='x', yref='y',
                 x0=fpr[threshold_idx], x1=fpr[threshold_idx], y0=0, y1=1,
                 line=dict(color="lightslategray", width=1)))
        
        rep = classification_report(true_y, np.where(pred_probas >= cutoff, 1,0), 
                                    output_dict=True)
        
        annotations = [go.layout.Annotation(x=0.6, y=0.45, 
                            text=f"Cutoff: {np.round(cutoff,3)}",
                            showarrow=False, align="right", 
                            xanchor='left', yanchor='top'),
                       go.layout.Annotation(x=0.6, y=0.4, 
                            text=f"Accuracy: {np.round(rep['accuracy'],3)}",
                            showarrow=False, align="right", 
                            xanchor='left', yanchor='top'),
                       go.layout.Annotation(x=0.6, y=0.35, 
                            text=f"Precision: {np.round(rep['1']['precision'], 3)}",
                            showarrow=False, align="right", 
                            xanchor='left', yanchor='top'),
                       go.layout.Annotation(x=0.6, y=0.30, 
                            text=f"Recall: {np.round(rep['1']['recall'], 3)}",
                            showarrow=False, align="right", 
                            xanchor='left', yanchor='top'),
                       go.layout.Annotation(x=0.6, y=0.25, 
                            text=f"F1-score: {np.round(rep['1']['f1-score'], 3)}",
                            showarrow=False, align="right", 
                            xanchor='left', yanchor='top'),
                       go.layout.Annotation(x=0.6, y=0.20, 
                            text=f"roc-auc-score: {np.round(roc_auc, 3)}",
                            showarrow=False, align="right", 
                            xanchor='left', yanchor='top'),]
        fig.update_layout(annotations=annotations)
                                            
    fig.update_layout(shapes=shapes)
    return fig


def plotly_pr_auc_curve(true_y, pred_probas, cutoff=None):
    precision, recall, thresholds = precision_recall_curve(true_y, pred_probas)
    pr_auc_score = average_precision_score(true_y, pred_probas)
    trace0 = go.Scatter(x=precision, y=recall,
                    mode='lines',
                    name='PR AUC CURVE',
                    text=[f"threshold: {np.round(th,2)} <br>" +\
                          f"precision: {np.round(p,2)} <br>" +\
                          f"recall: {np.round(r,2)}" 
                            for p, r, th in zip(precision, recall, thresholds)],
                    hoverinfo="text"
                )
    data = [trace0]
    layout = go.Layout(title='PR AUC CURVE', 
                    #    width=450,
                    #    height=450,
                       xaxis= dict(title='Precision', range=[0,1], constrain="domain"),
                       yaxis = dict(title='Recall', range=[0,1], constrain="domain", 
                       scaleanchor='x', scaleratio=1),
                       hovermode='closest',
                       plot_bgcolor = '#fff',)
    fig = go.Figure(data, layout)
    shapes = [] 
    
    if cutoff is not None:
        threshold_idx = np.argmin(np.abs(thresholds-cutoff))
        shapes.append(
            dict(type='line', xref='x', yref='y',
                x0=0, x1=1, 
                y0=recall[threshold_idx], y1=recall[threshold_idx],
                line=dict(color="lightslategray",width=1)))
        shapes.append(
            dict(type='line', xref='x', yref='y',
                 x0=precision[threshold_idx], x1=precision[threshold_idx], 
                 y0=0, y1=1,
                 line=dict(color="lightslategray", width=1)))
        
        report = classification_report(
                    true_y, np.where(pred_probas > cutoff, 1,0), 
                    output_dict=True)
        
        annotations = [go.layout.Annotation(x=0.15, y=0.45, 
                            text=f"Cutoff: {np.round(cutoff,3)}",
                            showarrow=False, align="right", 
                            xanchor='left', yanchor='top'),
                       go.layout.Annotation(x=0.15, y=0.4, 
                            text=f"Accuracy: {np.round(report['accuracy'],3)}",
                            showarrow=False, align="right", 
                            xanchor='left', yanchor='top'),
                       go.layout.Annotation(x=0.15, y=0.35, 
                            text=f"Precision: {np.round(report['1']['precision'], 3)}",
                            showarrow=False, align="right", 
                            xanchor='left', yanchor='top'),
                       go.layout.Annotation(x=0.15, y=0.30, 
                            text=f"Recall: {np.round(report['1']['recall'], 3)}",
                            showarrow=False, align="right", 
                            xanchor='left', yanchor='top'),
                       go.layout.Annotation(x=0.15, y=0.25, 
                            text=f"F1-score: {np.round(report['1']['f1-score'], 3)}",
                            showarrow=False, align="right", 
                            xanchor='left', yanchor='top'),
                       go.layout.Annotation(x=0.15, y=0.20, 
                            text=f"pr-auc-score: {np.round(pr_auc_score, 3)}",
                            showarrow=False, align="right", 
                            xanchor='left', yanchor='top'),]
        fig.update_layout(annotations=annotations)
                                            
    fig.update_layout(shapes=shapes)
    return fig


def plotly_shap_scatter_plot(shap_values, X, display_columns):
    
    # make sure that columns are actually in X:
    display_columns = [col for col in display_columns if col in X.columns.tolist()]    
    shap_df = pd.DataFrame(shap_values, columns=X.columns)
    min_shap = np.round(shap_values.min()-0.01, 2)
    max_shap = np.round(shap_values.max()+0.01, 2)

    fig =  make_subplots(rows=len(display_columns), cols=1, 
                         subplot_titles=display_columns, shared_xaxes=True)
    
    for i, col in enumerate(display_columns):
        
        if is_string_dtype(X[col]): 
            # if str type then categorical variable, 
            # so plot each category in a different color:
            for onehot_col in X[col].unique().tolist():
                fig.add_trace(go.Scatter(
                                x=shap_df[X[col]==onehot_col][col],
                                y=np.random.rand(len(shap_df[X[col]==onehot_col])),
                                mode='markers',
                                marker=dict(
                                      size=5,
                                      showscale=False,
                                      opacity=0.3,
                                  ),
                                name=onehot_col,
                                showlegend=False,
                                opacity=0.8,
                                hoverinfo="text",
                                text=[f"{col}={onehot_col}<br>shap={np.round(shap,3)}<br>index={i}" 
                                      for i, shap in zip(shap_df[X[col]==onehot_col].index,
                                                       shap_df[X[col]==onehot_col][col])],
                                ),
                     row=i+1, col=1);
        else:
            # numerical feature get a single bluered plot
            fig.add_trace(go.Scatter(x=shap_df[col],
                                   y=np.random.rand(len(shap_df)),
                                  mode='markers',
                                  marker=dict(
                                      size=5,
                                      color=X[col].replace({-999:np.nan}),
                                      colorscale='Bluered',
                                      showscale=True,
                                      opacity=0.3,
                                      colorbar=dict(
                                        title="feature value <br> (red is high)", 
                                        showticklabels=False),
                                  ),
                                name=col,
                                showlegend=False,
                                opacity=0.8,
                                hoverinfo="text",
                                text=[f"{col}={value}<br>shap={np.round(shap,3)}<br>index={i}" 
                                      for i, (shap, value) in enumerate(zip(
                                            shap_df[col], 
                                            X[col].replace({-999:np.nan})))],
                                ),
                     row=i+1, col=1);
        fig.update_xaxes(showgrid=False, zeroline=False, 
                         range=[min_shap, max_shap], row=i+1, col=1)
        fig.update_yaxes(showgrid=False, zeroline=False, 
                         showticklabels=False, row=i+1, col=1)
    
    fig.update_layout(height=100+len(display_columns)*50,
                      margin=go.layout.Margin(
                                l=50,
                                r=50,
                                b=50,
                                t=50,
                                pad=4
                            ),
                      hovermode='closest',
                      plot_bgcolor = '#fff',)
    return fig


def plotly_predicted_vs_actual(y, preds, units="", round=2, logs=False):
    trace0 = go.Scatter(
        x = y,
        y = preds,
        mode='markers', 
        name='predicted',
        text=[f"Predicted: {predicted}<br>Actual: {actual}" for actual, predicted in zip(np.round(y, round), np.round(preds, round))],
        hoverinfo="text",
    )
    
    trace1 = go.Scatter(
        x = y,
        y = y,
        mode='lines', 
        name='actual',
        hoverinfo="none",
    )
    
    data = [trace0, trace1]
    
    layout = go.Layout(
        title='Predicted vs Actual',
        yaxis=dict(
            title=f'Predicted {units}' 
        ),
        xaxis=dict(
            title=f'Actual {units}' 
        ),
        plot_bgcolor = '#fff',
    )
    
    fig = go.Figure(data, layout)
    if logs:
        fig.update_layout(xaxis_type='log', yaxis_type='log')
    return fig


def plotly_plot_residuals(y, preds, vs_actual=False, units="", round=2, ratio=False):
    
    residuals = y - preds
    residuals_ratio = residuals / y if vs_actual else residuals / preds
    
    trace0 = go.Scatter(
        x=y if vs_actual else preds, 
        y=residuals_ratio if ratio else residuals, 
        mode='markers', 
        name='residual ratio' if ratio else 'residuals',
        text=[f"Actual: {actual}<br>Prediction: {pred}<br>Residual: {residual}<br>Ratio: {res_ratio}" 
                  for actual, pred, residual, res_ratio in zip(np.round(y, round), 
                                                                np.round(preds, round), 
                                                                np.round(residuals, round),
                                                                np.round(residuals_ratio, round))],
        hoverinfo="text",
    )
    
    trace1 = go.Scatter(
        x=y if vs_actual else preds, 
        y=np.zeros(len(preds)),
        mode='lines', 
        name=f'Actual {units}' if vs_actual else f'Predicted {units}',
        hoverinfo="none",
    )
    
    data = [trace0, trace1]
    
    layout = go.Layout(
        title='Residuals vs Predicted',
        yaxis=dict(
            title='Relative Residuals' if ratio else 'Residuals'
        ),
        
        xaxis=dict(
            title=f'Actual {units}' if vs_actual else f'Predicted {units}' 
        ),
        plot_bgcolor = '#fff',
    )
    
    fig = go.Figure(data, layout)
    return fig
    

def plotly_residuals_vs_col(y, preds, col, col_name=None, cat=False, units="", round=2, ratio=False):
    
    if col_name is None:
        try:
            col_name = col.name
        except:
            col_name = 'Feature' 
            
    residuals = y - preds
    residuals_ratio = residuals / y
    
    trace0 = go.Scatter(
        x=col, 
        y=residuals_ratio if ratio else residuals, 
        mode='markers', 
        name='residual ratio' if ratio else 'residuals',
        text=[f"Actual: {actual}<br>Prediction: {pred}<br>Residual: {residual}<br>Ratio: {res_ratio}" 
                  for actual, pred, residual, res_ratio in zip(np.round(y, round), 
                                                                np.round(preds, round), 
                                                                np.round(residuals, round),
                                                                np.round(residuals_ratio, round))],
        hoverinfo="text",
    )
    
    trace1 = go.Scatter(
        x=col, 
        y=np.zeros(len(preds)),
        mode='lines', 
        name=col_name,
        hoverinfo="none",
    )
    
    data = [trace0, trace1]
    
    layout = go.Layout(
        title=f'Residuals vs {col_name}',
        yaxis=dict(
            title='Relative Residuals' if ratio else 'Residuals'
        ),
        
        xaxis=dict(
            title=f'{col_name} value'
        ),
        plot_bgcolor = '#fff',
    )
    
    fig = go.Figure(data, layout)
    return fig