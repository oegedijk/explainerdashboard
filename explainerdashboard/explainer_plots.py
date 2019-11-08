import numpy as np
import pandas as pd
import plotly.graph_objs as go
from pandas.api.types import is_numeric_dtype, is_string_dtype
from plotly.subplots import make_subplots
from sklearn.metrics import (classification_report, confusion_matrix,
                             precision_recall_curve, roc_curve, 
                             roc_auc_score, average_precision_score)


def plotly_contribution_plot(contrib_df, target="target", 
                         classification=False, higher_is_better=False,
                         include_base_value=True, round=2):
    """
    Takes in a DataFrame contrib_df with columns
    'col' -- name of columns
    'contribution' -- contribution of that column to the final predictions
    'cumulatief' -- the summed contributions from the top to the current row
    'base' -- the summed contributions from the top excluding the current row

    Outputs a bar chart displaying the contribution of each individual
    column to the final prediction. 
    """ 
    contrib_df = contrib_df.copy()
    multiplier = 100 if classification else 1
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
    
    if classification:
        title = f'Contribution to prediction probability = {prediction}%'
    else:
        title = f'Contribution to prediction {target} = {prediction}'

    data = [trace0, trace1]
    layout = go.Layout(
        title=title,
        barmode='stack',
        paper_bgcolor='rgba(245, 246, 249, 1)',
        plot_bgcolor='rgba(245, 246, 249, 1)',
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
    if classification:
        fig.update_yaxes(title_text='Prediction %')
    else:
        fig.update_yaxes(title_text='Prediction')
    return fig

def plotly_precision_plot(precision_df, 
                            count_label='counts', 
                            pos_label='positive', 
                            cutoff=0.5):
    """
    returns a plotly figure with bar plots for counts of observations for a 
    certain pred_proba bin,
    and a line trace for the actual fraction of positive labels for that bin.
    
    precision_df generated from get_precision_df(predictions_df)
    predictions_df generated from get_predictions_df(rf_model, X, y)
    """

    precision_df = precision_df.copy()
    trace1 = go.Bar(
        x=precision_df['p_avg'].values.tolist(),
        y=precision_df['count'].values.tolist(),
        width=0.9*precision_df['bin_width'].values,
        name=count_label
    )
    trace2 = go.Scatter(
        x=precision_df['p_avg'].values.tolist(),
        y=precision_df['precision'].values.tolist(),
        name='percentage ' + pos_label,
        yaxis='y2'
    )
    data = [trace1, trace2]
    layout = go.Layout(
        title='percentage ' + pos_label + ' vs predicted probability',
        yaxis=dict(
            title=count_label
        ),
        yaxis2=dict(
            title='percentage ' + pos_label,
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
        )
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
    if cutoff is not None:
        fig.update_layout(annotations=[
            go.layout.Annotation(x=cutoff, y=0.1, yref='y2',
                                    text=f"cutoff={cutoff}")])
    return fig


def plotly_dependence_plot(X, shap_values, col_name, interact_col_name=None, 
                            highlight_idx=None, na_fill=-999, round=2):
    """
    Returns a partial dependence plot based on shap values.

    Observations are colored according to the values in column interact_col_name
    """
    assert col_name in X.columns.tolist(), f'{col_name} not in X.columns'
    assert interact_col_name is None or interact_col_name in X.columns.tolist(),\
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

    layout = go.Layout(
            title=f'dependence plot for {col_name}',
            paper_bgcolor='rgba(245, 246, 249, 1)',
            plot_bgcolor='rgba(245, 246, 249, 1)',
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
    fig.update_traces(selector=dict(mode='markers'))
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

    layout = go.Layout(title = f'pdp plot for {feature_name}')

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


def plotly_importances_plot(importance_df):
    
    importance_name = importance_df.columns[1]
    longest_feature_name = importance_df['Feature'].str.len().max()
    
    imp = importance_df.sort_values(importance_name)

    trace0 = go.Bar(
                y=imp.iloc[:,0],
                x=imp.iloc[:,1],
                text=imp.iloc[:,1].round(4),
                textposition='inside',
                insidetextanchor='end',
                hoverinfo="text",
                orientation='h')

    data = [trace0]
    layout = go.Layout(
        title=importance_name,
        paper_bgcolor='rgba(245, 246, 249, 1)',
        plot_bgcolor='rgba(245, 246, 249, 1)',
        showlegend=False
    )
    fig = go.Figure(data=data, layout=layout)
    fig.update_yaxes(automargin=True)
    fig.update_xaxes(automargin=True)
    fig.update_layout(height=200+len(importance_df)*20,
                      margin=go.layout.Margin(
                                l=longest_feature_name*7,
                                r=50,
                                b=50,
                                t=50,
                                pad=4
                            ))
    return fig


def plotly_tree_predictions(model, observation, round=2):
    """
    returns a plot with all the individual predictions of the 
    DecisionTrees that make up the RandomForest.
    """
    if model.estimators_[0].classes_ is not None: #if classifier
         preds_df = pd.DataFrame({
                'model' : range(len(model.estimators_)), 
                'prediction' : [
                        np.round(100*m.predict_proba(observation)[0, 1], round) 
                                    for m in model.estimators_]
            })\
            .sort_values('prediction')\
            .reset_index(drop=True)
    else:
        preds_df = pd.DataFrame({
            'model' : range(len(model.estimators_)), 
            'prediction' : [np.round(m.predict(observation)[0] , round)
                                for m in model.estimators_]})\
            .sort_values('prediction')\
            .reset_index(drop=True)
        
    trace0 = go.Bar(x=preds_df.index, 
                    y=preds_df.prediction, 
                    text=[f"tree no {t}:<br> prediction={p}<br> click for detailed info"
                             for (t, p) in zip(preds_df.model.values, preds_df.prediction.values)],
                    hoverinfo="text")
    
    layout = go.Layout(
                title='individual predictions trees'
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



def plotly_confusion_matrix(y_true, pred_probas, cutoff=0.5, 
                            labels = None, normalized=True):

    cm = confusion_matrix(y_true, np.where(pred_probas>cutoff,1,0))
    cm_labels = np.array([['TN', 'FP'],['FN', 'TP']])
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
                    )]
    #     annotations = [go.layout.Annotation()]
    layout = go.Layout(
            title="Confusion Matrix",
            width=450,
            height=450,
            xaxis=dict(side='top'),
            yaxis=dict(autorange="reversed",side='left')
        )
    fig = go.Figure(data, layout)
    
    annotations = []

    for x in range(cm.shape[0]):
        for y in range(cm.shape[1]):
            text= str(cm[x,y]) + '%' if normalized else str(cm[x,y])
            annotations.append(
                go.layout.Annotation(x=fig.data[0].x[y], 
                                    y=fig.data[0].y[x], 
                                    text=text, 
                                    showarrow=False,
                                    font=dict(
                                        size=20
                                    ),))
            annotations.append(
                go.layout.Annotation(x=fig.data[0].x[y], 
                                    y=fig.data[0].y[x], 
                                    text=cm_labels[x,y], 
                                    showarrow=False,
                                    font=dict(
                                        family= "Old Standard TT, Bold", 
                                        size=90,
                                        color="black"
                                    ),
                                    opacity=0.05,
                                    ))

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
                       width=450,
                       height=450,
                       xaxis= dict(title='False Positive Rate', range=[0,1]),
                       yaxis = dict(title='True Positive Rate', range=[0,1], 
                                    scaleanchor='y', scaleratio=1))
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
                       width=450,
                       height=450,
                       xaxis= dict(title='Precision', range=[0,1]),
                       yaxis = dict(title='Recall', range=[0,1], 
                       scaleanchor='y', scaleratio=1))
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
                      hovermode='closest')
    return fig
