__all__ = ['shadow_trees_tab', 'shadow_trees_tab_register_callbacks']

import dash
import dash_core_components as dcc
import dash_daq as daq
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_table

from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

def shadow_trees_tab(explainer, round=2, **kwargs):
    """return layout for shadow trees tab that display distributions of individual
    prediction of DecisionTrees that make up RandomForest, and when clicked
    displays individual path through tree.
    
    :param explainer: RandomForestBunch Object (so should have shadow_trees attribute!)
    :type explainer: RandomForestBunch
    :param round: precision to round floats, defaults to 2
    :type round: int, optional
    :rtype: dbc.Container
    """
    return dbc.Container([
     dbc.Row([
        dbc.Col([
            html.H2('Predictions of individual decision trees.'),
            dbc.Input(id='tree-input-index', placeholder="Fill in index here...",
                        debounce=True),
            dcc.Store(id='tree-index-store'),
            html.H4('(click on a prediction to see decision path)'),
            dcc.Graph(id='tree-predictions-graph'),  
        ], width={"size": 8, "offset": 2})
    ]), 
    dbc.Row([
        dbc.Col([
            html.Label('Decision path in decision tree:'),
            dcc.Markdown(id='tree-basevalue'),
            dash_table.DataTable(
                id='tree-predictions-table',
                style_cell={'fontSize':20, 'font-family':'sans-serif'},
            ),
        ], width={"size": 6, "offset": 3})
    ]), 
    ],  fluid=True)


def shadow_trees_tab_register_callbacks(explainer, app, round=2, **kwargs):

    @app.callback(
        Output('tree-index-store', 'data'),
        [Input('tree-input-index', 'value')]
    )
    def update_bsn_div(input_index):
        if (explainer.idxs is None 
            and str(input_index).isdigit() 
            and int(input_index) <= len(explainer)):
            return int(input_index)
        elif (explainer.idxs is not None
             and str(input_index) in explainer.idxs):
             return str(input_index)
        raise PreventUpdate

    @app.callback(
        Output('tree-predictions-graph', 'figure'),
        [Input('tree-index-store', 'data')]
    )
    def update_output_div(idx):
        if idx is not None:
            return explainer.plot_trees(idx, round=round)
        raise PreventUpdate

    @app.callback(
        [Output('tree-basevalue', 'children'),
        Output('tree-predictions-table', 'columns'),
        Output('tree-predictions-table', 'data'),],
        [Input('tree-predictions-graph', 'clickData'),
         Input('tree-index-store', 'data')],
        [State('tree-predictions-table', 'columns')])
    def display_click_data(clickData, idx, old_columns):
        if clickData is not None and idx is not None:
            model = int(clickData['points'][0]['text'].split('tree no ')[1].split(':')[0]) if clickData is not None else 0
            (baseval, prediction, 
                    shadowtree_df) = explainer.shadowtree_df_summary(model, idx, round=round)
            columns=[{'id': c, 'name': c} for c in  shadowtree_df.columns.tolist()]
            baseval_str = f"Tree no {model}, Starting prediction   : {baseval}, final prediction : {prediction}"
            return (baseval_str, columns, shadowtree_df.to_dict('records'))
        raise PreventUpdate