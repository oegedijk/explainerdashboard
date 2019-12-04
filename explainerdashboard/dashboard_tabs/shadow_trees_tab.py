__all__ = ['ShadowTreesTab']

import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_table

from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from .dashboard_methods import *


class ShadowTreesTab:
    def __init__(self, explainer, standalone=False, tab_id="shadow_trees", title='Shadow Trees',
                 round=2, **kwargs):
        self.explainer = explainer
        self.standalone = standalone
        self.tab_id = tab_id
        self.title = title

        self.round = round
        self.kwargs = kwargs
        
        
    def layout(self):
        if self.standalone:
            return shadow_trees_layout(self.explainer, title=self.title, standalone=self.standalone, 
                                     round=self.round)
        else:
            return shadow_trees_layout(self.explainer,  
                                     round=self.round)
    
    def register_callbacks(self, app):
        shadow_trees_callbacks(self.explainer, app, standalone=self.standalone)


def shadow_trees_layout(explainer, 
            title=None, standalone=False, hide_selector=False,
            round=2, **kwargs):
    """return layout for shadow trees tab that display distributions of individual
    prediction of DecisionTrees that make up RandomForest, and when clicked
    displays individual path through tree.
    
    :param explainer: RandomForestBunch Object (so should have shadow_trees attribute!)
    :type explainer: RandomForestBunch
    :type title: str
    :param standalone: when standalone layout, include a a label_store, defaults to False
    :type standalone: bool
    :param hide_selector: if model is a classifier, optionally hide the positive label selector, defaults to False
    :type hide_selector: bool
    :param round: precision to round floats, defaults to 2
    :type round: int, optional
    :rtype: dbc.Container
    """
    return dbc.Container([
        title_and_label_selector(explainer, title, standalone, hide_selector),
        dbc.Row([
            dbc.Col([
                html.H2('Predictions of individual decision trees.'),
                dbc.Form(
                    [
                        dbc.FormGroup(
                            [
                                dbc.Label("Index", className="mr-2"),
                                dbc.Input(id='tree-input-index', 
                                            placeholder="Fill in index here...",
                                            debounce=True),
                            ],
                            className="mr-3",
                        ),
                        dbc.Button("Random Index", color="primary", id='tree-index-button'),
                    ], inline=True),
                dcc.Store(id='tree-index-store'),
                html.H4('(click on a prediction to see decision path)'),
                dcc.Loading(id="loading-trees-graph", 
                            children=[dcc.Graph(id='tree-predictions-graph')]),  
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


def shadow_trees_callbacks(explainer, app, 
             standalone=False, round=2, **kwargs):

    if standalone:
        label_selector_register_callback(explainer, app)

    @app.callback(
            Output('tree-input-index', 'value'),
            [Input('tree-index-button', 'n_clicks')]
        )
    def update_tree_input_index(n_clicks):
            return explainer.random_index(return_str=True)

    @app.callback(
        Output('tree-index-store', 'data'),
        [Input('tree-input-index', 'value')]
    )
    def update_tree_index_store(index):
        if (explainer.idxs is None 
            and str(index).isdigit() 
            and int(index) >= 0
            and int(index) <= len(explainer)):
            return int(index)
        if (explainer.idxs is not None
             and index in explainer.idxs):
            return index
        raise PreventUpdate

    @app.callback(
        Output('tree-predictions-graph', 'figure'),
        [Input('tree-index-store', 'data'),
         Input('label-store', 'data')]
    )
    def update_tree_graph(index, pos_label):
        if index is not None:
            return explainer.plot_trees(index, round=round)
        raise PreventUpdate

    @app.callback(
        [Output('tree-basevalue', 'children'),
        Output('tree-predictions-table', 'columns'),
        Output('tree-predictions-table', 'data'),],
        [Input('tree-predictions-graph', 'clickData'),
         Input('tree-index-store', 'data'),
         Input('label-store', 'data')],
        [State('tree-predictions-table', 'columns')])
    def display_click_data(clickData, idx, pos_label, old_columns):
        if clickData is not None and idx is not None:
            model = int(clickData['points'][0]['text'].split('tree no ')[1].split(':')[0]) if clickData is not None else 0
            (baseval, prediction, 
                    shadowtree_df) = explainer.shadowtree_df_summary(model, idx, round=round)
            columns=[{'id': c, 'name': c} for c in  shadowtree_df.columns.tolist()]
            baseval_str = f"Tree no {model}, Starting prediction   : {baseval}, final prediction : {prediction}"
            return (baseval_str, columns, shadowtree_df.to_dict('records'))
        raise PreventUpdate