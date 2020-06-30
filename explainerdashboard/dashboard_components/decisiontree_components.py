__all__ = [
    'DecisionTreesComponent',
    'DecisionPathTableComponent',
    'DecisionPathGraphComponent',
]

import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_table

from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from .dashboard_methods import *
from .connectors import ClassifierRandomIndexComponent, IndexConnector, HighlightConnector



class DecisionTreesComponent(ExplainerComponent):
    def __init__(self, explainer, title="Decision Trees",
                    header_mode="none", name=None,
                    hide_index=False, hide_highlight=False,
                    index=None, highlight=None):
        super().__init__(explainer, title, header_mode, name)
        self.hide_index, self.hide_highlight = hide_index, hide_highlight
        self.index, self.highlight = index, highlight

        self.index_name = 'decisiontrees-index-'+self.name
        self.highlight_name = 'decisiontrees-highlight-'+self.name

    def _layout(self):
        return html.Div([
            html.H3("Decision trees:"),
            dbc.Row([
                make_hideable(
                    dbc.Col([
                        dbc.Label("Index:"),
                        dcc.Dropdown(id='decisiontrees-index-'+self.name, 
                            options = [{'label': str(idx), 'value':idx} 
                                            for idx in self.explainer.idxs],
                            value=self.index)
                    ], md=4), hide=self.hide_index),
                make_hideable(
                    dbc.Col([
                        dbc.Label("Highlight tree:"),
                        dcc.Dropdown(id='decisiontrees-highlight-'+self.name, 
                            options = [{'label': str(tree), 'value': tree} 
                                            for tree in range(self.explainer.no_of_trees)],
                            value=self.highlight)
                    ], md=2), hide=self.hide_highlight), 
            ]),
            dbc.Row([
                dbc.Col([
                    dcc.Loading(id="loading-decisiontrees-graph-"+self.name, 
                        children=dcc.Graph(id="decisiontrees-graph-"+self.name)),  
                ])
            ]),
        ])

    def _register_callbacks(self, app):
        @app.callback(
            Output("decisiontrees-graph-"+self.name, 'figure'),
            [Input('decisiontrees-index-'+self.name, 'value'),
             Input('decisiontrees-highlight-'+self.name, 'value'),
             Input('pos-label', 'value')],
            [State('tabs', 'value')]
        )
        def update_tree_graph(index, highlight, pos_label, tab):
            if index is not None:
                return self.explainer.plot_trees(index, highlight_tree=highlight, pos_label=pos_label)
            return {}

        @app.callback(
            Output('decisiontrees-highlight-'+self.name, 'value'),
            [Input("decisiontrees-graph-"+self.name, 'clickData')])
        def update_highlight(clickdata):
            highlight_tree = int(clickdata['points'][0]['text'].split('tree no ')[1].split(':')[0]) if clickdata is not None else None
            if highlight_tree is not None:
                return highlight_tree
            raise PreventUpdate

class DecisionPathTableComponent(ExplainerComponent):
    def __init__(self, explainer, title="Decision path table",
                    header_mode="none", name=None,
                    hide_index=False, hide_highlight=False,
                    index=None, highlight=None):
        super().__init__(explainer, title, header_mode, name)
        self.hide_index, self.hide_highlight = hide_index, hide_highlight
        self.index, self.highlight = index, highlight

        self.index_name = 'decisionpath-table-index-'+self.name
        self.highlight_name = 'decisionpath-table-highlight-'+self.name
        self.register_dependencies("decision_trees")

    def _layout(self):
        return html.Div([
            html.H3("Decision path:"),
            dbc.Row([
                make_hideable(
                    dbc.Col([
                        dbc.Label("Index:"),
                        dcc.Dropdown(id='decisionpath-table-index-'+self.name, 
                            options = [{'label': str(idx), 'value':idx} 
                                            for idx in self.explainer.idxs],
                            value=self.index)
                    ], md=4), hide=self.hide_index),
                    make_hideable(
                    dbc.Col([
                        dbc.Label("Highlight tree:"),
                        dcc.Dropdown(id='decisionpath-table-highlight-'+self.name, 
                            options = [{'label': str(tree), 'value': tree} 
                                            for tree in range(self.explainer.no_of_trees)],
                            value=self.highlight)
                    ], md=2), hide=self.hide_highlight), 
            ]),
            dbc.Row([
                dbc.Col([
                    dcc.Loading(id="loading-decisionpath-table-"+self.name, 
                        children=html.Div(id="decisionpath-table-"+self.name)),  
                ])
            ]),
        ])

    def _register_callbacks(self, app):
        @app.callback(
            Output("decisionpath-table-"+self.name, 'children'),
            [Input('decisionpath-table-index-'+self.name, 'value'),
             Input('decisionpath-table-highlight-'+self.name, 'value'),
             Input('pos-label', 'value')],
            [State('tabs', 'value')]
        )
        def update_decisiontree_table(index, highlight, pos_label, tab):
            if index is not None and highlight is not None:
                decisionpath_df = self.explainer.decisiontree_df_summary(highlight, index, pos_label=pos_label)
                return dbc.Table.from_dataframe(decisionpath_df)
            raise PreventUpdate


class DecisionPathGraphComponent(ExplainerComponent):
    def __init__(self, explainer, title="Decision path table",
                    header_mode="none", name=None,
                    hide_index=False, hide_highlight=False,
                    index=None, highlight=None):
        super().__init__(explainer, title, header_mode, name)
        self.hide_index, self.hide_highlight = hide_index, hide_highlight
        self.index, self.highlight = index, highlight

        self.index_name = 'decisionpath-index-'+self.name
        self.highlight_name = 'decisionpath-highlight-'+self.name

    def _layout(self):
        return html.Div([
            html.H3("Decision path:"),
            dbc.Row([
                make_hideable(
                    dbc.Col([
                        dbc.Label("Index:"),
                        dcc.Dropdown(id='decisionpath-index-'+self.name, 
                            options = [{'label': str(idx), 'value':idx} 
                                            for idx in self.explainer.idxs],
                            value=self.index)
                    ], md=4), hide=self.hide_index),
                    make_hideable(
                    dbc.Col([
                        dbc.Label("Highlight tree:"),
                        dcc.Dropdown(id='decisionpath-highlight-'+self.name, 
                            options = [{'label': str(tree), 'value': tree} 
                                            for tree in range(self.explainer.no_of_trees)],
                            value=self.highlight)
                    ], md=2), hide=self.hide_highlight), 
            ]),
            dbc.Row([
                dbc.Col([
                    dcc.Loading(id="loading-decisionpath-"+self.name, 
                        children=html.Img(id="decisionpath-svg-"+self.name)),  
                ])
            ]),
        ])

    def _register_callbacks(self, app):
        @app.callback(
            Output("decisionpath-svg-"+self.name, 'src'),
            [Input('decisionpath-index-'+self.name, 'value'),
             Input('decisionpath-highlight-'+self.name, 'value'),
             Input('pos-label', 'value')],
            [State('tabs', 'value')]
        )
        def update_tree_graph(index, highlight, pos_label, tab):
            if index is not None and highlight is not None:
                return self.explainer.decision_path_encoded(highlight, index)
            raise PreventUpdate

        

def decision_trees_layout(explainer, round=2, **kwargs):
    return dbc.Container([
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
                        
                    ], inline=True),    
                dbc.Button("Random Index", color="primary", id='tree-index-button'),
                dcc.Store(id='tree-index-store'),
                html.H4('(click on a prediction to see decision path)'),
               # dcc.Loading(id="loading-trees-graph", 
                           # children=[
                                dcc.Graph(id='tree-predictions-graph'),
                                #]),  
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
        dbc.Row([
            dbc.Col([
                 dcc.Loading(id="loading-tree-svg-graph", 
                            children=[html.Img(id='dtreeviz-svg')])
            ])
            
        ])
    ],  fluid=True)


def decision_trees_callbacks(explainer, app, round=2, **kwargs):
    @app.callback(
        [Output('tree-predictions-table', 'data'),
         Output('tree-predictions-table', 'columns')],
        [Input('tree-predictions-graph', 'clickData'),
         Input('pos-label', 'value')], # this causes issues for some reason, only on this tab??
        [State('tree-index-store', 'data'),
         State('tabs', 'value')])
    def display_tree_click_data(clickdata, pos_label, index, tab):
        if clickdata is not None and index is not None:
            tree_idx = int(clickdata['points'][0]['text'].split('tree no ')[1].split(':')[0]) if clickdata is not None else 0
            _, _, decisiontree_df = explainer.decisiontree_df_summary(tree_idx, index, round=round, pos_label=pos_label)
            columns = [{'id': c, 'name': c} for c in  decisiontree_df.columns.tolist()]
            return (decisiontree_df.to_dict('records'), columns)
        raise PreventUpdate

    @app.callback(
        Output('dtreeviz-svg', 'src'),
        [Input('tree-predictions-graph', 'clickData'),
         #Input('pos-label', 'value')#this causes issues for some reason, only on this tab??
         ],
        [State('tree-index-store', 'data'),
         State('tabs', 'value')])
    def display_click_data(clickData, index, tab):
        if clickData is not None and index is not None and explainer.graphviz_available and explainer.is_classifier:
            tree_idx = int(clickData['points'][0]['text'].split('tree no ')[1].split(':')[0]) 
            svg_encoded = explainer.decision_path_encoded(tree_idx, index)
            return svg_encoded
        return ""

    @app.callback(
        Output('tree-predictions-graph', 'figure'),
        [Input('tree-index-store', 'data'),
         Input('pos-label', 'value'),
         Input('tree-predictions-graph', 'clickData')],
        [State('tabs', 'value')]
    )
    def update_tree_graph(index, pos_label, clickdata, tab):
        if index is not None:
            highlight_tree = int(clickdata['points'][0]['text'].split('tree no ')[1].split(':')[0]) if clickdata is not None else None
            return explainer.plot_trees(index, highlight_tree=highlight_tree, round=round, pos_label=pos_label)
        return {}

    @app.callback(
        Output('tree-index-store', 'data'),
        [Input('tree-input-index', 'value')],
        [State('tabs', 'value')]
    )
    def update_tree_index_store(index, tab):
        if (explainer.idxs is None 
            and str(index).isdigit() 
            and int(index) >= 0
            and int(index) <= len(explainer)):
            return int(index)
        if (explainer.idxs is not None and index in explainer.idxs):
            return index
        return None

    @app.callback(
        Output('tree-input-index', 'value'),
        [Input('tree-index-button', 'n_clicks')],
        [State('tabs', 'value')]
    )
    def update_tree_input_index(n_clicks, tab):
        return explainer.random_index(return_str=True)