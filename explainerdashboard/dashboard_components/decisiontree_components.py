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

from ..dashboard_methods import *
from .connectors import ClassifierRandomIndexComponent, IndexConnector, HighlightConnector


class DecisionTreesComponent(ExplainerComponent):
    def __init__(self, explainer, title="Decision Trees", name=None,
                    hide_title=False, hide_index=False, hide_highlight=False,
                    hide_selector=False,
                    pos_label=None, index=None, highlight=None):
        """Show prediction from individual decision trees inside RandomForest component

        Args:
            explainer (Explainer): explainer object constructed with either
                        ClassifierExplainer() or RegressionExplainer()
            title (str, optional): Title of tab or page. Defaults to 
                        "Decision Trees".
            name (str, optional): unique name to add to Component elements. 
                        If None then random uuid is generated to make sure 
                        it's unique. Defaults to None.
            hide_title (bool, optional): hide title, Defaults to False.
            hide_index (bool, optional): Hide index selector. Defaults to False.
            hide_highlight (bool, optional): Hide tree highlight selector. Defaults to False.
            hide_selector (bool, optional): hide pos label selectors. Defaults to False.
            pos_label ({int, str}, optional): initial pos label. 
                        Defaults to explainer.pos_label
            index ({str, int}, optional): Initial index to display. Defaults to None.
            highlight ([type], optional): Initial tree to highlight. Defaults to None.
        """
        super().__init__(explainer, title, name)

        self.index_name = 'decisiontrees-index-'+self.name
        self.highlight_name = 'decisiontrees-highlight-'+self.name

        self.selector = PosLabelSelector(explainer, name=self.name, pos_label=pos_label)
        self.register_dependencies("preds", "pred_probas")

    def layout(self):
        return html.Div([
            make_hideable(
                html.H3("Decision trees:"), hide=self.hide_title),
            dbc.Row([
                make_hideable(
                    dbc.Col([
                        dbc.Label(f"{self.explainer.index_name}:"),
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
                make_hideable(
                        dbc.Col([self.selector.layout()
                    ], width=2), hide=self.hide_selector)
            ]),
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id="decisiontrees-graph-"+self.name,
                                config=dict(modeBarButtons=[['toImage']], displaylogo=False)),  
                ])
            ]),
        ])

    def _register_callbacks(self, app):
        @app.callback(
            Output("decisiontrees-graph-"+self.name, 'figure'),
            [Input('decisiontrees-index-'+self.name, 'value'),
             Input('decisiontrees-highlight-'+self.name, 'value'),
             Input('pos-label-'+self.name, 'value')],
        )
        def update_tree_graph(index, highlight, pos_label):
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
    def __init__(self, explainer, title="Decision path table", name=None,
                    hide_title=False, hide_index=False, hide_highlight=False,
                    hide_selector=False,
                    pos_label=None, index=None, highlight=None):
        """Display a table of the decision path through a particular decision tree

        Args:
            explainer (Explainer): explainer object constructed with either
                        ClassifierExplainer() or RegressionExplainer()
            title (str, optional): Title of tab or page. Defaults to 
                        "Decision path table".
            name (str, optional): unique name to add to Component elements. 
                        If None then random uuid is generated to make sure 
                        it's unique. Defaults to None.
            hide_title (bool, optional): hide title, Defaults to False.
            hide_index (bool, optional): Hide index selector. 
                        Defaults to False.
            hide_highlight (bool, optional): Hide tree index selector. 
                        Defaults to False.
            hide_selector (bool, optional): hide pos label selectors. 
                        Defaults to False.
            pos_label ({int, str}, optional): initial pos label. 
                        Defaults to explainer.pos_label
            index ({str, int}, optional): Initial index to display decision 
                        path for. Defaults to None.
            highlight (int, optional): Initial tree idx to display decision 
                        path for. Defaults to None.
        """
        super().__init__(explainer, title, name)

        self.index_name = 'decisionpath-table-index-'+self.name
        self.highlight_name = 'decisionpath-table-highlight-'+self.name

        self.selector = PosLabelSelector(explainer, name=self.name, pos_label=pos_label)
        self.register_dependencies("decision_trees")

    def layout(self):
        return html.Div([
            make_hideable(
                html.H3("Decision path:"), hide=self.hide_title),
            dbc.Row([
                make_hideable(
                    dbc.Col([
                        dbc.Label(f"{self.explainer.index_name}:"),
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
                    make_hideable(
                        dbc.Col([self.selector.layout()
                    ], width=2), hide=self.hide_selector)
            ]),
            dbc.Row([
                dbc.Col([
                    html.Div(id="decisionpath-table-"+self.name),  
                ])
            ]),
        ])

    def _register_callbacks(self, app):
        @app.callback(
            Output("decisionpath-table-"+self.name, 'children'),
            [Input('decisionpath-table-index-'+self.name, 'value'),
             Input('decisionpath-table-highlight-'+self.name, 'value'),
             Input('pos-label-'+self.name, 'value')],
        )
        def update_decisiontree_table(index, highlight, pos_label):
            if index is not None and highlight is not None:
                decisionpath_df = self.explainer.decisiontree_summary_df(highlight, index, pos_label=pos_label)
                return dbc.Table.from_dataframe(decisionpath_df)
            raise PreventUpdate


class DecisionPathGraphComponent(ExplainerComponent):
    def __init__(self, explainer, title="Decision path graph", name=None,
                    hide_title=False, hide_index=False, 
                    hide_highlight=False, hide_button=False,
                    hide_selector=False,
                    pos_label=None, index=None, highlight=None):
        """Display dtreeviz decision path

        Args:
            explainer (Explainer): explainer object constructed with either
                        ClassifierExplainer() or RegressionExplainer()
            title (str, optional): Title of tab or page. Defaults to 
                        "Decision path graph".
            name (str, optional): unique name to add to Component elements. 
                        If None then random uuid is generated to make sure 
                        it's unique. Defaults to None.
            hide_title (bool, optional): hide title
            hide_index (bool, optional): hide index selector. Defaults to False.
            hide_highlight (bool, optional): hide tree idx selector. Defaults to False.
            hide_button (bool, optional): hide the button, Defaults to False.
            hide_selector (bool, optional): hide pos label selectors. Defaults to False.
            pos_label ({int, str}, optional): initial pos label. 
                        Defaults to explainer.pos_label
            index ({str, int}, optional): Initial index to display. Defaults to None.
            highlight ([type], optional): Initial tree idx to display. Defaults to None.
        """
        super().__init__(explainer, title, name)
        # if explainer.is_regression:
        #     raise ValueError("DecisionPathGraphComponent only available for classifiers for now!")

        self.index_name = 'decisionpath-index-'+self.name
        self.highlight_name = 'decisionpath-highlight-'+self.name

        self.selector = PosLabelSelector(explainer, name=self.name, pos_label=pos_label)

    def layout(self):
        return html.Div([
            make_hideable(
                html.H3("Decision Tree Graph:"), hide=self.hide_title),
            dbc.Row([
                make_hideable(
                    dbc.Col([
                        dbc.Label(f"{self.explainer.index_name}:"),
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
                    make_hideable(
                        dbc.Col([self.selector.layout()
                    ], width=2), hide=self.hide_selector),
                    make_hideable(
                    dbc.Col([
                        dbc.Button("Generate Tree Graph", color="primary", 
                                    id='decisionpath-button-'+self.name)
                    ], md=2, align="end"), hide=self.hide_button),           
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
            [Input('decisionpath-button-'+self.name, 'n_clicks')],
            [State('decisionpath-index-'+self.name, 'value'),
             State('decisionpath-highlight-'+self.name, 'value'),
             State('pos-label-'+self.name, 'value')]
        )
        def update_tree_graph(n_clicks, index, highlight, pos_label):
            if n_clicks is not None and index is not None and highlight is not None:
                return self.explainer.decision_path_encoded(highlight, index)
            raise PreventUpdate