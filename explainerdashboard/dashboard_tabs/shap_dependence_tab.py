__all__ = ['ShapDependenceTab']

import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_table

from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from .dashboard_methods import *

class ShapDependenceTab:
    def __init__(self, explainer, standalone=False, tab_id="shap_dependence", title='Shap Dependence',
                 n_features=10, **kwargs):
        self.explainer = explainer
        self.standalone = standalone
        self.tab_id = tab_id
        self.title = title

        self.n_features = n_features
        self.kwargs = kwargs

        if self.standalone:
            self.label_selector = TitleAndLabelSelector(explainer, title=title)
             
    def layout(self):
        return dbc.Container([
            self.label_selector.layout() if self.standalone else None,
            # need to add dummy to make callbacks on tab change work:
            html.Div(id='tabs') if self.standalone else None, 
            shap_dependence_layout(self.explainer, n_features=self.n_features)
    
        ], fluid=True)
    
    def register_callbacks(self, app):
        if self.standalone:
            self.label_selector.register_callbacks(app)
        shap_dependence_callbacks(self.explainer, app)


def shap_dependence_layout(explainer, n_features=10, cats=True, **kwargs):

    cats_display = 'none' if explainer.cats is None else 'inline-block'
    return dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H3('Shap Summary'),
            dbc.Row([
                dbc.Col([
                    dbc.Label("Depth:"),
                    dcc.Dropdown(id='dependence-scatter-depth',
                        options = [{'label': str(i+1), 'value':i+1} 
                                        for i in range(len(explainer.columns_ranked(cats))-1)],
                        value=min(n_features, len(explainer.columns_ranked(cats))-1))],
                    width=3), 
                dbc.Col([
                    dbc.FormGroup(
                        [
                            dbc.Label("Summary Type"),
                            dbc.RadioItems(
                                options=[
                                    {"label": "Aggregate", "value": "aggregate"},
                                    {"label": "Detailed", "value": "detailed"},
                                ],
                                value="aggregate",
                                id="dependence-summary-type",
                                inline=True,
                            ),
                        ]
                    )
                ], width=3),
                dbc.Col([
                    dbc.Label("Grouping:"),
                    dbc.FormGroup(
                    [
                        dbc.RadioButton(
                            id='dependence-group-categoricals', 
                            className="form-check-input",
                            checked=True),
                        dbc.Label("Group Cats",
                                html_for='dependence-group-categoricals',
                                className="form-check-label"),
                    ], check=True)],
                    width=3),
                ], form=True, justify="between"),

            dbc.Label('(Click on a dot to display dependece graph)'),
            dcc.Loading(id="loading-dependence-shap-summary", 
                    children=[dcc.Graph(id='dependence-shap-summary-graph')])
        ], width=6),
        dbc.Col([
            html.H3('Shap Dependence Plot'),
            dbc.Row([
                dbc.Col([
                    html.Label('Plot dependence for column:'),
                    dcc.Dropdown(id='dependence-col', 
                        options=[{'label': col, 'value':col} 
                                    for col in explainer.columns_ranked(cats)],
                        value=explainer.columns_ranked(cats)[0])],
                    width=5), 
                dbc.Col([
                     html.Label('Color observation by column:'),
                    dcc.Dropdown(id='dependence-color-col', 
                        options=[{'label': col, 'value':col} 
                                    for col in explainer.columns_ranked(cats)],
                        value=explainer.columns_ranked(cats)[1])],
                    width=5), 
                dbc.Col([
                    html.Label('Highlight:'),
                    dbc.Input(id='dependence-highlight-index', 
                            placeholder="Highlight index...",
                            debounce=True)]
                    , width=2) 
                ], form=True),
            
            dcc.Loading(id="loading-dependence-graph", 
                         children=[dcc.Graph(id='dependence-graph')]),
        ], width=6),
        ]),
    ],  fluid=True)


def shap_dependence_callbacks(explainer, app, **kwargs):
    
    @app.callback(
        [Output('dependence-shap-summary-graph', 'figure'),
         Output('dependence-col', 'options'),
         Output('dependence-scatter-depth', 'options')],
        [Input('dependence-summary-type', 'value'),
         Input('dependence-group-categoricals', 'checked'),
         Input('dependence-scatter-depth', 'value'),
         Input('label-store', 'data')],
        [State('tabs', 'value')])
    def update_dependence_shap_scatter_graph(summary_type, cats, depth, pos_label, tab):
        ctx = dash.callback_context
        if ctx.triggered:
            if depth is None: depth = 10
            if summary_type=='aggregate':
                plot = explainer.plot_importances(
                        type='shap', topx=depth, cats=cats)
            elif summary_type=='detailed':
                plot = explainer.plot_shap_summary(topx=depth, cats=cats)

            trigger = ctx.triggered[0]['prop_id'].split('.')[0]

            if trigger=='dependence-group-categoricals':
                if cats:
                    col_options = [{'label':col, 'value':col} 
                                for col in explainer.columns_cats]
                else:
                    col_options = [{'label':col, 'value':col} 
                                for col in explainer.columns]

                depth_options = [{'label': str(i+1), 'value':i+1} 
                            for i in range(len(col_options))]
                return (plot, col_options, depth_options)
                
            else:
                return (plot, dash.no_update, dash.no_update)
                
        raise PreventUpdate

    @app.callback(
        [Output('dependence-highlight-index', 'value'),
         Output('dependence-col', 'value')],
        [Input('dependence-shap-summary-graph', 'clickData')],
        [State('dependence-group-categoricals', 'checked')])
    def display_scatter_click_data(clickdata, cats):
        if clickdata is not None and clickdata['points'][0] is not None:
            if isinstance(clickdata['points'][0]['y'], float): # detailed
                idx = clickdata['points'][0]['pointIndex']
                col = clickdata['points'][0]['text'].split('=')[0]                             
                return (idx, col)
            elif  isinstance(clickdata['points'][0]['y'], str): # aggregate
                col = clickdata['points'][0]['y']
                return (dash.no_update, col) 
        raise PreventUpdate

    @app.callback(
        [Output('dependence-color-col', 'options'),
         Output('dependence-color-col', 'value')],
        [Input('dependence-col', 'value')],
        [State('dependence-group-categoricals', 'checked')])
    def set_color_col_dropdown(col, cats):
        sorted_interact_cols = explainer.shap_top_interactions(col, cats=cats)
        options = [{'label': col, 'value':col} 
                                    for col in sorted_interact_cols]
        value =   sorted_interact_cols[1]                                
        return (options, value)


    @app.callback(
        Output('dependence-graph', 'figure'),
        [Input('dependence-color-col', 'value'),
         Input('dependence-highlight-index', 'value'),
         Input('label-store', 'data')],
        [State('dependence-col', 'value'),
         State('dependence-group-categoricals', 'checked')])
    def update_dependence_graph(color_col, idx, pos_label, col, cats):
        if color_col is not None:
            return explainer.plot_shap_dependence(
                        col, color_col, highlight_idx=idx, cats=cats)
        raise PreventUpdate