__all__ = ['ShapDependenceTab',
            'ShapSummaryComponent',
            'ShapDependenceComponent']

import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_table

from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from .dashboard_methods import *

class ShapDependenceTab:
    def __init__(self, explainer, 
                    tab_id="shap_dependence", title='Shap Dependence',
                    header_mode="none",
                    n_features=None, cats=True, **kwargs):
        self.explainer = explainer
        self.tab_id = tab_id
        self.title = title

        self.n_features = n_features
        self.kwargs = kwargs
        self.header = ExplainerHeader(explainer, mode=header_mode)

        self.shap_summary = ShapSummaryComponent(self.explainer)
        self.shap_dependence = ShapDependenceComponent(
                                    self.explainer, hide_cats_toggle=True)
        self.connector = ShapSummaryDependenceConnector(self.explainer)
             
    def layout(self):
        return dbc.Container([
            self.header.layout(),
            dbc.Row([
                dbc.Col([
                    self.shap_summary.layout()
                ], md=6),
                dbc.Col([
                    self.shap_dependence.layout()
                ], md=6),
                ]),
            ],  fluid=True)

    def register_callbacks(self, app):
        self.shap_summary.register_callbacks(app)
        self.shap_dependence.register_callbacks(app)
        self.connector.register_callbacks(app)


class ShapSummaryComponent():
    def __init__(self, explainer,  title='Shap Dependence Summary',
                    header_mode="none",
                    n_features=None, cats=True, **kwargs):
        self.explainer = explainer
        self.title = title

        self.n_features = n_features
        self.kwargs = kwargs
        self.cats = cats
        
        if self.n_features is None:
            self.n_features = len(self.explainer.columns_ranked_by_shap(cats))

        self.header = ExplainerHeader(explainer, mode=header_mode)
             
    def layout(self):
        cats_display = 'none' if self.explainer.cats is None else 'inline-block'
        return dbc.Container([
            self.header.layout(),
            html.H3('Shap Summary'),
            dbc.Row([
                dbc.Col([
                    dbc.Label("Depth:"),
                    dcc.Dropdown(id='shap-summary-depth',
                        options=[{'label': str(i+1), 'value': i+1} 
                                        for i in range(len(self.explainer.columns_ranked_by_shap(self.cats)))],
                        value=min(self.n_features, len(self.explainer.columns_ranked_by_shap(self.cats)))
                        )],
                    md=3), 
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
                                id="shap-summary-type",
                                inline=True,
                            ),
                        ]
                    )
                ], md=3),
                dbc.Col([
                    dbc.Label("Grouping:"),
                    dbc.FormGroup(
                    [
                        dbc.RadioButton(
                            id='shap-summary-group-cats', 
                            className="form-check-input",
                            checked=True),
                        dbc.Label("Group Cats",
                                html_for='shap-summary-group-cats',
                                className="form-check-label"),
                    ], check=True)],
                    md=3),
                ], form=True, justify="between"),

            dbc.Label('(Click on a dot to display dependece graph)'),
            dcc.Loading(id="loading-dependence-shap-summary", 
                    children=[dcc.Graph(id='shap-summary-graph')])
            
        ], fluid=True)
    
    def register_callbacks(self, app):
        @app.callback(
            [Output('shap-summary-graph', 'figure'),
             Output('shap-summary-depth', 'options')],
            [Input('shap-summary-type', 'value'),
             Input('shap-summary-group-cats', 'checked'),
             Input('shap-summary-depth', 'value'),
             Input('pos-label', 'value')],
            [State('tabs', 'value')])
        def update_shap_summary_graph(summary_type, cats, depth, pos_label, tab):
            if summary_type == 'aggregate':
                plot = self.explainer.plot_importances(
                        kind='shap', topx=depth, cats=cats, pos_label=pos_label)
            elif summary_type == 'detailed':
                plot = self.explainer.plot_shap_summary(
                        topx=depth, cats=cats, pos_label=pos_label)
            ctx = dash.callback_context
            trigger = ctx.triggered[0]['prop_id'].split('.')[0]
            if trigger == 'shap-summary-group-cats':
                depth_options = [{'label': str(i+1), 'value': i+1} 
                                        for i in range(len(self.explainer.columns_ranked_by_shap(cats)))]
                return (plot, depth_options)
            else:
                return (plot, dash.no_update)


class ShapDependenceComponent():
    def __init__(self, explainer, title='Shap Dependence Summary',
                    header_mode="none", hide_cats_toggle=False,
                    cats=True, **kwargs):
        self.explainer = explainer
        self.hide_cats_toggle = hide_cats_toggle
        self.title = title

        self.cats = cats
        self.kwargs = kwargs

        self.header = ExplainerHeader(explainer, title=title, mode=header_mode)
             
    def layout(self):
        if self.hide_cats_toggle:
            cats_toggle_group = html.Div([
                dbc.RadioButton(
                            id='shap-dependence-group-cats', 
                            className="form-check-input",
                            checked=self.cats)
            ], style=dict(display="none"))
        else:
            cats_toggle_group = dbc.Col([
                    dbc.Label("Grouping:"),
                    dbc.FormGroup(
                    [
                        dbc.RadioButton(
                            id='shap-dependence-group-cats', 
                            className="form-check-input",
                            checked=self.cats),
                        dbc.Label("Group Cats",
                                html_for='shap-dependence-group-cats',
                                className="form-check-label"),
                    ], check=True)],
                    md=2)
            

        return dbc.Container([
            self.header.layout(),
            html.H3('Shap Dependence Plot'),
            dbc.Row([
                cats_toggle_group,
                dbc.Col([
                    html.Label('Plot dependence for column:'),
                    dcc.Dropdown(id='shap-dependence-col', 
                        options=[{'label': col, 'value':col} 
                                    for col in self.explainer.columns_ranked_by_shap(self.cats)],
                        value=self.explainer.columns_ranked_by_shap(self.cats)[0])],
                    md=4), 
                dbc.Col([
                     html.Label('Color observation by column:'),
                    dcc.Dropdown(id='shap-dependence-color-col', 
                        options=[{'label': col, 'value':col} 
                                    for col in self.explainer.columns_ranked_by_shap(self.cats)],
                        value=self.explainer.columns_ranked_by_shap(self.cats)[1])],
                    md=4), 
                dbc.Col([
                    html.Label('Highlight:'),
                    dbc.Input(id='shap-dependence-highlight-index', 
                            placeholder="Highlight index...",
                            debounce=True)]
                    , md=2) 
                ], form=True),
            
            dcc.Loading(id="loading-dependence-graph", 
                         children=[dcc.Graph(id='shap-dependence-graph')]),
        ], fluid=True)

    def register_callbacks(self, app):
        @app.callback(
            [Output('shap-dependence-color-col', 'options'),
             Output('shap-dependence-color-col', 'value')],
            [Input('shap-dependence-col', 'value')],
            [State('shap-dependence-group-cats', 'checked'),
            State('pos-label', 'value')])
        def set_color_col_dropdown(col, cats, pos_label):
            sorted_interact_cols = self.explainer.shap_top_interactions(col, cats=cats, pos_label=pos_label)
            options = [{'label': col, 'value':col} 
                                        for col in sorted_interact_cols]
            value = sorted_interact_cols[1]                                
            return (options, value)

        @app.callback(
            Output('shap-dependence-graph', 'figure'),
            [Input('shap-dependence-color-col', 'value'),
            Input('shap-dependence-highlight-index', 'value'),
            Input('pos-label', 'value')],
            [State('shap-dependence-col', 'value')])
        def update_dependence_graph(color_col, idx, pos_label, col):
            if color_col is not None:
                return self.explainer.plot_shap_dependence(
                            col, color_col, highlight_idx=idx, pos_label=pos_label)
            raise PreventUpdate

        @app.callback(
            Output('shap-dependence-col', 'options'),
            [Input('shap-dependence-group-cats', 'checked')],
            [State('tabs', 'value')])
        def update_dependence_shap_scatter_graph(cats,  tab):
            return [{'label': col, 'value': col} 
                                    for col in self.explainer.columns_ranked_by_shap(cats)]

class ShapSummaryDependenceConnector:
    def __init__(self, explainer):
        self.explainer = explainer

    def register_callbacks(self, app):
        @app.callback(
            Output('shap-dependence-group-cats', 'checked'),
            [Input('shap-summary-group-cats', 'checked')],
            [State('tabs', 'value')])
        def update_dependence_shap_scatter_graph(cats,  tab):
            return cats

        @app.callback(
            [Output('shap-dependence-highlight-index', 'value'),
             Output('shap-dependence-col', 'value')],
            [Input('shap-summary-graph', 'clickData')])
        def display_scatter_click_data(clickdata):
            if clickdata is not None and clickdata['points'][0] is not None:
                if isinstance(clickdata['points'][0]['y'], float): # detailed
                    # if detailed, clickdata returns scatter marker location -> type==float
                    idx = clickdata['points'][0]['pointIndex']
                    col = clickdata['points'][0]['text'].split('=')[0]                             
                    return (idx, col)
                elif isinstance(clickdata['points'][0]['y'], str): # aggregate
                    # in aggregate clickdata returns col name -> type==str
                    col = clickdata['points'][0]['y'].split(' ')[1]
                    return (dash.no_update, col)
            raise PreventUpdate