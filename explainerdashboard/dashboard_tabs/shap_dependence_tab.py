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
    def __init__(self, explainer, 
                    standalone=False, hide_title=False,
                    tab_id="shap_dependence", title='Shap Dependence',
                    n_features=None, **kwargs):
        self.explainer = explainer
        self.standalone = standalone
        self.tab_id = tab_id
        self.title = title

        self.n_features = n_features
        self.kwargs = kwargs

        if self.standalone:
            # If standalone then no 'pos-label-selector' or 'tabs'
            # component has been defined by overarching Dashboard.
            # The callbacks expect these to be there, so we add them in here.
            self.label_selector = TitleAndLabelSelector(
                                    explainer, title=title, 
                                    hidden=hide_title, dummy_tabs=True)
        else:
            # No need to define anything, so just add empty dummy
            self.label_selector = DummyComponent()
             
    def layout(self):
        return dbc.Container([
            self.label_selector.layout(),
            shap_dependence_layout(self.explainer, n_features=self.n_features)
    
        ], fluid=True)
    
    def register_callbacks(self, app):
        self.label_selector.register_callbacks(app)
        shap_dependence_callbacks(self.explainer, app)


def shap_dependence_layout(explainer, n_features=None, cats=True, **kwargs):
    cats_display = 'none' if explainer.cats is None else 'inline-block'
    if n_features is None:
        n_features = len(explainer.columns_ranked_by_shap(cats))
    return dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H3('Shap Summary'),
            dbc.Row([
                dbc.Col([
                    dbc.Label("Depth:"),
                    dcc.Dropdown(id='dependence-scatter-depth',
                        options=[{'label': str(i+1), 'value': i+1} 
                                        for i in range(len(explainer.columns_ranked_by_shap(cats)))],
                        value=min(n_features, len(explainer.columns_ranked_by_shap(cats)))
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
                                id="dependence-summary-type",
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
                            id='dependence-group-categoricals', 
                            className="form-check-input",
                            checked=True),
                        dbc.Label("Group Cats",
                                html_for='dependence-group-categoricals',
                                className="form-check-label"),
                    ], check=True)],
                    md=3),
                ], form=True, justify="between"),

            dbc.Label('(Click on a dot to display dependece graph)'),
            dcc.Loading(id="loading-dependence-shap-summary", 
                    children=[dcc.Graph(id='dependence-shap-summary-graph')])
        ], md=6),
        dbc.Col([
            html.H3('Shap Dependence Plot'),
            dbc.Row([
                dbc.Col([
                    html.Label('Plot dependence for column:'),
                    dcc.Dropdown(id='dependence-col', 
                        options=[{'label': col, 'value':col} 
                                    for col in explainer.columns_ranked_by_shap(cats)],
                        value=explainer.columns_ranked_by_shap(cats)[0])],
                    md=5), 
                dbc.Col([
                     html.Label('Color observation by column:'),
                    dcc.Dropdown(id='dependence-color-col', 
                        options=[{'label': col, 'value':col} 
                                    for col in explainer.columns_ranked_by_shap(cats)],
                        value=explainer.columns_ranked_by_shap(cats)[1])],
                    md=5), 
                dbc.Col([
                    html.Label('Highlight:'),
                    dbc.Input(id='dependence-highlight-index', 
                            placeholder="Highlight index...",
                            debounce=True)]
                    , md=2) 
                ], form=True),
            
            dcc.Loading(id="loading-dependence-graph", 
                         children=[dcc.Graph(id='dependence-graph')]),
        ], md=6),
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
        if summary_type == 'aggregate':
            plot = explainer.plot_importances(
                    kind='shap', topx=depth, cats=cats, pos_label=pos_label)
        elif summary_type == 'detailed':
            plot = explainer.plot_shap_summary(
                    topx=depth, cats=cats, pos_label=pos_label)
        ctx = dash.callback_context
        trigger = ctx.triggered[0]['prop_id'].split('.')[0]
        if trigger == 'dependence-group-categoricals':
            # if change to group cats, adjust columns and depth
            if cats:
                col_options = [{'label': col, 'value': col} 
                                    for col in explainer.columns_cats]
            else:
                col_options = [{'label': col, 'value': col} 
                                    for col in explainer.columns]

            depth_options = [{'label': str(i+1), 'value': i+1} 
                                    for i in range(len(col_options))]
            return (plot, col_options, depth_options)
        else:
            return (plot, dash.no_update, dash.no_update)

    @app.callback(
        [Output('dependence-highlight-index', 'value'),
         Output('dependence-col', 'value')],
        [Input('dependence-shap-summary-graph', 'clickData')],
        [State('dependence-group-categoricals', 'checked'),
         State('label-store', 'data')])
    def display_scatter_click_data(clickdata, cats, pos_label):
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

    @app.callback(
        [Output('dependence-color-col', 'options'),
         Output('dependence-color-col', 'value')],
        [Input('dependence-col', 'value')],
        [State('dependence-group-categoricals', 'checked'),
         State('label-store', 'data')])
    def set_color_col_dropdown(col, cats, pos_label):
        sorted_interact_cols = explainer.shap_top_interactions(col, cats=cats, pos_label=pos_label)
        options = [{'label': col, 'value':col} 
                                    for col in sorted_interact_cols]
        value = sorted_interact_cols[1]                                
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
                        col, color_col, highlight_idx=idx, pos_label=pos_label)
        raise PreventUpdate