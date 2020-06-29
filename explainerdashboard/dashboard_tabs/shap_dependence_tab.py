__all__ = ['ShapDependenceTab',
            'ShapSummaryComponent',
            'ShapDependenceComponent',
            'ShapSummaryDependenceConnector']

import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_table

from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from .dashboard_methods import *

class ShapDependenceTab(ExplainerComponent):
    def __init__(self, explainer, title='Shap Dependence',
                    header_mode="none", name=None,
                    tab_id="shap_dependence", 
                    depth=None, cats=True):
        super().__init__(explainer, title, header_mode, name)
        self.tab_id = tab_id
        
        self.depth = depth
        self.cats = cats

        self.shap_summary = ShapSummaryComponent(self.explainer)
        self.shap_dependence = ShapDependenceComponent(
                                    self.explainer, hide_cats=True)
        self.connector = ShapSummaryDependenceConnector(self.shap_summary, self.shap_dependence)
        self.register_components(self.shap_summary, self.shap_dependence, self.connector)

    def _layout(self):
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    self.shap_summary.layout()
                ], md=6),
                dbc.Col([
                    self.shap_dependence.layout()
                ], md=6),
                ]),
            ],  fluid=True)

class ShapSummaryComponent(ExplainerComponent):
    def __init__(self, explainer, title='Shap Dependence Summary',
                    header_mode="none", name=None,
                    hide_depth=False, hide_type=False, hide_cats=False,
                    depth=None, summary_type="aggregate", cats=True):
        super().__init__(explainer, title, header_mode, name)

        self.hide_depth, self.hide_type, self.hide_cats = hide_depth, hide_type, hide_cats
        self.depth, self.summary_type, self.cats = depth, summary_type, cats

        if self.explainer.cats is None or not self.explainer.cats:
            self.hide_cats = True
        
        if self.depth is not None:
            self.depth = min(self.depth, len(self.explainer.columns_ranked_by_shap(cats)))

        self.register_dependencies('shap_values', 'shap_values_cats')
             
    def _layout(self):
        return dbc.Container([
            html.H3('Shap Summary'),
            dbc.Row([
                make_hideable(
                    dbc.Col([
                        dbc.Label("Depth:"),
                        dcc.Dropdown(id='shap-summary-depth-'+self.name,
                            options=[{'label': str(i+1), 'value': i+1} for i in 
                                        range(len(self.explainer.columns_ranked_by_shap(self.cats)))],
                            value=self.depth)
                    ], md=3), self.hide_depth),
                make_hideable(
                    dbc.Col([
                        dbc.FormGroup(
                            [
                                dbc.Label("Summary Type"),
                                dbc.RadioItems(
                                    options=[
                                        {"label": "Aggregate", "value": "aggregate"},
                                        {"label": "Detailed", "value": "detailed"},
                                    ],
                                    value=self.summary_type,
                                    id="shap-summary-type-"+self.name,
                                    inline=True,
                                ),
                            ]
                        )
                    ]), self.hide_type),
                make_hideable(
                    dbc.Col([
                        dbc.Label("Grouping:"),
                        dbc.FormGroup(
                        [
                            dbc.RadioButton(
                                id='shap-summary-group-cats-'+self.name, 
                                className="form-check-input",
                                checked=self.cats),
                            dbc.Label("Group Cats",
                                    html_for='shap-summary-group-'+self.name, 
                                    className="form-check-label"),
                        ], check=True)
                    ], md=3), self.hide_cats),
                ], form=True),

            dcc.Loading(id="loading-dependence-shap-summary-"+self.name, 
                    children=[dcc.Graph(id="shap-summary-graph-"+self.name)])
            
        ], fluid=True)
    
    def _register_callbacks(self, app):
        @app.callback(
            [Output('shap-summary-graph-'+self.name, 'figure'),
             Output('shap-summary-depth-'+self.name, 'options')],
            [Input('shap-summary-type-'+self.name, 'value'),
             Input('shap-summary-group-cats-'+self.name, 'checked'),
             Input('shap-summary-depth-'+self.name, 'value'),
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
            if trigger == 'shap-summary-group-cats-'+self.name:
                depth_options = [{'label': str(i+1), 'value': i+1} 
                                        for i in range(len(self.explainer.columns_ranked_by_shap(cats)))]
                print(depth_options)
                return (plot, depth_options)
            else:
                return (plot, dash.no_update)


class ShapDependenceComponent(ExplainerComponent):
    def __init__(self, explainer, title='Shap Dependence Summary',
                    header_mode="none", name=None,
                    hide_cats=False, hide_col=False, hide_color_col=False, hide_highlight=False,
                    cats=True, col=None, color_col=None, highlight=None):
        super().__init__(explainer, title, header_mode, name)

        self.hide_cats, self.hide_col = hide_cats, hide_col
        self.hide_color_col, self.hide_highlight = hide_color_col, hide_highlight

        self.cats = cats
        self.col, self.color_col, self.highlight = col, color_col, highlight
        if self.col is None:
            self.col = self.explainer.columns_ranked_by_shap(self.cats)[0]
        if self.color_col is None:
            self.color_col = self.explainer.shap_top_interactions(self.col, cats=cats)[1]
        self.register_dependencies('shap_values', 'shap_values_cats')
             
    def _layout(self):
        return dbc.Container([
            html.H3('Shap Dependence Plot'),
            dbc.Row([
                    make_hideable(
                        dbc.Col([
                            dbc.Label("Grouping:"),
                            dbc.FormGroup(
                            [
                                dbc.RadioButton(
                                    id='shap-dependence-group-cats-'+self.name, 
                                    className="form-check-input",
                                    checked=self.cats),
                                dbc.Label("Group Cats",
                                        html_for='shap-dependence-group-cats-'+self.name,
                                        className="form-check-label"),
                            ], check=True),
                        ],  md=2), self.hide_cats),
                    make_hideable(
                        dbc.Col([
                            dbc.Label('Feature:'),
                            dcc.Dropdown(id='shap-dependence-col-'+self.name, 
                                options=[{'label': col, 'value':col} 
                                            for col in self.explainer.columns_ranked_by_shap(self.cats)],
                                value=self.col)
                        ], md=4), self.hide_col),
                    make_hideable(dbc.Col([
                            html.Label('Color feature:'),
                            dcc.Dropdown(id='shap-dependence-color-col-'+self.name, 
                                options=[{'label': col, 'value':col} 
                                            for col in self.explainer.columns_ranked_by_shap(self.cats)],
                                value=self.color_col),   
                    ], md=4), self.hide_color_col),
                    make_hideable(
                        dbc.Col([
                                html.Label('Highlight:'),
                                dbc.Input(id='shap-dependence-highlight-index-'+self.name, 
                                        placeholder="Highlight index...",
                                        debounce=True,
                                        value=self.highlight)
                        ], md=2), self.hide_highlight),
            ], form=True),
            dcc.Loading(id="loading-dependence-graph-"+self.name, 
                         children=[dcc.Graph(id='shap-dependence-graph-'+self.name)]),
        ], fluid=False)

    def _register_callbacks(self, app):
        @app.callback(
            [Output('shap-dependence-color-col-'+self.name, 'options'),
             Output('shap-dependence-color-col-'+self.name, 'value')],
            [Input('shap-dependence-col-'+self.name, 'value')],
            [State('shap-dependence-group-cats-'+self.name, 'checked'),
             State('pos-label', 'value')])
        def set_color_col_dropdown(col, cats, pos_label):
            sorted_interact_cols = self.explainer.shap_top_interactions(col, cats=cats, pos_label=pos_label)
            options = [{'label': col, 'value':col} 
                                        for col in sorted_interact_cols]
            value = sorted_interact_cols[1]                                
            return (options, value)

        @app.callback(
            Output('shap-dependence-graph-'+self.name, 'figure'),
            [Input('shap-dependence-color-col-'+self.name, 'value'),
             Input('shap-dependence-highlight-index-'+self.name, 'value'),
             Input('pos-label', 'value')],
            [State('shap-dependence-col-'+self.name, 'value')])
        def update_dependence_graph(color_col, idx, pos_label, col):
            if color_col is not None:
                return self.explainer.plot_shap_dependence(
                            col, color_col, highlight_idx=idx, pos_label=pos_label)
            raise PreventUpdate

        @app.callback(
            Output('shap-dependence-col-'+self.name, 'options'),
            [Input('shap-dependence-group-cats-'+self.name, 'checked')],
            [State('tabs', 'value')])
        def update_dependence_shap_scatter_graph(cats,  tab):
            return [{'label': col, 'value': col} 
                                    for col in self.explainer.columns_ranked_by_shap(cats)]

class ShapSummaryDependenceConnector(ExplainerComponent):
    def __init__(self, shap_summary_component, shap_dependence_component):
        self.sum_name = shap_summary_component.name
        self.dep_name = shap_dependence_component.name

    def register_callbacks(self, app):
        @app.callback(
            Output('shap-dependence-group-cats-'+self.dep_name, 'checked'),
            [Input('shap-summary-group-cats-'+self.sum_name, 'checked')],
            [State('tabs', 'value')])
        def update_dependence_shap_scatter_graph(cats,  tab):
            return cats

        @app.callback(
            [Output('shap-dependence-highlight-index-'+self.dep_name, 'value'),
             Output('shap-dependence-col-'+self.dep_name, 'value')],
            [Input('shap-summary-graph-'+self.sum_name, 'clickData')])
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