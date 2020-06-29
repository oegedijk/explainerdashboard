__all__ = ['ShapInteractionsTab',
            'InteractionSummaryComponent',
            'InteractionDependenceComponent']

import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_table

from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from .dashboard_methods import *


class ShapInteractionsTab(ExplainerComponent):
    def __init__(self, explainer, title='Shap Interactions',
                    header_mode="none", name=None,
                    tab_id="shap_interactions", 
                    depth=None, cats=True):
        super().__init__(explainer, title, header_mode, name)
        self.explainer = explainer
        self.tab_id = tab_id

        self.depth = depth
        self.cats = cats

        self.interaction_summary = InteractionSummaryComponent(explainer)
        self.interaction_dependence = InteractionDependenceComponent(explainer)
        self.connector = InteractionSummaryDependenceConnector(
            self.interaction_summary, self.interaction_dependence)
        self.register_components(
            self.interaction_summary, self.interaction_dependence, self.connector)
        
    def _layout(self):
        return dbc.Container([
            dbc.Col([
                self.interaction_summary.layout()
            ], width=6),
            dbc.Col([
                self.interaction_dependence.layout()
            ], width=6),
        ], fluid=True)
    

class InteractionSummaryComponent(ExplainerComponent):
    def __init__(self, explainer, title="Interactions Summary",
                    header_mode="none", name=None,
                    hide_col=False, hide_depth=False, hide_type=False, hide_cats=False,
                    col=None, depth=None, summary_type="aggregate", cats=True):
        super().__init__(explainer, title, header_mode, name)

        self.hide_col, self.hide_depth, self.hide_type, self.hide_cats = \
            hide_col, hide_depth, hide_type, hide_cats
        self.col, self.depth, self.summary_type, self.cats = \
            col, depth, summary_type, cats
    
        if self.col is None:
            self.col = self.explainer.columns_ranked_by_shap(self.cats)[0]
        if self.depth is not None:
            self.depth = min(self.depth, len(self.explainer.columns_ranked_by_shap(self.cats)-1))

    def _layout(self):
        return html.Div([
            html.H3('Shap Interaction Summary'),
            dbc.Row([
                make_hideable(
                    dbc.Col([
                        dbc.Label("Feature"),
                        dcc.Dropdown(id='interaction-summary-col-'+self.name, 
                            options=[{'label': col, 'value': col} 
                                        for col in self.explainer.columns_ranked_by_shap(self.cats)],
                            value=self.col),
                    ], md=4), self.hide_col),
                make_hideable(
                    dbc.Col([
    
                        dbc.Label("Depth:"),
                        dcc.Dropdown(id='interaction-summary-depth-'+self.name, 
                            options = [{'label': str(i+1), 'value':i+1} 
                                            for i in range(len(self.explainer.columns_ranked_by_shap(self.cats)-1))],
                            value=self.depth)
                    ], md=2), self.hide_depth),
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
                                    id='interaction-summary-type-'+self.name, 
                                    inline=True,
                                ),
                            ]
                        )
                    ], md=3), self.hide_type),
                make_hideable(
                    dbc.Col([
                        dbc.Label("Grouping:"),
                        dbc.FormGroup(
                        [
                            dbc.RadioButton(
                                id='interaction-summary-group-cats-'+self.name, 
                                className="form-check-input",
                                checked=self.cats),
                            dbc.Label("Group Cats",
                                    html_for='interaction-summary-group-cats-'+self.name, 
                                    className="form-check-label"),
                        ], check=True)
                    ],md=3), self.hide_cats),
                ], form=True),
            dcc.Loading(id='loading-interaction-summary-graph-'+self.name, 
                         children=[dcc.Graph(id='interaction-summary-graph-'+self.name, )])
        ])

    def _register_callbacks(self, app):
        @app.callback(
            [Output('interaction-summary-graph-'+self.name, 'figure'),
             Input('interaction-summary-depth-'+self.name, 'options')],
            [Input('interaction-summary-col-'+self.name, 'value'),
             Input('interaction-summary-depth-'+self.name, 'value'),
             Input('interaction-summary-type-'+self.name, 'value'),
             Input('interaction-summary-group-cats-'+self.name, 'checked'),
             Input('pos-label', 'value')])
        def update_interaction_scatter_graph(col, depth, summary_type, cats, pos_label):
            if col is not None:
                if summary_type=='aggregate':
                    plot = self.explainer.plot_interactions(col, topx=depth, cats=cats, pos_label=pos_label)
                elif summary_type=='detailed':
                    plot = self.explainer.plot_shap_interaction_summary(col, topx=depth, cats=cats, pos_label=pos_label)
                ctx = dash.callback_context
                trigger = ctx.triggered[0]['prop_id'].split('.')[0]
                if trigger == 'interaction-summary-group-cats-'+self.name:
                    depth_options = [{'label': str(i+1), 'value': i+1} 
                                            for i in range(len(self.explainer.columns_ranked_by_shap(cats)))]
                    return (plot, depth_options)
                else:
                    return (plot, dash.no_update)
            raise PreventUpdate
        

class InteractionDependenceComponent(ExplainerComponent):
    def __init__(self, explainer, title="Interaction Dependence",
                    header_mode="none", name=None,
                    hide_cats=False, hide_col=False, hide_interact_col=False, hide_highlight=False,
                    hide_top=False, hide_bottom=False,
                    cats=True, col=None, interact_col=None, highlight=None):
        super().__init__(explainer, title, header_mode, name)

        self.hide_cats, self.hide_col, self.hide_interact_col, self.hide_highlight = \
            hide_cats, hide_col, hide_interact_col, hide_highlight
        self.hide_top, self.hide_bottom = hide_top, hide_bottom

        self.cats, self.col, self.interact_col, self.highlight = \
            cats, col, interact_col, highlight

        if self.col is None:
            self.col = explainer.columns_ranked_by_shap(cats)[0]
        if self.interact_col is None:
            self.interact_col = explainer.shap_top_interactions(self.col, cats=cats)[1]
        
        self.register_dependencies("shap_interaction_values", "shap_interaction_values_cats")

    def _layout(self):
        return html.Div([
            html.H3('Shap Interaction Plots'),
            dbc.Row([
                make_hideable(
                    dbc.Col([
                        dbc.Label("Grouping:"),
                        dbc.FormGroup(
                        [
                            dbc.RadioButton(
                                id='interaction-dependence-group-cats-'+self.name, 
                                className="form-check-input",
                                checked=self.cats),
                            dbc.Label("Group Cats",
                                    html_for='interaction-dependence-group-cats-'+self.name, 
                                    className="form-check-label"),
                        ], check=True)
                    ], md=2), hide=self.hide_cats),
                make_hideable(
                    dbc.Col([
                        dbc.Label("Feature"),
                        dcc.Dropdown(id='interaction-dependence-col-'+self.name,
                            options=[{'label': col, 'value':col} 
                                        for col in self.explainer.columns_ranked_by_shap(self.cats)],
                            value=self.col
                        ),
                    ], md=4), hide=self.hide_col), 
                make_hideable(
                    dbc.Col([
                        dbc.Label("Interaction Feature:"),
                        dcc.Dropdown(id='interaction-dependence-interact-col-'+self.name, 
                            options=[{'label': col, 'value':col} 
                                        for col in self.explainer.columns_ranked_by_shap(self.cats)],
                            value=self.interact_col
                        ),
                    ], md=4), hide=self.hide_interact_col), 
                make_hideable(
                    dbc.Col([
                        dbc.Label("Highlight index:"),
                        dbc.Input(id='interaction-dependence-highlight-index-'+self.name, 
                            placeholder="Highlight index...", debounce=True)],
                        md=2), hide=self.hide_highlight), 
                ], form=True),
            make_hideable(
                dcc.Loading(id='loading-interaction-dependence-graph-'+self.name, 
                         children=[dcc.Graph(id='interaction-dependence-graph-'+self.name)]),
                         hide=self.hide_top),
            make_hideable(
                dcc.Loading(id='loading-reverse-interaction-graph-'+self.name, 
                         children=[dcc.Graph(id='interaction-dependence-reverse-graph-'+self.name)]),
                         hide=self.hide_bottom),
        ])

    def _register_callbacks(self, app):
        @app.callback(
            [Output('interaction-dependence-graph-'+self.name, 'figure'),
            Output('interaction-dependence-reverse-graph-'+self.name, 'figure')],
            [Input('interaction-dependence-interact-col-'+self.name, 'value'),
             Input('interaction-dependence-highlight-index-'+self.name, 'value'),
             Input('pos-label', 'value')],
            [State('interaction-dependence-col-'+self.name, 'value'),
            State('interaction-dependence-group-cats-'+self.name, 'checked')])
        def update_dependence_graph(interact_col, index, pos_label, col, cats):
            if col is not None and interact_col is not None:
                return (self.explainer.plot_shap_interaction(
                            col, interact_col, highlight_idx=index, pos_label=pos_label),
                        self.explainer.plot_shap_interaction(
                            interact_col, col, highlight_idx=index, pos_label=pos_label))
            raise PreventUpdate

        @app.callback(
            Output('interaction-dependence-interact-col-'+self.name, 'options'),
            [Input('interaction-dependence-col-'+self.name, 'value'),
             Input('interaction-dependence-group-cats-'+self.name, 'checked'),
             Input('pos-label', 'value')],
            [State('interaction-dependence-interact-col-'+self.name, 'value')])
        def update_interaction_dependence_interact_col(col, cats, pos_label, old_interact_col):
            if col is not None:
                new_interact_cols = self.explainer.shap_top_interactions(col, cats=cats, pos_label=pos_label)
                new_interact_options = [{'label': col, 'value':col} for col in new_interact_cols]
                return new_interact_options
            raise PreventUpdate

class InteractionSummaryDependenceConnector(ExplainerComponent):
    def __init__(self, interaction_summary_component, interaction_dependence_component):
        self.sum_name = interaction_summary_component.name
        self.dep_name = interaction_dependence_component.name

    def register_callbacks(self, app):
        @app.callback(
            Output('interaction-dependence-group-cats-'+self.dep_name, 'checked'),
            [Input('interaction-summary-group-cats-'+self.sum_name, 'checked')],
            [State('tabs', 'value')])
        def update_dependence_shap_scatter_graph(cats,  tab):
            return cats

        @app.callback(
            [Output('interaction-dependence-col-'+self.dep_name, 'value'),
             Output('interaction-dependence-highlight-index-'+self.dep_name, 'value'),
             Output('interaction-dependence-interact-col-'+self.dep_name, 'value')],
            [Input('interaction-summary-col-'+self.sum_name, 'value'),
             Input('interaction-summary-graph-'+self.sum_name, 'clickData')])
        def update_interact_col_highlight(col, clickdata):
            if clickdata is not None and clickdata['points'][0] is not None:
                if isinstance(clickdata['points'][0]['y'], float): # detailed
                    # if detailed, clickdata returns scatter marker location -> type==float
                    idx = clickdata['points'][0]['pointIndex']
                    interact_col = clickdata['points'][0]['text'].split('=')[0]                             
                    return (col, idx, col)
                elif isinstance(clickdata['points'][0]['y'], str): # aggregate
                    # in aggregate clickdata returns col name -> type==str
                    interact_col = clickdata['points'][0]['y'].split(' ')[1]
                    return (col, dash.no_update, interact_col)
            else:
                return (col, dash.no_update, dash.no_update)
            raise PreventUpdate    

