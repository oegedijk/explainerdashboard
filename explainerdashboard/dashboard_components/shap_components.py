__all__ = ['ShapSummaryComponent',
            'ShapDependenceComponent',
            'ShapSummaryDependenceConnector',
            'InteractionSummaryComponent',
            'InteractionDependenceComponent',
            'InteractionSummaryDependenceConnector',
            'ShapContributionsTableComponent',
            'ShapContributionsGraphComponent']

import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html

from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from ..dashboard_methods import *


class ShapSummaryComponent(ExplainerComponent):
    def __init__(self, explainer, title='Shap Dependence Summary', name=None,
                    hide_title=False, hide_depth=False, 
                    hide_type=False, hide_cats=False, hide_index=False, hide_selector=False,
                    pos_label=None, depth=None, 
                    summary_type="aggregate", cats=True, index=None):
        """Shows shap summary component

        Args:
            explainer (Explainer): explainer object constructed with either
                        ClassifierExplainer() or RegressionExplainer()
            title (str, optional): Title of tab or page. Defaults to 
                        "Shap Dependence Summary".
            name (str, optional): unique name to add to Component elements. 
                        If None then random uuid is generated to make sure 
                        it's unique. Defaults to None.
            hide_title (bool, optional): hide the title above component. 
                        Defaults to False.
            hide_depth (bool, optional): hide the depth toggle. 
                        Defaults to False.
            hide_type (bool, optional): hide the summary type toggle 
                        (aggregated, detailed). Defaults to False.
            hide_cats (bool, optional): hide the group cats toggle. Defaults to False.
            hide_selector (bool, optional): hide pos label selector. Defaults to False.
            pos_label ({int, str}, optional): initial pos label. 
                        Defaults to explainer.pos_label
            depth (int, optional): initial number of features to show. Defaults to None.
            summary_type (str, {'aggregate', 'detailed'}. optional): type of 
                        summary graph to show. Defaults to "aggregate".
            cats (bool, optional): group cats. Defaults to True.
        """
        super().__init__(explainer, title, name)

        if self.explainer.cats is None or not self.explainer.cats:
            self.hide_cats = True
        
        if self.depth is not None:
            self.depth = min(self.depth, self.explainer.n_features(cats))

        self.index_name = 'shap-summary-index-'+self.name
        self.selector = PosLabelSelector(explainer, name=self.name, pos_label=pos_label)
        self.register_dependencies('shap_values', 'shap_values_cats')
             
    def layout(self):
        return dbc.Container([
            make_hideable(html.H3('Shap Summary'), hide=self.hide_title),
            dbc.Row([
                make_hideable(
                    dbc.Col([
                        dbc.Label("Depth:"),
                        dcc.Dropdown(id='shap-summary-depth-'+self.name,
                            options=[{'label': str(i+1), 'value': i+1} for i in 
                                        range(self.explainer.n_features(self.cats))],
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
                make_hideable(
                    dbc.Col([
                        html.Div([
                            dbc.Label(f"{self.explainer.index_name}:"),
                            dcc.Dropdown(id='shap-summary-index-'+self.name, 
                                options = [{'label': str(idx), 'value':idx} 
                                                for idx in self.explainer.idxs],
                                value=self.index),
                        ], id='shap-summary-index-col-'+self.name, style=dict(display="none")), 
                    ], md=3), hide=self.hide_index),  
                make_hideable(
                        dbc.Col([self.selector.layout()
                    ], width=2), hide=self.hide_selector)
                ], form=True),

            dcc.Loading(id="loading-dependence-shap-summary-"+self.name, 
                    children=[dcc.Graph(id="shap-summary-graph-"+self.name,
                                        config=dict(modeBarButtons=[['toImage']], displaylogo=False))])
            
        ], fluid=True)
    
    def _register_callbacks(self, app):

        @app.callback(
            Output('shap-summary-index-'+self.name, 'value'),
            [Input('shap-summary-graph-'+self.name, 'clickData')])
        def display_scatter_click_data(clickdata):
            if clickdata is not None and clickdata['points'][0] is not None:
                if isinstance(clickdata['points'][0]['y'], float): # detailed
                    index = clickdata['points'][0]['text'].split('=')[1].split('<br>')[0]                 
                    return index
            raise PreventUpdate

        @app.callback(
            [Output('shap-summary-graph-'+self.name, 'figure'),
             Output('shap-summary-depth-'+self.name, 'options'),
             Output('shap-summary-index-col-'+self.name, 'style')],
            [Input('shap-summary-type-'+self.name, 'value'),
             Input('shap-summary-group-cats-'+self.name, 'checked'),
             Input('shap-summary-depth-'+self.name, 'value'),
             Input('shap-summary-index-'+self.name, 'value'),
             Input('pos-label-'+self.name, 'value')])
        def update_shap_summary_graph(summary_type, cats, depth, index, pos_label):
            if summary_type == 'aggregate':
                plot = self.explainer.plot_importances(
                        kind='shap', topx=depth, cats=cats, pos_label=pos_label)
            elif summary_type == 'detailed':
                plot = self.explainer.plot_shap_summary(
                        topx=depth, cats=cats, pos_label=pos_label, index=index)
            ctx = dash.callback_context
            trigger = ctx.triggered[0]['prop_id'].split('.')[0]
            if trigger == 'shap-summary-group-cats-'+self.name:
                depth_options = [{'label': str(i+1), 'value': i+1} 
                                        for i in range(self.explainer.n_features(cats))]
                return (plot, depth_options, dash.no_update)
            elif trigger == 'shap-summary-type-'+self.name:
                if summary_type == 'aggregate':
                    return (plot, dash.no_update, dict(display="none"))
                elif summary_type == 'detailed':
                    return (plot, dash.no_update, {})
            else:
                return (plot, dash.no_update, dash.no_update)


class ShapDependenceComponent(ExplainerComponent):
    def __init__(self, explainer, title='Shap Dependence', name=None,
                    hide_title=False, hide_cats=False, hide_col=False, 
                    hide_color_col=False, hide_index=False,
                    hide_selector=False,
                    pos_label=None, cats=True, col=None, 
                    color_col=None, index=None):
        """Show shap dependence graph

        Args:
            explainer (Explainer): explainer object constructed with either
                        ClassifierExplainer() or RegressionExplainer()
            title (str, optional): Title of tab or page. Defaults to 
                        "Shap Dependence".
            name (str, optional): unique name to add to Component elements. 
                        If None then random uuid is generated to make sure 
                        it's unique. Defaults to None.
            hide_title (bool, optional): hide component title. Defaults to False.
            hide_cats (bool, optional): hide group cats toggle. Defaults to False.
            hide_col (bool, optional): hide feature selector. Defaults to False.
            hide_color_col (bool, optional): hide color feature selector Defaults to False.
            hide_index (bool, optional): hide index selector Defaults to False.
            hide_selector (bool, optional): hide pos label selector. Defaults to False.
            pos_label ({int, str}, optional): initial pos label. 
                        Defaults to explainer.pos_label
            cats (bool, optional): group cats. Defaults to True.
            col (str, optional): Feature to display. Defaults to None.
            color_col (str, optional): Color plot by values of this Feature. 
                        Defaults to None.
            index (int, optional): Highlight a particular index. Defaults to None.
        """
        super().__init__(explainer, title, name)

        if self.col is None:
            self.col = self.explainer.columns_ranked_by_shap(self.cats)[0]
        if self.color_col is None:
            self.color_col = self.explainer.shap_top_interactions(self.col, cats=cats)[1]

        self.selector = PosLabelSelector(explainer, name=self.name, pos_label=pos_label)
        self.index_name = 'shap-dependence-index-'+self.name
        self.register_dependencies('shap_values', 'shap_values_cats')
             
    def layout(self):
        return html.Div([
            make_hideable(html.H3('Shap Dependence Plot'), hide=self.hide_title),
            dbc.Row([
                make_hideable(
                        dbc.Col([self.selector.layout()
                    ], width=2), hide=self.hide_selector),    
            ]),
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
                        ], md=3), self.hide_col),
                    make_hideable(dbc.Col([
                            html.Label('Color feature:'),
                            dcc.Dropdown(id='shap-dependence-color-col-'+self.name, 
                                options=[{'label': col, 'value':col} 
                                            for col in self.explainer.columns_ranked_by_shap(self.cats)],
                                value=self.color_col),   
                    ], md=3), self.hide_color_col),
                    make_hideable(
                        dbc.Col([
                            dbc.Label(f"{self.explainer.index_name}:"),
                            dcc.Dropdown(id='shap-dependence-index-'+self.name, 
                                options = [{'label': str(idx), 'value':idx} 
                                                for idx in self.explainer.idxs],
                                value=self.index)
                        ], md=4), hide=self.hide_index),         
            ], form=True),
            dcc.Loading(id="loading-dependence-graph-"+self.name, 
                         children=[dcc.Graph(id='shap-dependence-graph-'+self.name,
                                            config=dict(modeBarButtons=[['toImage']], displaylogo=False))]),
        ])

    def _register_callbacks(self, app):
        @app.callback(
            [Output('shap-dependence-color-col-'+self.name, 'options'),
             Output('shap-dependence-color-col-'+self.name, 'value')],
            [Input('shap-dependence-col-'+self.name, 'value')],
            [State('shap-dependence-group-cats-'+self.name, 'checked'),
             State('pos-label-'+self.name, 'value')])
        def set_color_col_dropdown(col, cats, pos_label):
            sorted_interact_cols = self.explainer.shap_top_interactions(col, cats=cats, pos_label=pos_label)
            options = [{'label': col, 'value':col} 
                                        for col in sorted_interact_cols]
            value = sorted_interact_cols[1]                                
            return (options, value)

        @app.callback(
            Output('shap-dependence-graph-'+self.name, 'figure'),
            [Input('shap-dependence-color-col-'+self.name, 'value'),
             Input('shap-dependence-index-'+self.name, 'value'),
             Input('pos-label-'+self.name, 'value')],
            [State('shap-dependence-col-'+self.name, 'value')])
        def update_dependence_graph(color_col, index, pos_label, col):
            if col is not None:
                return self.explainer.plot_shap_dependence(
                            col, color_col, highlight_index=index, pos_label=pos_label)
            raise PreventUpdate

        @app.callback(
            Output('shap-dependence-col-'+self.name, 'options'),
            [Input('shap-dependence-group-cats-'+self.name, 'checked')],
            [State('shap-dependence-col-'+self.name, 'value')])
        def update_dependence_shap_scatter_graph(cats, old_col):
            options = [{'label': col, 'value': col} 
                                    for col in self.explainer.columns_ranked_by_shap(cats)]
            return options
            

class ShapSummaryDependenceConnector(ExplainerComponent):
    def __init__(self, shap_summary_component, shap_dependence_component):
        """Connects a ShapSummaryComponent with a ShapDependence Component:

        - When group cats in ShapSummary, then group cats in ShapDependence
        - When clicking on feature in ShapSummary, then select that feature in ShapDependence

        Args:
            shap_summary_component (ShapSummaryComponent): ShapSummaryComponent
            shap_dependence_component (ShapDependenceComponent): ShapDependenceComponent
        """
        self.sum_name = shap_summary_component.name
        self.dep_name = shap_dependence_component.name

    def _register_callbacks(self, app):
        @app.callback(
            Output('shap-dependence-group-cats-'+self.dep_name, 'checked'),
            [Input('shap-summary-group-cats-'+self.sum_name, 'checked')])
        def update_dependence_shap_scatter_graph(cats):
            return cats

        @app.callback(
            [Output('shap-dependence-index-'+self.dep_name, 'value'),
             Output('shap-dependence-col-'+self.dep_name, 'value')],
            [Input('shap-summary-graph-'+self.sum_name, 'clickData')])
        def display_scatter_click_data(clickdata):
            if clickdata is not None and clickdata['points'][0] is not None:
                if isinstance(clickdata['points'][0]['y'], float): # detailed
                    index = clickdata['points'][0]['text'].split('=')[1].split('<br>')[0]
                    col = clickdata['points'][0]['text'].split('=')[1].split('<br>')[1]                        
                    return (index, col)
                elif isinstance(clickdata['points'][0]['y'], str): # aggregate
                    # in aggregate clickdata returns col name -> type==str
                    col = clickdata['points'][0]['y'].split(' ')[1]
                    return (dash.no_update, col)
            raise PreventUpdate


class InteractionSummaryComponent(ExplainerComponent):
    def __init__(self, explainer, title="Interactions Summary", name=None,
                    hide_title=False, hide_col=False, hide_depth=False, 
                    hide_type=False, hide_cats=False, hide_index=False, hide_selector=False,
                    pos_label=None, col=None, depth=None, 
                    summary_type="aggregate", cats=True, index=None):
        """Show SHAP Interaciton values summary component

        Args:
            explainer (Explainer): explainer object constructed with either
                        ClassifierExplainer() or RegressionExplainer()
            title (str, optional): Title of tab or page. Defaults to 
                        "Interactions Summary".
            name (str, optional): unique name to add to Component elements. 
                        If None then random uuid is generated to make sure 
                        it's unique. Defaults to None.
            hide_title (bool, optional): hide the component title. Defaults to False.
            hide_col (bool, optional): Hide the feature selector. Defaults to False.
            hide_depth (bool, optional): Hide depth toggle. Defaults to False.
            hide_type (bool, optional): Hide summary type toggle. Defaults to False.
            hide_cats (bool, optional): Hide group cats toggle. Defaults to False.
            hide_index (bool, optional): Hide the index selector. Defaults to False
            hide_selector (bool, optional): hide pos label selector. Defaults to False.
            pos_label ({int, str}, optional): initial pos label. 
                        Defaults to explainer.pos_label
            col (str, optional): Feature to show interaction summary for. Defaults to None.
            depth (int, optional): Number of interaction features to display. Defaults to None.
            summary_type (str, {'aggregate', 'detailed'}, optional): type of summary graph to display. Defaults to "aggregate".
            cats (bool, optional): Group categorical features. Defaults to True.
            index (str):    Default index. Defaults to None.
        """
        super().__init__(explainer, title, name)
    
        if self.col is None:
            self.col = self.explainer.columns_ranked_by_shap(self.cats)[0]
        if self.depth is not None:
            self.depth = min(self.depth, self.explainer.n_features(self.cats)-1)
        self.index_name = 'interaction-summary-index-'+self.name
        self.selector = PosLabelSelector(explainer, name=self.name, pos_label=pos_label)
        self.register_dependencies("shap_interaction_values", "shap_interaction_values_cats")

    def layout(self):
        return dbc.Container([
            make_hideable(html.H3('Shap Interaction Summary'), hide=self.hide_title),
            dbc.Row([
                make_hideable(
                    dbc.Col([
                        dbc.Label("Feature"),
                        dcc.Dropdown(id='interaction-summary-col-'+self.name, 
                            options=[{'label': col, 'value': col} 
                                        for col in self.explainer.columns_ranked_by_shap(self.cats)],
                            value=self.col),
                    ], md=3), self.hide_col),
                make_hideable(
                    dbc.Col([
    
                        dbc.Label("Depth:"),
                        dcc.Dropdown(id='interaction-summary-depth-'+self.name, 
                            options = [{'label': str(i+1), 'value':i+1} 
                                            for i in range(self.explainer.n_features(self.cats)-1)],
                            value=self.depth)
                    ], md=1), self.hide_depth),
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
                    ],md=2), self.hide_cats),
                make_hideable(
                    dbc.Col([
                        html.Div([
                            dbc.Label(f"{self.explainer.index_name}:"),
                            dcc.Dropdown(id='interaction-summary-index-'+self.name, 
                                options = [{'label': str(idx), 'value':idx} 
                                                for idx in self.explainer.idxs],
                                value=self.index),
                        ], id='interaction-summary-index-col-'+self.name, style=dict(display="none")), 
                    ], md=3), hide=self.hide_index),  
                make_hideable(
                        dbc.Col([self.selector.layout()
                    ], width=2), hide=self.hide_selector),
                ], form=True),
            dbc.Row([
                dbc.Col([
                    dcc.Loading(id='loading-interaction-summary-graph-'+self.name, 
                         children=[dcc.Graph(id='interaction-summary-graph-'+self.name, 
                                            config=dict(modeBarButtons=[['toImage']], displaylogo=False))])
                ])
            ]), 
        ], fluid=True)

    def _register_callbacks(self, app):
        @app.callback(
            Output('interaction-summary-index-'+self.name, 'value'),
            [Input('interaction-summary-graph-'+self.name, 'clickData')])
        def display_scatter_click_data(clickdata):
            if clickdata is not None and clickdata['points'][0] is not None:
                if isinstance(clickdata['points'][0]['y'], float): # detailed
                    index = clickdata['points'][0]['text'].split('=')[1].split('<br>')[0]                 
                    return index
            raise PreventUpdate

        @app.callback(
            [Output('interaction-summary-depth-'+self.name, 'options'),
             Output('interaction-summary-col-'+self.name, 'options')],
            [Input('interaction-summary-group-cats-'+self.name, 'checked'),
             Input('pos-label-'+self.name, 'value')])
        def update_interaction_scatter_graph(cats, pos_label):
            depth_options = [{'label': str(i+1), 'value': i+1} 
                                    for i in range(self.explainer.n_features(cats))]
            new_cols = self.explainer.columns_ranked_by_shap(cats, pos_label=pos_label)
            new_col_options = [{'label': col, 'value':col} for col in new_cols]
            return depth_options, new_col_options

        @app.callback(
            [Output('interaction-summary-graph-'+self.name, 'figure'),
             Output('interaction-summary-index-col-'+self.name, 'style')],
            [Input('interaction-summary-col-'+self.name, 'value'),
             Input('interaction-summary-depth-'+self.name, 'value'),
             Input('interaction-summary-type-'+self.name, 'value'),
             Input('interaction-summary-index-'+self.name, 'value'),
             Input('pos-label-'+self.name, 'value'),
             Input('interaction-summary-group-cats-'+self.name, 'checked')])
        def update_interaction_scatter_graph(col, depth, summary_type, index, pos_label, cats):
            if col is not None:
                if summary_type=='aggregate':
                    plot = self.explainer.plot_interactions(
                        col, topx=depth, cats=cats, pos_label=pos_label)
                    return plot, dict(display="none")
                elif summary_type=='detailed':
                    plot = self.explainer.plot_shap_interaction_summary(
                        col, topx=depth, cats=cats, pos_label=pos_label, index=index)
                return plot, {}
            raise PreventUpdate


class InteractionDependenceComponent(ExplainerComponent):
    def __init__(self, explainer, title="Interaction Dependence", name=None,
                    hide_title=False, hide_cats=False, hide_col=False, 
                    hide_interact_col=False, hide_index=False,
                    hide_selector=False, hide_top=False, hide_bottom=False,
                    pos_label=None, cats=True, col=None, interact_col=None, index=None):
        """Interaction Dependence Component.

        Shows two graphs:
            top graph: col vs interact_col
            bottom graph: interact_col vs col

        Args:
            explainer (Explainer): explainer object constructed with either
                        ClassifierExplainer() or RegressionExplainer()
            title (str, optional): Title of tab or page. Defaults to 
                        "Interactions Dependence".
            name (str, optional): unique name to add to Component elements. 
                        If None then random uuid is generated to make sure 
                        it's unique. Defaults to None.
            hide_title (bool, optional): Hide component title. Defaults to False.
            hide_cats (bool, optional): Hide group cats toggle. Defaults to False.
            hide_col (bool, optional): Hide feature selector. Defaults to False.
            hide_interact_col (bool, optional): Hide interaction 
                        feature selector. Defaults to False.
            hide_highlight (bool, optional): Hide highlight index selector.
                        Defaults to False.
            hide_selector (bool, optional): hide pos label selector. 
                        Defaults to False.
            hide_top (bool, optional): Hide the top interaction graph 
                        (col vs interact_col). Defaults to False.
            hide_bottom (bool, optional): hide the bottom interaction graph 
                        (interact_col vs col). Defaults to False.
            pos_label ({int, str}, optional): initial pos label. 
                        Defaults to explainer.pos_label
            cats (bool, optional): group categorical features. Defaults to True.
            col (str, optional): Feature to find interactions for. Defaults to None.
            interact_col (str, optional): Feature to interact with. Defaults to None.
            highlight (int, optional): Index row to highlight Defaults to None.
        """
        super().__init__(explainer, title, name)

        if self.col is None:
            self.col = explainer.columns_ranked_by_shap(cats)[0]
        if self.interact_col is None:
            self.interact_col = explainer.shap_top_interactions(self.col, cats=cats)[1]
        
        self.selector = PosLabelSelector(explainer, name=self.name, pos_label=pos_label)
        self.register_dependencies("shap_interaction_values", "shap_interaction_values_cats")

    def layout(self):
        return dbc.Container([
            make_hideable(html.H3('Shap Interaction Plots'), hide=self.hide_title),
            dbc.Row([
                make_hideable(
                        dbc.Col([self.selector.layout()
                    ], width=2), hide=self.hide_selector),
            ]),
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
                    ], md=3), hide=self.hide_col), 
                make_hideable(
                    dbc.Col([
                        dbc.Label("Interaction:"),
                        dcc.Dropdown(id='interaction-dependence-interact-col-'+self.name, 
                            options=[{'label': col, 'value':col} 
                                        for col in self.explainer.shap_top_interactions(col=self.col, cats=self.cats)],
                            value=self.interact_col
                        ),
                    ], md=3), hide=self.hide_interact_col), 
                make_hideable(
                        dbc.Col([
                            dbc.Label(f"{self.explainer.index_name}:"),
                            dcc.Dropdown(id='interaction-dependence-index-'+self.name, 
                                options = [{'label': str(idx), 'value':idx} 
                                                for idx in self.explainer.idxs],
                                value=self.index)
                        ], md=4), hide=self.hide_index), 
                ], form=True),
        dbc.Row([
            dbc.Col([
                make_hideable(
                dcc.Loading(id='loading-interaction-dependence-graph-'+self.name, 
                         children=[dcc.Graph(id='interaction-dependence-graph-'+self.name,
                                        config=dict(modeBarButtons=[['toImage']], displaylogo=False))]),
                         hide=self.hide_top),
            ])
        ]),
        dbc.Row([
            dbc.Col([
                make_hideable(
                dcc.Loading(id='loading-reverse-interaction-graph-'+self.name, 
                         children=[dcc.Graph(id='interaction-dependence-reverse-graph-'+self.name,
                                        config=dict(modeBarButtons=[['toImage']], displaylogo=False))]),
                         hide=self.hide_bottom),
            ])
        ]),
        ], fluid=True)

    def _register_callbacks(self, app):
        @app.callback(
            Output('interaction-dependence-col-'+self.name, 'options'), 
            [Input('interaction-dependence-group-cats-'+self.name, 'checked'),
             Input('pos-label-'+self.name, 'value')])
        def update_interaction_dependence_interact_col(cats, pos_label):
            new_cols = self.explainer.columns_ranked_by_shap(cats, pos_label=pos_label)
            new_col_options = [{'label': col, 'value':col} for col in new_cols]
            return new_col_options

        @app.callback(
            Output('interaction-dependence-interact-col-'+self.name, 'options'),
            [Input('interaction-dependence-col-'+self.name, 'value'),
             Input('pos-label-'+self.name, 'value')],
            [State('interaction-dependence-group-cats-'+self.name, 'checked'),
             State('interaction-dependence-interact-col-'+self.name, 'value')])
        def update_interaction_dependence_interact_col(col, pos_label, cats, old_interact_col):
            if col is not None:
                new_interact_cols = self.explainer.shap_top_interactions(col, cats=cats, pos_label=pos_label)
                new_interact_options = [{'label': col, 'value':col} for col in new_interact_cols]
                return new_interact_options
            raise PreventUpdate

        @app.callback(
            [Output('interaction-dependence-graph-'+self.name, 'figure'),
             Output('interaction-dependence-reverse-graph-'+self.name, 'figure')],
            [Input('interaction-dependence-interact-col-'+self.name, 'value'),
             Input('interaction-dependence-index-'+self.name, 'value'),
             Input('pos-label-'+self.name, 'value'),
             Input('interaction-dependence-col-'+self.name, 'value')])
        def update_dependence_graph(interact_col, index, pos_label, col):
            if col is not None and interact_col is not None:
                return (self.explainer.plot_shap_interaction(
                            col, interact_col, highlight_index=index, pos_label=pos_label),
                        self.explainer.plot_shap_interaction(
                            interact_col, col, highlight_index=index, pos_label=pos_label))
            raise PreventUpdate

        
class InteractionSummaryDependenceConnector(ExplainerComponent):
    def __init__(self, interaction_summary_component, interaction_dependence_component):
        """Connects a InteractionSummaryComponent with an InteractionDependenceComponent:

        - When group cats in Summary, then group cats in Dependence
        - When select feature in summary, then select col in Dependence
        - When clicking on interaction feature in Summary, then select that interaction 
            feature in Dependence.

        Args:
            shap_summary_component (ShapSummaryComponent): ShapSummaryComponent
            shap_dependence_component (ShapDependenceComponent): ShapDependenceComponent
        """
        self.sum_name = interaction_summary_component.name
        self.dep_name = interaction_dependence_component.name

    def _register_callbacks(self, app):
        @app.callback(
            Output('interaction-dependence-group-cats-'+self.dep_name, 'checked'),
            [Input('interaction-summary-group-cats-'+self.sum_name, 'checked')])
        def update_dependence_shap_scatter_graph(cats):
            return cats

        @app.callback(
            [Output('interaction-dependence-col-'+self.dep_name, 'value'),
             Output('interaction-dependence-index-'+self.dep_name, 'value'),
             Output('interaction-dependence-interact-col-'+self.dep_name, 'value')],
            [Input('interaction-summary-col-'+self.sum_name, 'value'),
             Input('interaction-summary-graph-'+self.sum_name, 'clickData')])
        def update_interact_col_highlight(col, clickdata):
            if clickdata is not None and clickdata['points'][0] is not None:
                if isinstance(clickdata['points'][0]['y'], float): # detailed
                    index = clickdata['points'][0]['text'].split('=')[1].split('<br>')[0]
                    interact_col = clickdata['points'][0]['text'].split('=')[1].split('<br>')[1]                          
                    return (col, index, interact_col)
                elif isinstance(clickdata['points'][0]['y'], str): # aggregate
                    # in aggregate clickdata returns col name -> type==str
                    interact_col = clickdata['points'][0]['y'].split(' ')[1]
                    return (col, dash.no_update, interact_col)
            else:
                return (col, dash.no_update, dash.no_update)
            raise PreventUpdate   


class ShapContributionsGraphComponent(ExplainerComponent):
    def __init__(self, explainer, title="Contributions", name=None,
                    hide_title=False, hide_index=False, hide_depth=False, 
                    hide_sort=False, hide_orientation=False, hide_cats=False, 
                    hide_selector=False,
                    pos_label=None, index=None, depth=None, sort='high-to-low', 
                    orientation='vertical', cats=True):
        """Display Shap contributions to prediction graph component

        Args:
            explainer (Explainer): explainer object constructed with either
                        ClassifierExplainer() or RegressionExplainer()
            title (str, optional): Title of tab or page. Defaults to 
                        "Contributions".
            name (str, optional): unique name to add to Component elements. 
                        If None then random uuid is generated to make sure 
                        it's unique. Defaults to None.
            hide_title (bool, optional): Hide component title. Defaults to False.
            hide_index (bool, optional): Hide index selector. Defaults to False.
            hide_depth (bool, optional): Hide depth toggle. Defaults to False.
            hide_sort (bool, optional): Hide the sorting dropdown. Defaults to False.
            hide_orientation (bool, optional): Hide the orientation dropdown. 
                    Defaults to False
            hide_cats (bool, optional): Hide group cats toggle. Defaults to False.
            hide_selector (bool, optional): hide pos label selector. Defaults to False.
            pos_label ({int, str}, optional): initial pos label. 
                        Defaults to explainer.pos_label
            index ({int, bool}, optional): Initial index to display. Defaults to None.
            depth (int, optional): Initial number of features to display. Defaults to None.
            sort ({'abs', 'high-to-low', 'low-to-high', 'importance'}, optional): sorting of shap values. 
                        Defaults to 'high-to-low'.
            orientation ({'vertical', 'horizontal'}, optional): orientation of bar chart.
                        Defaults to 'vertical'.
            cats (bool, optional): Group cats. Defaults to True.
        """
        super().__init__(explainer, title, name)

        self.index_name = 'contributions-graph-index-'+self.name

        if self.depth is not None:
            self.depth = min(self.depth, self.explainer.n_features(self.cats))

        self.selector = PosLabelSelector(explainer, name=self.name, pos_label=pos_label)
        self.register_dependencies('shap_values', 'shap_values_cats')

    def layout(self):
        return html.Div([
            make_hideable(html.H3("Contributions to prediction:"), hide=self.hide_title),
            dbc.Row([
                make_hideable(
                        dbc.Col([self.selector.layout()
                    ], md=2), hide=self.hide_selector),
            ], justify="right"),
            dbc.Row([
                make_hideable(
                    dbc.Col([
                        dbc.Label(f"{self.explainer.index_name}:"),
                        dcc.Dropdown(id='contributions-graph-index-'+self.name, 
                            options = [{'label': str(idx), 'value':idx} 
                                            for idx in self.explainer.idxs],
                            value=None)
                    ], md=4), hide=self.hide_index), 
                make_hideable(
                    dbc.Col([
                        dbc.Label("Depth:"),
                        dcc.Dropdown(id='contributions-graph-depth-'+self.name, 
                            options = [{'label': str(i+1), 'value':i+1} 
                                            for i in range(self.explainer.n_features(self.cats))],
                            value=self.depth)
                    ], md=2), hide=self.hide_depth),
                make_hideable(
                    dbc.Col([
                        dbc.Label("Sorting:"),
                        dcc.Dropdown(id='contributions-graph-sorting-'+self.name, 
                            options = [{'label': 'Absolute', 'value': 'abs'},
                                        {'label': 'High to Low', 'value': 'high-to-low'},
                                        {'label': 'Low to High', 'value': 'low-to-high'},
                                        {'label': 'Importance', 'value': 'importance'}],
                            value=self.sort)
                    ], md=2), hide=self.hide_sort),
                make_hideable(
                    dbc.Col([
                        dbc.Label("Orientation:"),
                        dcc.Dropdown(id='contributions-graph-orientation-'+self.name, 
                            options = [{'label': 'Vertical', 'value': 'vertical'},
                                        {'label': 'Horizontal', 'value': 'horizontal'}],
                            value=self.orientation)
                    ], md=2), hide=self.hide_orientation),
                make_hideable(
                    dbc.Col([
                        dbc.Label("Grouping:"),
                        dbc.FormGroup(
                        [
                            dbc.RadioButton(
                                id='contributions-graph-group-cats-'+self.name, 
                                className="form-check-input",
                                checked=self.cats),
                            dbc.Label("Group Cats",
                                    html_for='contributions-graph-group-cats-'+self.name, 
                                    className="form-check-label"),
                        ], check=True)
                    ], md=2), hide=self.hide_cats),
                ], form=True),
                
            dbc.Row([
                dbc.Col([
                    dcc.Loading(id='loading-contributions-graph-'+self.name, 
                        children=[dcc.Graph(id='contributions-graph-'+self.name,
                                config=dict(modeBarButtons=[['toImage']], displaylogo=False))]),
                ]),
            ]),
            dbc.Row([
                
                ], form=True),
        ])
        
    def _register_callbacks(self, app):
        @app.callback(
            [Output('contributions-graph-'+self.name, 'figure'),
             Output('contributions-graph-depth-'+self.name, 'options')],
            [Input('contributions-graph-index-'+self.name, 'value'),
             Input('contributions-graph-depth-'+self.name, 'value'),
             Input('contributions-graph-sorting-'+self.name, 'value'),
             Input('contributions-graph-orientation-'+self.name, 'value'),
             Input('contributions-graph-group-cats-'+self.name, 'checked'),
             Input('pos-label-'+self.name, 'value')])
        def update_output_div(index, depth, sort, orientation, cats, pos_label):
            if index is None:
                raise PreventUpdate

            plot = self.explainer.plot_shap_contributions(index, topx=depth, 
                        cats=cats, sort=sort, orientation=orientation, pos_label=pos_label)
            
            ctx = dash.callback_context
            trigger = ctx.triggered[0]['prop_id'].split('.')[0]
            if trigger == 'contributions-graph-group-cats-'+self.name:
                depth_options = [{'label': str(i+1), 'value': i+1} 
                                        for i in range(self.explainer.n_features(cats))]
                return (plot, depth_options)
            else:
                return (plot, dash.no_update)


class ShapContributionsTableComponent(ExplainerComponent):
    def __init__(self, explainer, title="Contributions", name=None,
                    hide_title=False, hide_index=False, 
                    hide_depth=False, hide_sort=False, hide_cats=False, 
                    hide_selector=False,
                    pos_label=None, index=None, depth=None, sort='abs', cats=True):
        """Show SHAP values contributions to prediction in a table component

        Args:
            explainer (Explainer): explainer object constructed with either
                        ClassifierExplainer() or RegressionExplainer()
            title (str, optional): Title of tab or page. Defaults to 
                        "Contributions".
            name (str, optional): unique name to add to Component elements. 
                        If None then random uuid is generated to make sure 
                        it's unique. Defaults to None.
            hide_title (bool, optional): Hide component title. Defaults to False.
            hide_index (bool, optional): Hide index selector. Defaults to False.
            hide_depth (bool, optional): Hide depth selector. Defaults to False.
            hide_sort (bool, optional): Hide sorting dropdown. Default to False.
            hide_cats (bool, optional): Hide group cats toggle. Defaults to False.
            hide_selector (bool, optional): hide pos label selector. Defaults to False.
            pos_label ({int, str}, optional): initial pos label. 
                        Defaults to explainer.pos_label
            index ([type], optional): Initial index to display. Defaults to None.
            depth ([type], optional): Initial number of features to display. Defaults to None.
            cats (bool, optional): Group categoricals. Defaults to True.
        """
        super().__init__(explainer, title, name)

        self.index_name = 'contributions-table-index-'+self.name

        if self.depth is not None:
            self.depth = min(self.depth, self.explainer.n_features(self.cats))

        self.selector = PosLabelSelector(explainer, name=self.name, pos_label=pos_label)
        self.register_dependencies('shap_values', 'shap_values_cats')

    def layout(self):
        return html.Div([
            make_hideable(html.H3("Contributions to prediction:"), hide=self.hide_title),
            dbc.Row([
                make_hideable(
                    dbc.Col([
                        dbc.Label(f"{self.explainer.index_name}:"),
                        dcc.Dropdown(id='contributions-table-index-'+self.name, 
                            options = [{'label': str(idx), 'value':idx} 
                                            for idx in self.explainer.idxs],
                            value=None)
                    ], md=4), hide=self.hide_index), 
                make_hideable(
                    dbc.Col([
                        dbc.Label("Depth:"),
                        dcc.Dropdown(id='contributions-table-depth-'+self.name, 
                            options = [{'label': str(i+1), 'value':i+1} 
                                            for i in range(self.explainer.n_features(self.cats))],
                            value=self.depth)
                    ], md=2), hide=self.hide_depth),
                make_hideable(
                    dbc.Col([
                        dbc.Label("Sorting:"),
                        dcc.Dropdown(id='contributions-table-sorting-'+self.name, 
                            options = [{'label': 'Absolute', 'value': 'abs'},
                                        {'label': 'High to Low', 'value': 'high-to-low'},
                                        {'label': 'Low to High', 'value': 'low-to-high'}],
                            value=self.sort)
                    ], md=2), hide=self.hide_sort),
                make_hideable(
                        dbc.Col([self.selector.layout()
                    ], width=2), hide=self.hide_selector),
                make_hideable(
                    dbc.Col([
                        dbc.Label("Grouping:"),
                        dbc.FormGroup(
                        [
                            dbc.RadioButton(
                                id='contributions-table-group-cats-'+self.name, 
                                className="form-check-input",
                                checked=self.cats),
                            dbc.Label("Group Cats",
                                    html_for='contributions-table-group-cats-'+self.name, 
                                    className="form-check-label"),
                        ], check=True)
                    ], md=3), hide=self.hide_cats),
            ], form=True),
            dbc.Row([
                dbc.Col([
                    html.Div(id='contributions-table-'+self.name)
                ]),
            ]),
        ])
        
    def _register_callbacks(self, app):
        @app.callback(
            [Output('contributions-table-'+self.name, 'children'),
             Output('contributions-table-depth-'+self.name, 'options')],
            [Input('contributions-table-index-'+self.name, 'value'),
             Input('contributions-table-depth-'+self.name, 'value'),
             Input('contributions-table-sorting-'+self.name, 'value'),
             Input('contributions-table-group-cats-'+self.name, 'checked'),
             Input('pos-label-'+self.name, 'value')])
        def update_output_div(index, depth, sort, cats, pos_label):
            if index is None:
                raise PreventUpdate

            contributions_table = dbc.Table.from_dataframe(
                self.explainer.contrib_summary_df(index, cats=cats, topx=depth, 
                                sort=sort, pos_label=pos_label))

            tooltip_cols = {}
            for tr in contributions_table.children[1].children:
                # insert tooltip target id's into the table html.Tr() elements:
                tds = tr.children
                col = tds[0].children.split(" = ")[0]
                if self.explainer.description(col) != "":
                    tr.id = f"contributions-table-hover-{col}-"+self.name
                    tooltip_cols[col] = self.explainer.description(col)
            
            tooltips = [dbc.Tooltip(desc,
                        target=f"contributions-table-hover-{col}-"+self.name, 
                        placement="top") for col, desc in tooltip_cols.items()]

            output_div = html.Div([contributions_table, *tooltips])

            ctx = dash.callback_context
            trigger = ctx.triggered[0]['prop_id'].split('.')[0]
            if trigger == 'contributions-table-group-cats-'+self.name:
                depth_options = [{'label': str(i+1), 'value': i+1} 
                                        for i in range(self.explainer.n_features(cats))]
                return (output_div, depth_options)
            else:
                return (output_div, dash.no_update) 

