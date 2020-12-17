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
    def __init__(self, explainer, title='Shap Summary', name=None,
                    subtitle="Ordering features by shap value",
                    hide_title=False, hide_subtitle=False, hide_depth=False, 
                    hide_type=False, hide_cats=False, hide_index=False, hide_selector=False,
                    pos_label=None, depth=None, 
                    summary_type="aggregate", cats=True, index=None,
                    description=None, **kwargs):
        """Shows shap summary component

        Args:
            explainer (Explainer): explainer object constructed with either
                        ClassifierExplainer() or RegressionExplainer()
            title (str, optional): Title of tab or page. Defaults to 
                        "Shap Dependence Summary".
            name (str, optional): unique name to add to Component elements. 
                        If None then random uuid is generated to make sure 
                        it's unique. Defaults to None.
            subtitle (str): subtitle
            hide_title (bool, optional): hide the title. Defaults to False.
            hide_subtitle (bool, optional): Hide subtitle. Defaults to False.
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
            description (str, optional): Tooltip to display when hover over
                component title. When None default text is shown. 
        """
        super().__init__(explainer, title, name)

        if self.explainer.cats is None or not self.explainer.cats:
            self.hide_cats = True
        
        if self.depth is not None:
            self.depth = min(self.depth, self.explainer.n_features(cats))

        self.index_name = 'shap-summary-index-'+self.name
        self.selector = PosLabelSelector(explainer, name=self.name, pos_label=pos_label)
        if self.description is None: self.description = """
        The shap summary summarizes the shap values per feature.
        You can either select an aggregates display that shows mean absolute shap value
        per feature. Or get a more detailed look at the spread of shap values per
        feature and how they correlate the the feature value (red is high).
        """
        self.register_dependencies('shap_values', 'shap_values_cats')
             
    def layout(self):
        return dbc.Card([
            make_hideable(
                dbc.CardHeader([
                    html.Div([
                        html.H3(self.title, id='shap-summary-title-'+self.name),
                        make_hideable(html.H6(self.subtitle, className='card-subtitle'), hide=self.hide_subtitle),
                        dbc.Tooltip(self.description, target='shap-summary-title-'+self.name),
                    ]), 
                ]), hide=self.hide_title),
            dbc.CardBody([
                dbc.Row([
                    make_hideable(
                        dbc.Col([
                            dbc.Label("Depth:", id='shap-summary-depth-label-'+self.name),
                            dbc.Tooltip("Number of features to display", 
                                        target='shap-summary-depth-label-'+self.name),
                            dbc.Select(id='shap-summary-depth-'+self.name,
                                options=[{'label': str(i+1), 'value': i+1} for i in 
                                            range(self.explainer.n_features(self.cats))],
                                value=self.depth)
                        ], md=2), self.hide_depth),
                    make_hideable(
                        dbc.Col([
                            dbc.FormGroup(
                                [
                                    dbc.Label("Summary Type", id='shap-summary-type-label-'+self.name),
                                    dbc.Tooltip("Display mean absolute SHAP value per feature (aggregate)"
                                                " or display every single shap value per feature (detailed)", 
                                                target='shap-summary-type-label-'+self.name),
                                    dbc.Select(
                                        options=[
                                            {"label": "Aggregate", "value": "aggregate"},
                                            {"label": "Detailed", "value": "detailed"},
                                        ],
                                        value=self.summary_type,
                                        id="shap-summary-type-"+self.name,
                                    ),
                                ]
                            )
                        ]), self.hide_type),
                    make_hideable(
                        dbc.Col([
                            dbc.FormGroup([
                                    dbc.Label("Grouping:", id='shap-summary-group-cats-label-'+self.name),
                                    dbc.Tooltip("Group onehot encoded categorical variables together", 
                                                target='shap-summary-group-cats-label-'+self.name),
                                    dbc.Checklist(
                                        options=[{"label": "Group cats", "value": True}],
                                        value=[True] if self.cats else [],
                                        id='shap-summary-group-cats-'+self.name,
                                        inline=True,
                                        switch=True,
                                    ),
                                ]),
                        ], md=3), self.hide_cats),
                    make_hideable(
                        dbc.Col([
                            html.Div([
                                dbc.Label(f"{self.explainer.index_name}:", id='shap-summary-index-label-'+self.name),
                                dbc.Tooltip(f"Select {self.explainer.index_name} to highlight in plot. "
                                            "You can also select by clicking on a scatter point in the graph.", 
                                            target='shap-summary-index-label-'+self.name),
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
                                            config=dict(modeBarButtons=[['toImage']], displaylogo=False))]),
            ]),
        ])
    
    def component_callbacks(self, app):

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
             Input('shap-summary-group-cats-'+self.name, 'value'),
             Input('shap-summary-depth-'+self.name, 'value'),
             Input('shap-summary-index-'+self.name, 'value'),
             Input('pos-label-'+self.name, 'value')])
        def update_shap_summary_graph(summary_type, cats, depth, index, pos_label):
            cats = bool(cats)
            depth = None if depth is None else int(depth)
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
                    subtitle="Relationship between feature value and SHAP value",
                    hide_title=False, hide_subtitle=False, hide_cats=False, hide_col=False, 
                    hide_color_col=False, hide_index=False,
                    hide_selector=False,
                    pos_label=None, cats=True, col=None, 
                    color_col=None, index=None, description=None, **kwargs):
        """Show shap dependence graph

        Args:
            explainer (Explainer): explainer object constructed with either
                        ClassifierExplainer() or RegressionExplainer()
            title (str, optional): Title of tab or page. Defaults to 
                        "Shap Dependence".
            name (str, optional): unique name to add to Component elements. 
                        If None then random uuid is generated to make sure 
                        it's unique. Defaults to None.
            subtitle (str): subtitle
            hide_title (bool, optional): hide component title. Defaults to False.
            hide_subtitle (bool, optional): Hide subtitle. Defaults to False.
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
            description (str, optional): Tooltip to display when hover over
                component title. When None default text is shown. 
        """
        super().__init__(explainer, title, name)

        if self.col is None:
            self.col = self.explainer.columns_ranked_by_shap(self.cats)[0]
        if self.color_col is None:
            self.color_col = self.explainer.shap_top_interactions(self.col, cats=cats)[1]

        self.selector = PosLabelSelector(explainer, name=self.name, pos_label=pos_label)
        self.index_name = 'shap-dependence-index-'+self.name

        if self.description is None: self.description = """
        This plot shows the relation between feature values and shap values.
        This allows you to investigate the general relationship between feature
        value and impact on the prediction, i.e. "older passengers were predicted
        to be less likely to survive the titanic". You can check whether the mode
        uses features in line with your intuitions, or use the plots to learn
        about the relationships that the model has learned between the input features
        and the predicted outcome.
        """
        self.register_dependencies('shap_values', 'shap_values_cats')
             
    def layout(self):
        return dbc.Card([
            make_hideable(
                dbc.CardHeader([
                        html.Div([
                            html.H3(self.title, id='shap-dependence-title-'+self.name),
                            make_hideable(html.H6(self.subtitle, className='card-subtitle'), hide=self.hide_subtitle),
                            dbc.Tooltip(self.description, target='shap-dependence-title-'+self.name),
                        ]), 
                ]), hide=self.hide_title),
            dbc.CardBody([
                dbc.Row([
                    make_hideable(
                            dbc.Col([self.selector.layout()
                        ], width=2), hide=self.hide_selector),    
                ]),
                dbc.Row([
                    make_hideable(
                        dbc.Col([
                            dbc.FormGroup([
                                    dbc.Label("Grouping:", id='shap-dependence-group-cats-label-'+self.name),
                                    dbc.Tooltip("Group onehot encoded categorical variables together", 
                                                target='shap-dependence-group-cats-label-'+self.name),
                                    dbc.Checklist(
                                        options=[{"label": "Group cats", "value": True}],
                                        value=[True] if self.cats else [],
                                        id='shap-dependence-group-cats-'+self.name,
                                        inline=True,
                                        switch=True,
                                    ),
                                ]),
                            
                        ],  md=2), self.hide_cats),
                    make_hideable(
                        dbc.Col([
                            dbc.Label('Feature:', id='shap-dependence-col-label-'+self.name),
                            dbc.Tooltip("Select feature to display shap dependence for", 
                                        target='shap-dependence-col-label-'+self.name),
                            dbc.Select(id='shap-dependence-col-'+self.name, 
                                options=[{'label': col, 'value':col} 
                                            for col in self.explainer.columns_ranked_by_shap(self.cats)],
                                value=self.col)
                        ], md=3), self.hide_col),
                    make_hideable(dbc.Col([
                            html.Label('Color feature:', id='shap-dependence-color-col-label-'+self.name),
                            dbc.Tooltip("Select feature to color the scatter markers by. This "
                                        "allows you to see interactions between various features in the graph.", 
                                        target='shap-dependence-color-col-label-'+self.name),
                            dbc.Select(id='shap-dependence-color-col-'+self.name, 
                                options=[{'label': col, 'value':col} 
                                            for col in self.explainer.columns_ranked_by_shap(self.cats)],
                                value=self.color_col),   
                    ], md=3), self.hide_color_col),
                    make_hideable(
                        dbc.Col([
                            dbc.Label(f"{self.explainer.index_name}:", id='shap-dependence-index-label-'+self.name),
                            dbc.Tooltip(f"Select {self.explainer.index_name} to highlight in the plot."
                                        "You can also select by clicking on a scatter marker in the accompanying"
                                        " shap summary plot (detailed).", 
                                        target='shap-dependence-index-label-'+self.name),
                            dcc.Dropdown(id='shap-dependence-index-'+self.name, 
                                options = [{'label': str(idx), 'value':idx} 
                                                for idx in self.explainer.idxs],
                                value=self.index)
                        ], md=4), hide=self.hide_index),         
                ], form=True),
                dcc.Loading(id="loading-dependence-graph-"+self.name, 
                            children=[
                                dcc.Graph(id='shap-dependence-graph-'+self.name,
                                    config=dict(modeBarButtons=[['toImage']], displaylogo=False))]),
            ]), 
        ])

    def component_callbacks(self, app):
        @app.callback(
            [Output('shap-dependence-color-col-'+self.name, 'options'),
             Output('shap-dependence-color-col-'+self.name, 'value')],
            [Input('shap-dependence-col-'+self.name, 'value')],
            [State('shap-dependence-group-cats-'+self.name, 'value'),
             State('pos-label-'+self.name, 'value')])
        def set_color_col_dropdown(col, cats, pos_label):
            sorted_interact_cols = self.explainer.shap_top_interactions(
                                    col, cats=bool(cats), pos_label=pos_label)
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
            [Input('shap-dependence-group-cats-'+self.name, 'value')],
            [State('shap-dependence-col-'+self.name, 'value')])
        def update_dependence_shap_scatter_graph(cats, old_col):
            options = [{'label': col, 'value': col} 
                                    for col in self.explainer.columns_ranked_by_shap(bool(cats))]
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

    def component_callbacks(self, app):
        @app.callback(
            Output('shap-dependence-group-cats-'+self.dep_name, 'value'),
            [Input('shap-summary-group-cats-'+self.sum_name, 'value')])
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
                    subtitle="Ordering features by shap interaction value",
                    hide_title=False, hide_subtitle=False, hide_col=False, hide_depth=False, 
                    hide_type=False, hide_cats=False, hide_index=False, hide_selector=False,
                    pos_label=None, col=None, depth=None, 
                    summary_type="aggregate", cats=True, index=None, description=None,
                    **kwargs):
        """Show SHAP Interaciton values summary component

        Args:
            explainer (Explainer): explainer object constructed with either
                        ClassifierExplainer() or RegressionExplainer()
            title (str, optional): Title of tab or page. Defaults to 
                        "Interactions Summary".
            name (str, optional): unique name to add to Component elements. 
                        If None then random uuid is generated to make sure 
                        it's unique. Defaults to None.
            subtitle (str): subtitle
            hide_title (bool, optional): hide the component title. Defaults to False.
            hide_subtitle (bool, optional): Hide subtitle. Defaults to False.
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
            description (str, optional): Tooltip to display when hover over
                component title. When None default text is shown. 
        """
        super().__init__(explainer, title, name)
    
        if self.col is None:
            self.col = self.explainer.columns_ranked_by_shap(self.cats)[0]
        if self.depth is not None:
            self.depth = min(self.depth, self.explainer.n_features(self.cats)-1)
        if not self.explainer.cats:
            self.hide_cats = True
        self.index_name = 'interaction-summary-index-'+self.name
        self.selector = PosLabelSelector(explainer, name=self.name, pos_label=pos_label)
        if self.description is None: self.description = """
        Shows shap interaction values. Each shap value can be decomposed into a direct
        effect and indirect effects. The indirect effects are due to interactions
        of the feature with other feature. For example the fact that you know
        the gender of a passenger on the titanic will have a direct effect (women
        more likely to survive then men), but may also have indirect effects through
        for example passenger class (first class women more likely to survive than
        average woman, third class women less likely).
        """
        self.register_dependencies("shap_interaction_values", "shap_interaction_values_cats")

    def layout(self):
        return dbc.Card([
            make_hideable(
                dbc.CardHeader([
                    html.Div([
                        html.H3(self.title, id='interaction-summary-title-'+self.name),
                        make_hideable(html.H6(self.subtitle, className='card-subtitle'), hide=self.hide_subtitle),
                        dbc.Tooltip(self.description, target='interaction-summary-title-'+self.name),
                    ]), 
                ]), hide=self.hide_title),
            dbc.CardBody([
                dbc.Row([
                    make_hideable(
                        dbc.Col([
                            dbc.Label("Feature", id='interaction-summary-col-label-'+self.name),
                            dbc.Tooltip("Feature to select interactions effects for",
                                    target='interaction-summary-col-label-'+self.name),
                            dbc.Select(id='interaction-summary-col-'+self.name, 
                                options=[{'label': col, 'value': col} 
                                            for col in self.explainer.columns_ranked_by_shap(self.cats)],
                                value=self.col),
                        ], md=2), self.hide_col),
                    make_hideable(
                        dbc.Col([
        
                            dbc.Label("Depth:", id='interaction-summary-depth-label-'+self.name),
                            dbc.Tooltip("Number of interaction features to display",
                                    target='interaction-summary-depth-label-'+self.name),
                            dbc.Select(id='interaction-summary-depth-'+self.name, 
                                options = [{'label': str(i+1), 'value':i+1} 
                                                for i in range(self.explainer.n_features(self.cats)-1)],
                                value=self.depth)
                        ], md=2), self.hide_depth),
                    make_hideable(
                        dbc.Col([
                            dbc.FormGroup(
                                [
                                    dbc.Label("Summary Type", id='interaction-summary-type-label-'+self.name),
                                    dbc.Tooltip("Display mean absolute SHAP value per feature (aggregate)"
                                                " or display every single shap value per feature (detailed)", 
                                                target='interaction-summary-type-label-'+self.name),
                                    dbc.Select(
                                        options=[
                                            {"label": "Aggregate", "value": "aggregate"},
                                            {"label": "Detailed", "value": "detailed"},
                                        ],
                                        value=self.summary_type,
                                        id='interaction-summary-type-'+self.name, 
                                    ),
                                ]
                            )
                        ], md=3), self.hide_type),
                    make_hideable(
                        dbc.Col([
                            dbc.FormGroup([
                                    dbc.Label("Grouping:", id='interaction-summary-group-cats-label-'+self.name),
                                    dbc.Tooltip("Group onehot encoded categorical variables together", 
                                                target='interaction-summary-group-cats-label-'+self.name),
                                    dbc.Checklist(
                                        options=[{"label": "Group cats", "value": True}],
                                        value=[True] if self.cats else [],
                                        id='interaction-summary-group-cats-'+self.name,
                                        inline=True,
                                        switch=True,
                                    ),
                                ]),
                        ],md=2), self.hide_cats),
                    make_hideable(
                        dbc.Col([
                            html.Div([
                                dbc.Label(f"{self.explainer.index_name}:", id='interaction-summary-index-label-'+self.name),
                                dbc.Tooltip(f"Select {self.explainer.index_name} to highlight in plot. "
                                            "You can also select by clicking on a scatter point in the graph.", 
                                            target='interaction-summary-index-label-'+self.name),
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
            ]),
        ])

    def component_callbacks(self, app):
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
            [Input('interaction-summary-group-cats-'+self.name, 'value'),
             Input('pos-label-'+self.name, 'value')])
        def update_interaction_scatter_graph(cats, pos_label):
            depth_options = [{'label': str(i+1), 'value': i+1} 
                                    for i in range(self.explainer.n_features(bool(cats)))]
            new_cols = self.explainer.columns_ranked_by_shap(bool(cats), pos_label=pos_label)
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
             Input('interaction-summary-group-cats-'+self.name, 'value')])
        def update_interaction_scatter_graph(col, depth, summary_type, index, pos_label, cats):
            if col is not None:
                depth = None if depth is None else int(depth)
                if summary_type=='aggregate':
                    plot = self.explainer.plot_interactions(
                        col, topx=depth, cats=bool(cats), pos_label=pos_label)
                    return plot, dict(display="none")
                elif summary_type=='detailed':
                    plot = self.explainer.plot_shap_interaction_summary(
                        col, topx=depth, cats=bool(cats), pos_label=pos_label, index=index)
                return plot, {}
            raise PreventUpdate


class InteractionDependenceComponent(ExplainerComponent):
    def __init__(self, explainer, title="Interaction Dependence", name=None,
                    subtitle="Relation between feature value and shap interaction value",
                    hide_title=False, hide_subtitle=False, hide_cats=False, hide_col=False, 
                    hide_interact_col=False, hide_index=False,
                    hide_selector=False, hide_top=False, hide_bottom=False,
                    pos_label=None, cats=True, col=None, interact_col=None,
                    description=None, index=None, **kwargs):
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
            subtitle (str): subtitle
            hide_title (bool, optional): Hide component title. Defaults to False.
            hide_subtitle (bool, optional): Hide subtitle. Defaults to False.
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
            description (str, optional): Tooltip to display when hover over
                component title. When None default text is shown. 
        """
        super().__init__(explainer, title, name)

        if self.col is None:
            self.col = explainer.columns_ranked_by_shap(cats)[0]
        if self.interact_col is None:
            self.interact_col = explainer.shap_top_interactions(self.col, cats=cats)[1]
        

        self.selector = PosLabelSelector(explainer, name=self.name, pos_label=pos_label)
        if self.description is None: self.description = """
        This plot shows the relation between feature values and shap interaction values.
        This allows you to investigate interactions between features in determining
        the prediction of the model.
        """
        self.register_dependencies("shap_interaction_values", "shap_interaction_values_cats")

    def layout(self):
        return dbc.Card([
            make_hideable(
                dbc.CardHeader([
                    html.Div([
                        html.H3(self.title, id='interaction-dependence-title-'+self.name),
                        make_hideable(html.H6(self.subtitle, className='card-subtitle'), hide=self.hide_subtitle),
                        dbc.Tooltip(self.description, target='interaction-dependence-title-'+self.name),
                    ]), 
                ]), hide=self.hide_title),
            dbc.CardBody([
                dbc.Row([
                    make_hideable(
                            dbc.Col([self.selector.layout()
                        ], width=2), hide=self.hide_selector),
                ]),
                dbc.Row([
                    make_hideable(
                        dbc.Col([
                            dbc.FormGroup([
                                dbc.Label("Grouping:", id='interaction-dependence-group-cats-label-'+self.name),
                                dbc.Tooltip("Group onehot encoded categorical variables together", 
                                            target='interaction-dependence-group-cats-label-'+self.name),
                                dbc.Checklist(
                                    options=[{"label": "Group cats", "value": True}],
                                    value=[True] if self.cats else [],
                                    id='interaction-dependence-group-cats-'+self.name,
                                    inline=True,
                                    switch=True,
                                ),
                            ]),
                        ], md=2), hide=self.hide_cats),
                    make_hideable(
                        dbc.Col([
                            dbc.Label('Feature:', id='interaction-dependence-col-label-'+self.name),
                                dbc.Tooltip("Select feature to display shap interactions for", 
                                            target='interaction-dependence-col-label-'+self.name),
                            dbc.Select(id='interaction-dependence-col-'+self.name,
                                options=[{'label': col, 'value':col} 
                                            for col in self.explainer.columns_ranked_by_shap(self.cats)],
                                value=self.col
                            ),
                        ], md=3), hide=self.hide_col), 
                    make_hideable(
                        dbc.Col([
                            html.Label('Interaction feature:', id='interaction-dependence-interact-col-label-'+self.name),
                                dbc.Tooltip("Select feature to show interaction values for.  Two plots will be shown: "
                                            "both Feature vs Interaction Feature and Interaction Feature vs Feature.", 
                                            target='interaction-dependence-interact-col-label-'+self.name),
                            dbc.Select(id='interaction-dependence-interact-col-'+self.name, 
                                options=[{'label': col, 'value':col} 
                                            for col in self.explainer.shap_top_interactions(col=self.col, cats=self.cats)],
                                value=self.interact_col
                            ),
                        ], md=3), hide=self.hide_interact_col), 
                    make_hideable(
                            dbc.Col([
                                dbc.Label(f"{self.explainer.index_name}:", id='interaction-dependence-index-label-'+self.name),
                                dbc.Tooltip(f"Select {self.explainer.index_name} to highlight in the plot."
                                            "You can also select by clicking on a scatter marker in the accompanying"
                                            " shap interaction summary plot (detailed).", 
                                            target='interaction-dependence-index-label-'+self.name),
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
                    ]),
                ]),
                dbc.Row([
                    dbc.Col([
                        make_hideable(
                        dcc.Loading(id='loading-reverse-interaction-graph-'+self.name, 
                                children=[dcc.Graph(id='interaction-dependence-reverse-graph-'+self.name,
                                                config=dict(modeBarButtons=[['toImage']], displaylogo=False))]),
                                hide=self.hide_bottom),
                    ]),
                ]),
            ]),
        ])

    def component_callbacks(self, app):
        @app.callback(
            Output('interaction-dependence-col-'+self.name, 'options'), 
            [Input('interaction-dependence-group-cats-'+self.name, 'value'),
             Input('pos-label-'+self.name, 'value')])
        def update_interaction_dependence_interact_col(cats, pos_label):
            new_cols = self.explainer.columns_ranked_by_shap(bool(cats), pos_label=pos_label)
            new_col_options = [{'label': col, 'value':col} for col in new_cols]
            return new_col_options

        @app.callback(
            Output('interaction-dependence-interact-col-'+self.name, 'options'),
            [Input('interaction-dependence-col-'+self.name, 'value'),
             Input('pos-label-'+self.name, 'value')],
            [State('interaction-dependence-group-cats-'+self.name, 'value'),
             State('interaction-dependence-interact-col-'+self.name, 'value')])
        def update_interaction_dependence_interact_col(col, pos_label, cats, old_interact_col):
            if col is not None:
                new_interact_cols = self.explainer.shap_top_interactions(
                    col, cats=bool(cats), pos_label=pos_label)
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

    def component_callbacks(self, app):
        @app.callback(
            Output('interaction-dependence-group-cats-'+self.dep_name, 'value'),
            [Input('interaction-summary-group-cats-'+self.sum_name, 'value')])
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
    def __init__(self, explainer, title="Contributions Plot", name=None,
                    subtitle="How has each feature contributed to the prediction?",
                    hide_title=False, hide_subtitle=False, hide_index=False, hide_depth=False, 
                    hide_sort=False, hide_orientation=True, hide_cats=False, 
                    hide_selector=False, feature_input_component=None,
                    pos_label=None, index=None, depth=None, sort='high-to-low', 
                    orientation='vertical', cats=True, higher_is_better=True,
                    description=None, **kwargs):
        """Display Shap contributions to prediction graph component

        Args:
            explainer (Explainer): explainer object constructed , with either
                        ClassifierExplainer() or RegressionExplainer()
            title (str, optional): Title of tab or page. Defaults to 
                        "Contributions Plot".
            name (str, optional): unique name to add to Component elements. 
                        If None then random uuid is generated to make sure 
                        it's unique. Defaults to None.
            subtitle (str): subtitle
            hide_title (bool, optional): Hide component title. Defaults to False.
            hide_subtitle (bool, optional): Hide subtitle. Defaults to False.
            hide_index (bool, optional): Hide index selector. Defaults to False.
            hide_depth (bool, optional): Hide depth toggle. Defaults to False.
            hide_sort (bool, optional): Hide the sorting dropdown. Defaults to False.
            hide_orientation (bool, optional): Hide the orientation dropdown. 
                    Defaults to True.
            hide_cats (bool, optional): Hide group cats toggle. Defaults to False.
            hide_selector (bool, optional): hide pos label selector. Defaults to False.
            feature_input_component (FeatureInputComponent): A FeatureInputComponent
                that will give the input to the graph instead of the index selector.
                If not None, hide_index=True. Defaults to None.
            pos_label ({int, str}, optional): initial pos label. 
                        Defaults to explainer.pos_label
            index ({int, bool}, optional): Initial index to display. Defaults to None.
            depth (int, optional): Initial number of features to display. Defaults to None.
            sort ({'abs', 'high-to-low', 'low-to-high', 'importance'}, optional): sorting of shap values. 
                        Defaults to 'high-to-low'.
            orientation ({'vertical', 'horizontal'}, optional): orientation of bar chart.
                        Defaults to 'vertical'.
            cats (bool, optional): Group cats. Defaults to True.
            higher_is_better (bool, optional): Color positive shap values green and
                negative shap values red, or the reverse. 
            description (str, optional): Tooltip to display when hover over
                component title. When None default text is shown. 
        """
        super().__init__(explainer, title, name)

        self.index_name = 'contributions-graph-index-'+self.name

        if self.depth is not None:
            self.depth = min(self.depth, self.explainer.n_features(self.cats))

        if not self.explainer.cats:
            self.hide_cats = True
        
        if self.feature_input_component is not None:
            self.hide_index = True

        if self.description is None: self.description = """
        This plot shows the contribution that each individual feature has had
        on the prediction for a specific observation. The contributions (starting
        from the population average) add up to the final prediction. This allows you
        to explain exactly how each individual prediction has been built up
        from all the individual ingredients in the model.
        """

        self.selector = PosLabelSelector(explainer, name=self.name, pos_label=pos_label)
        self.register_dependencies('shap_values', 'shap_values_cats')

    def layout(self):
        return dbc.Card([
            make_hideable(
                dbc.CardHeader([
                    html.Div([
                        html.H3(self.title, id='contributions-graph-title-'+self.name),
                        make_hideable(html.H6(self.subtitle, className='card-subtitle'), hide=self.hide_subtitle),
                        dbc.Tooltip(self.description, target='contributions-graph-title-'+self.name),
                    ]), 
                ]), hide=self.hide_title),
            dbc.CardBody([
                dbc.Row([
                    make_hideable(
                            dbc.Col([self.selector.layout()
                        ], md=2), hide=self.hide_selector),
                ], justify="right"),
                dbc.Row([
                    make_hideable(
                        dbc.Col([
                            dbc.Label(f"{self.explainer.index_name}:", id='contributions-graph-index-label-'+self.name),
                            dbc.Tooltip(f"Select the {self.explainer.index_name} to display the feature contributions for", 
                                        target='contributions-graph-index-label-'+self.name),
                            dcc.Dropdown(id='contributions-graph-index-'+self.name, 
                                options = [{'label': str(idx), 'value':idx} 
                                                for idx in self.explainer.idxs],
                                value=self.index)
                        ], md=4), hide=self.hide_index), 
                    make_hideable(
                        dbc.Col([
                            dbc.Label("Depth:", id='contributions-graph-depth-label-'+self.name),
                            dbc.Tooltip("Number of features to display",
                                    target='contributions-graph-depth-label-'+self.name),
                            dbc.Select(id='contributions-graph-depth-'+self.name, 
                                options = [{'label': str(i+1), 'value':i+1} 
                                                for i in range(self.explainer.n_features(self.cats))],
                                value=None if self.depth is None else str(self.depth))
                        ], md=2), hide=self.hide_depth),
                    make_hideable(
                        dbc.Col([
                            dbc.Label("Sorting:", id='contributions-graph-sorting-label-'+self.name),
                            dbc.Tooltip("Sort the features either by highest absolute (positive or negative) impact (absolute), "
                                        "from most positive the most negative (high-to-low)"
                                        "from most negative to most positive (low-to-high or "
                                        "according the global feature importance ordering (importance).",
                                    target='contributions-graph-sorting-label-'+self.name),
                            dbc.Select(id='contributions-graph-sorting-'+self.name, 
                                options = [{'label': 'Absolute', 'value': 'abs'},
                                            {'label': 'High to Low', 'value': 'high-to-low'},
                                            {'label': 'Low to High', 'value': 'low-to-high'},
                                            {'label': 'Importance', 'value': 'importance'}],
                                value=self.sort)
                        ], md=2), hide=self.hide_sort),
                    make_hideable(
                        dbc.Col([
                            dbc.Label("Orientation:", id='contributions-graph-orientation-label-'+self.name),
                            dbc.Tooltip("Show vertical bars left to right or horizontal bars from top to bottom",
                                    target='contributions-graph-orientation-label-'+self.name),
                            dbc.Select(id='contributions-graph-orientation-'+self.name, 
                                options = [{'label': 'Vertical', 'value': 'vertical'},
                                            {'label': 'Horizontal', 'value': 'horizontal'}],
                                value=self.orientation)
                        ], md=2), hide=self.hide_orientation),
                    make_hideable(
                        dbc.Col([
                            dbc.FormGroup([
                                dbc.Label("Grouping:", id='contributions-graph-group-cats-label-'+self.name),
                                dbc.Tooltip("Group onehot encoded categorical variables together", 
                                            target='contributions-graph-group-cats-label-'+self.name),
                                dbc.Checklist(
                                    options=[{"label": "Group cats", "value": True}],
                                    value=[True] if self.cats else [],
                                    id='contributions-graph-group-cats-'+self.name,
                                    inline=True,
                                    switch=True,
                                ),
                            ]),
                        ], md=2), hide=self.hide_cats),
                    ], form=True),
                    
                dbc.Row([
                    dbc.Col([
                        dcc.Loading(id='loading-contributions-graph-'+self.name, 
                            children=[dcc.Graph(id='contributions-graph-'+self.name,
                                    config=dict(modeBarButtons=[['toImage']], displaylogo=False))]),
                    ]),
                ]),
            ]),
        ])
        
    def component_callbacks(self, app):
        
        if self.feature_input_component is None:
            @app.callback(
                [Output('contributions-graph-'+self.name, 'figure'),
                 Output('contributions-graph-depth-'+self.name, 'options')],
                [Input('contributions-graph-index-'+self.name, 'value'),
                 Input('contributions-graph-depth-'+self.name, 'value'),
                 Input('contributions-graph-sorting-'+self.name, 'value'),
                 Input('contributions-graph-orientation-'+self.name, 'value'),
                 Input('contributions-graph-group-cats-'+self.name, 'value'),
                 Input('pos-label-'+self.name, 'value')])
            def update_output_div(index, depth, sort, orientation, cats, pos_label):
                if index is None:
                    raise PreventUpdate
                depth = None if depth is None else int(depth)
                plot = self.explainer.plot_shap_contributions(index, topx=depth, 
                            cats=bool(cats), sort=sort, orientation=orientation, 
                            pos_label=pos_label, higher_is_better=self.higher_is_better)

                ctx = dash.callback_context
                trigger = ctx.triggered[0]['prop_id'].split('.')[0]
                if trigger == 'contributions-graph-group-cats-'+self.name:
                    depth_options = [{'label': str(i+1), 'value': i+1} 
                                            for i in range(self.explainer.n_features(bool(cats)))]
                    return (plot, depth_options)
                else:
                    return (plot, dash.no_update)
        else:
            @app.callback(
                [Output('contributions-graph-'+self.name, 'figure'),
                 Output('contributions-graph-depth-'+self.name, 'options')],
                [Input('contributions-graph-depth-'+self.name, 'value'),
                 Input('contributions-graph-sorting-'+self.name, 'value'),
                 Input('contributions-graph-orientation-'+self.name, 'value'),
                 Input('contributions-graph-group-cats-'+self.name, 'value'),
                 Input('pos-label-'+self.name, 'value'),
                 *self.feature_input_component._feature_callback_inputs])
            def update_output_div(depth, sort, orientation, cats, pos_label, *inputs):
                depth = None if depth is None else int(depth)
                X_row = self.explainer.get_row_from_input(inputs, ranked_by_shap=True)
                plot = self.explainer.plot_shap_contributions(X_row=X_row, 
                            topx=depth, cats=bool(cats), sort=sort, orientation=orientation, 
                            pos_label=pos_label, higher_is_better=self.higher_is_better)

                ctx = dash.callback_context
                trigger = ctx.triggered[0]['prop_id'].split('.')[0]
                if trigger == 'contributions-graph-group-cats-'+self.name:
                    depth_options = [{'label': str(i+1), 'value': i+1} 
                                            for i in range(self.explainer.n_features(bool(cats)))]
                    return (plot, depth_options)
                else:
                    return (plot, dash.no_update)


class ShapContributionsTableComponent(ExplainerComponent):
    def __init__(self, explainer, title="Contributions Table", name=None,
                    subtitle="How has each feature contributed to the prediction?",
                    hide_title=False, hide_subtitle=False, hide_index=False, 
                    hide_depth=False, hide_sort=False, hide_cats=False, 
                    hide_selector=False, feature_input_component=None,
                    pos_label=None, index=None, depth=None, sort='abs', cats=True, 
                    description=None, **kwargs):
        """Show SHAP values contributions to prediction in a table component

        Args:
            explainer (Explainer): explainer object constructed with either
                        ClassifierExplainer() or RegressionExplainer()
            title (str, optional): Title of tab or page. Defaults to 
                        "Contributions Table".
            name (str, optional): unique name to add to Component elements. 
                        If None then random uuid is generated to make sure 
                        it's unique. Defaults to None.
            subtitle (str): subtitle
            hide_title (bool, optional): Hide component title. Defaults to False.
            hide_subtitle (bool, optional): Hide subtitle. Defaults to False.
            hide_index (bool, optional): Hide index selector. Defaults to False.
            hide_depth (bool, optional): Hide depth selector. Defaults to False.
            hide_sort (bool, optional): Hide sorting dropdown. Default to False.
            hide_cats (bool, optional): Hide group cats toggle. Defaults to False.
            hide_selector (bool, optional): hide pos label selector. Defaults to False.
            feature_input_component (FeatureInputComponent): A FeatureInputComponent
                that will give the input to the graph instead of the index selector.
                If not None, hide_index=True. Defaults to None.
            pos_label ({int, str}, optional): initial pos label. 
                        Defaults to explainer.pos_label
            index ([type], optional): Initial index to display. Defaults to None.
            depth ([type], optional): Initial number of features to display. Defaults to None.
            sort ({'abs', 'high-to-low', 'low-to-high', 'importance'}, optional): sorting of shap values. 
                        Defaults to 'high-to-low'.
            cats (bool, optional): Group categoricals. Defaults to True.
            description (str, optional): Tooltip to display when hover over
                component title. When None default text is shown. 
        """
        super().__init__(explainer, title, name)

        self.index_name = 'contributions-table-index-'+self.name

        if self.depth is not None:
            self.depth = min(self.depth, self.explainer.n_features(self.cats))
        
        if not self.explainer.cats:
            self.hide_cats = True

        if self.feature_input_component is not None:
            self.hide_index = True

        if self.description is None: self.description = """
        This tables shows the contribution that each individual feature has had
        on the prediction for a specific observation. The contributions (starting
        from the population average) add up to the final prediction. This allows you
        to explain exactly how each individual prediction has been built up
        from all the individual ingredients in the model.
        """
        self.selector = PosLabelSelector(explainer, name=self.name, pos_label=pos_label)
        self.register_dependencies('shap_values', 'shap_values_cats')

    def layout(self):
        return dbc.Card([
            make_hideable(
                dbc.CardHeader([
                    html.Div([
                        html.H3(self.title, id='contributions-table-title-'+self.name),
                        make_hideable(html.H6(self.subtitle, className='card-subtitle'), hide=self.hide_subtitle),
                        dbc.Tooltip(self.description, target='contributions-table-title-'+self.name),
                    ]), 
                ]), hide=self.hide_title),
            dbc.CardBody([
                dbc.Row([
                    make_hideable(
                        dbc.Col([
                            dbc.Label(f"{self.explainer.index_name}:", id='contributions-table-index-label-'+self.name),
                            dbc.Tooltip(f"Select the {self.explainer.index_name} to display the feature contributions for", 
                                        target='contributions-table-index-label-'+self.name),
                            dcc.Dropdown(id='contributions-table-index-'+self.name, 
                                options = [{'label': str(idx), 'value':idx} 
                                                for idx in self.explainer.idxs],
                                value=self.index)
                        ], md=4), hide=self.hide_index), 
                    make_hideable(
                        dbc.Col([
                            dbc.Label("Depth:", id='contributions-table-depth-label-'+self.name),
                            dbc.Tooltip("Number of features to display",
                                    target='contributions-table-depth-label-'+self.name),
                            dbc.Select(id='contributions-table-depth-'+self.name, 
                                options = [{'label': str(i+1), 'value':i+1} 
                                                for i in range(self.explainer.n_features(self.cats))],
                                value=self.depth)
                        ], md=2), hide=self.hide_depth),
                    make_hideable(
                        dbc.Col([
                            dbc.Label("Sorting:", id='contributions-table-sorting-label-'+self.name),
                            dbc.Tooltip("Sort the features either by highest absolute (positive or negative) impact (absolute), "
                                        "from most positive the most negative (high-to-low)"
                                        "from most negative to most positive (low-to-high or "
                                        "according the global feature importance ordering (importance).",
                                    target='contributions-table-sorting-label-'+self.name),
                            dbc.Select(id='contributions-table-sorting-'+self.name, 
                                options = [{'label': 'Absolute', 'value': 'abs'},
                                            {'label': 'High to Low', 'value': 'high-to-low'},
                                            {'label': 'Low to High', 'value': 'low-to-high'},
                                            {'label': 'Importance', 'value': 'importance'}],
                                value=self.sort)
                        ], md=2), hide=self.hide_sort),
                    make_hideable(
                            dbc.Col([self.selector.layout()
                        ], width=2), hide=self.hide_selector),
                    make_hideable(
                        dbc.Col([
                            dbc.FormGroup([
                                dbc.Label("Grouping:", id='contributions-table-group-cats-label-'+self.name),
                                dbc.Tooltip("Group onehot encoded categorical variables together", 
                                            target='contributions-table-group-cats-label-'+self.name),
                                dbc.Checklist(
                                    options=[{"label": "Group cats", "value": True}],
                                    value=[True] if self.cats else [],
                                    id='contributions-table-group-cats-'+self.name,
                                    inline=True,
                                    switch=True,
                                ),
                            ]),
                        ], md=3), hide=self.hide_cats),
                ], form=True),
                dbc.Row([
                    dbc.Col([
                        html.Div(id='contributions-table-'+self.name)
                    ]),
                ]),
            ]),   
        ])
        
    def component_callbacks(self, app):
        if self.feature_input_component is None:
            @app.callback(
                [Output('contributions-table-'+self.name, 'children'),
                Output('contributions-table-depth-'+self.name, 'options')],
                [Input('contributions-table-index-'+self.name, 'value'),
                Input('contributions-table-depth-'+self.name, 'value'),
                Input('contributions-table-sorting-'+self.name, 'value'),
                Input('contributions-table-group-cats-'+self.name, 'value'),
                Input('pos-label-'+self.name, 'value')])
            def update_output_div(index, depth, sort, cats, pos_label):
                if index is None:
                    raise PreventUpdate
                depth = None if depth is None else int(depth)
                contributions_table = dbc.Table.from_dataframe(
                    self.explainer.contrib_summary_df(index, cats=bool(cats), topx=depth, 
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
                                            for i in range(self.explainer.n_features(bool(cats)))]
                    return (output_div, depth_options)
                else:
                    return (output_div, dash.no_update) 
        else:
            @app.callback(
                [Output('contributions-table-'+self.name, 'children'),
                 Output('contributions-table-depth-'+self.name, 'options')],
                [Input('contributions-table-depth-'+self.name, 'value'),
                 Input('contributions-table-sorting-'+self.name, 'value'),
                 Input('contributions-table-group-cats-'+self.name, 'value'),
                 Input('pos-label-'+self.name, 'value'),
                 *self.feature_input_component._feature_callback_inputs])
            def update_output_div(depth, sort, cats, pos_label, *inputs):
                X_row = self.explainer.get_row_from_input(inputs, ranked_by_shap=True)
                depth = None if depth is None else int(depth)
                contributions_table = dbc.Table.from_dataframe(
                    self.explainer.contrib_summary_df(X_row=X_row, cats=bool(cats), topx=depth, 
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
                                            for i in range(self.explainer.n_features(bool(cats)))]
                    return (output_div, depth_options)
                else:
                    return (output_div, dash.no_update) 


