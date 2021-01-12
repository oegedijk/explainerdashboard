__all__ = [
    'PredictionSummaryComponent',
    'ImportancesComponent',
    'FeatureInputComponent',
    'PdpComponent',
]
from math import ceil

import numpy as np
import pandas as pd

import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_table

from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from ..dashboard_methods import *

class PredictionSummaryComponent(ExplainerComponent):
    def __init__(self, explainer, title="Prediction Summary", name=None,
                    hide_index=False, hide_percentile=False, 
                    hide_title=False,  hide_subtitle=False, hide_selector=False,
                    pos_label=None, index=None, percentile=True,
                    description=None, **kwargs):
        """Shows a summary for a particular prediction

        Args:
            explainer (Explainer): explainer object constructed with either
                        ClassifierExplainer() or RegressionExplainer()
            title (str, optional): Title of tab or page. Defaults to 
                        "Prediction Summary".
            name (str, optional): unique name to add to Component elements. 
                        If None then random uuid is generated to make sure 
                        it's unique. Defaults to None.
            hide_index (bool, optional): hide index selector. Defaults to False.
            hide_percentile (bool, optional): hide percentile toggle. Defaults to False.
            hide_title (bool, optional): hide title. Defaults to False.
            hide_subtitle (bool, optional): Hide subtitle. Defaults to False.
            hide_selector (bool, optional): hide pos label selectors. Defaults to False.
            pos_label ({int, str}, optional): initial pos label. 
                        Defaults to explainer.pos_label
            index ({int, str}, optional): Index to display prediction summary for. Defaults to None.
            percentile (bool, optional): Whether to add the prediction percentile. Defaults to True.

        """
        super().__init__(explainer, title, name)

        self.index_name = 'modelprediction-index-'+self.name
        self.selector = PosLabelSelector(explainer, name=self.name, pos_label=pos_label)

    def layout(self):
        return dbc.Card([
            make_hideable(
                dbc.CardHeader([
                    html.H3(self.title), 
                ]), hide=self.hide_title), 
            dbc.CardBody([
                dbc.Row([
                    make_hideable(
                        dbc.Col([
                            dbc.Label(f"{self.explainer.index_name}:"),
                            dcc.Dropdown(id='modelprediction-index-'+self.name, 
                                    options = [{'label': str(idx), 'value':idx} 
                                                    for idx in self.explainer.idxs],
                                    value=self.index)
                        ], md=6), hide=self.hide_index),
                    make_hideable(
                            dbc.Col([self.selector.layout()
                        ], width=3), hide=self.hide_selector),
                    make_hideable(
                        dbc.Col([
                            dbc.Label("Show Percentile:"),
                            dbc.FormGroup(
                            [
                                dbc.RadioButton(
                                    id='modelprediction-percentile-'+self.name, 
                                    className="form-check-input",
                                    checked=self.percentile),
                                dbc.Label("Show percentile",
                                        html_for='modelprediction-percentile'+self.name, 
                                        className="form-check-label"),
                            ], check=True)
                        ], md=3), hide=self.hide_percentile),
                    
                ]),
                dbc.Row([
                    dbc.Col([
                        dcc.Markdown(id='modelprediction-'+self.name),    
                    ], md=12)
                ])
            ])
        ])

    def component_callbacks(self, app):
        @app.callback(
            Output('modelprediction-'+self.name, 'children'),
            [Input('modelprediction-index-'+self.name, 'value'),
             Input('modelprediction-percentile-'+self.name, 'checked'),
             Input('pos-label-'+self.name, 'value')])
        def update_output_div(index, include_percentile, pos_label):
            if index is not None:
                return self.explainer.prediction_result_markdown(index, include_percentile=include_percentile, pos_label=pos_label)
            raise PreventUpdate


class ImportancesComponent(ExplainerComponent):
    def __init__(self, explainer, title="Feature Importances", name=None,
                        subtitle="Which features had the biggest impact?",
                        hide_type=False, hide_depth=False, hide_cats=False,
                        hide_title=False,  hide_subtitle=False, hide_selector=False,
                        pos_label=None, importance_type="shap", depth=None, 
                        cats=True, no_permutations=False,
                        description=None, **kwargs):
        """Display features importances component

        Args:
            explainer (Explainer): explainer object constructed with either
                        ClassifierExplainer() or RegressionExplainer()
            title (str, optional): Title of tab or page. Defaults to 
                        "Feature Importances".
            name (str, optional): unique name to add to Component elements. 
                        If None then random uuid is generated to make sure 
                        it's unique. Defaults to None.
            subtitle(str, optional): Subtitle.
            hide_type (bool, optional): Hide permutation/shap selector toggle. 
                        Defaults to False.
            hide_depth (bool, optional): Hide number of features toggle. 
                        Defaults to False.
            hide_cats (bool, optional): Hide group cats toggle. 
                        Defaults to False.
            hide_title (bool, optional): hide title. Defaults to False.
            hide_subtitle (bool, optional): Hide subtitle. Defaults to False.
            hide_selector (bool, optional): hide pos label selectors. 
                        Defaults to False.
            pos_label ({int, str}, optional): initial pos label. 
                        Defaults to explainer.pos_label
            importance_type (str, {'permutation', 'shap'} optional): 
                        initial importance type to display. Defaults to "shap".
            depth (int, optional): Initial number of top features to display. 
                        Defaults to None (=show all).
            cats (bool, optional): Group categoricals. Defaults to True.
            no_permutations (bool, optional): Do not use the permutation
                importances for this component. Defaults to False. 
            description (str, optional): Tooltip to display when hover over
                component title. When None default text is shown. 
        """
        super().__init__(explainer, title, name)

        if not self.explainer.onehot_cols:
            self.hide_cats = True

        assert importance_type in ['shap', 'permutation'], \
            "importance type must be either 'shap' or 'permutation'!"

        if depth is not None:
            self.depth = min(depth, len(explainer.columns_ranked_by_shap(cats)))

        self.selector = PosLabelSelector(explainer, name=self.name, pos_label=pos_label)

        if self.explainer.y_missing or self.no_permutations:
            self.hide_type = True
            self.importance_type = 'shap'
        if self.description is None: self.description = f"""
        Shows the features sorted from most important to least important. Can 
        be either sorted by absolute SHAP value (average absolute impact of 
        the feature on final prediction) or by permutation importance (how much
        does the model get worse when you shuffle this feature, rendering it
        useless?).
        """
        self.register_dependencies('shap_values', 'shap_values_cats')
        if not (self.hide_type and self.importance_type == 'shap'):
            self.register_dependencies('permutation_importances', 'permutation_importances_cats')

    def layout(self):
        return dbc.Card([
            make_hideable(
                dbc.CardHeader([
                    html.Div([
                        html.H3(self.title, className="card-title", id='importances-title-'+self.name),
                        make_hideable(html.H6(self.subtitle, className='card-subtitle'), hide=self.hide_subtitle),
                        dbc.Tooltip(self.description, target='importances-title-'+self.name),
                    ]),
            ]), hide=self.hide_title),
            dbc.CardBody([
                dbc.Row([
                    make_hideable(
                        dbc.Col([
                            dbc.FormGroup([
                                dbc.Label("Importances type:"),
                                dbc.Select(
                                    options=[
                                        {'label': 'Permutation Importances', 
                                        'value': 'permutation'},
                                        {'label': 'SHAP values', 
                                        'value': 'shap'}
                                    ],
                                    value=self.importance_type,
                                    id='importances-permutation-or-shap-'+self.name,
                                    #inline=True,
                                ),
                                
                            ], id='importances-permutation-or-shap-form-'+self.name),
                            dbc.Tooltip("Select Feature importance type: \n"
                                    "Permutation Importance: How much does performance metric decrease when shuffling this feature?\n"
                                    "SHAP values: What is the average SHAP contribution (positive or negative) of this feature?",
                                    target='importances-permutation-or-shap-form-'+self.name),
                        ], md=3), self.hide_type),
                    make_hideable(
                        dbc.Col([
                            html.Label('Depth:', id='importances-depth-label-'+self.name),
                            dbc.Select(id='importances-depth-'+self.name,
                                        options = [{'label': str(i+1), 'value':i+1} 
                                                    for i in range(self.explainer.n_features(self.cats))],
                                        value=self.depth),
                            dbc.Tooltip("Select how many features to display", target='importances-depth-label-'+self.name)
                        ], md=2), self.hide_depth),
                    make_hideable(
                        dbc.Col([
                            dbc.FormGroup([
                                dbc.Label("Grouping:", id='importances-group-cats-label-'+self.name),
                                dbc.Tooltip("Group onehot encoded categorical variables together", 
                                            target='importances-group-cats-label-'+self.name),
                                dbc.Checklist(
                                    options=[
                                        {"label": "Group cats", "value": True},
                                    ],
                                    value=[True] if self.cats else [],
                                    id='importances-group-cats-'+self.name,
                                    inline=True,
                                    switch=True,
                                ),
                            ]),
                        ]),  self.hide_cats),    
                    make_hideable(
                            dbc.Col([self.selector.layout()
                        ], width=2), hide=self.hide_selector)    
                ], form=True),
                dbc.Row([
                    dbc.Col([
                        dcc.Loading(id='importances-graph-loading-'+self.name,
                            children=dcc.Graph(id='importances-graph-'+self.name,
                                        config=dict(modeBarButtons=[['toImage']], displaylogo=False))),
                    ]),
                ]), 
            ])         
        ])
        
    def component_callbacks(self, app, **kwargs):
        @app.callback(  
            [Output('importances-graph-'+self.name, 'figure'),
             Output('importances-depth-'+self.name, 'options')],
            [Input('importances-depth-'+self.name, 'value'),
             Input('importances-group-cats-'+self.name, 'value'),
             Input('importances-permutation-or-shap-'+self.name, 'value'),
             Input('pos-label-'+self.name, 'value')],
        )
        def update_importances(depth, cats, permutation_shap, pos_label):
            depth = None if depth is None else int(depth)
            plot =  self.explainer.plot_importances(
                        kind=permutation_shap, topx=depth, 
                        cats=bool(cats), pos_label=pos_label)
            trigger = dash.callback_context.triggered[0]['prop_id'].split('.')[0]
            if trigger == 'importances-group-cats-'+self.name:
                depth_options = [{'label': str(i+1), 'value': i+1} 
                                        for i in range(self.explainer.n_features(bool(cats)))]
                return (plot, depth_options)
            else:
                return (plot, dash.no_update)


class PdpComponent(ExplainerComponent):
    def __init__(self, explainer, title="Partial Dependence Plot", name=None,
                    subtitle="How does the prediction change if you change one feature?",
                    hide_col=False, hide_index=False, hide_cats=False,
                    hide_title=False,  hide_subtitle=False, 
                    hide_footer=False, hide_selector=False,
                    hide_dropna=False, hide_sample=False, 
                    hide_gridlines=False, hide_gridpoints=False, hide_cats_sort=False,
                    feature_input_component=None,
                    pos_label=None, col=None, index=None, cats=True,
                    dropna=True, sample=100, gridlines=50, gridpoints=10,
                    cats_sort='freq',
                    description=None, **kwargs):
        """Show Partial Dependence Plot component

        Args:
            explainer (Explainer): explainer object constructed with either
                        ClassifierExplainer() or RegressionExplainer()
            title (str, optional): Title of tab or page. Defaults to 
                        "Partial Dependence Plot".
            name (str, optional): unique name to add to Component elements. 
                        If None then random uuid is generated to make sure 
                        it's unique. Defaults to None.
            subtitle (str): subtitle
            hide_col (bool, optional): Hide feature selector. Defaults to False.
            hide_index (bool, optional): Hide index selector. Defaults to False.
            hide_cats (bool, optional): Hide group cats toggle. Defaults to False.
            hide_title (bool, optional): Hide title, Defaults to False.
            hide_subtitle (bool, optional): Hide subtitle. Defaults to False.
            hide_footer (bool, optional): hide the footer at the bottom of the component
            hide_selector (bool, optional): hide pos label selectors. Defaults to False.
            hide_dropna (bool, optional): Hide drop na's toggle Defaults to False.
            hide_sample (bool, optional): Hide sample size input. Defaults to False.
            hide_gridlines (bool, optional): Hide gridlines input. Defaults to False.
            hide_gridpoints (bool, optional): Hide gridpounts input. Defaults to False.
            hide_cats_sort (bool, optional): Hide the categorical sorting dropdown. Defaults to False.
            feature_input_component (FeatureInputComponent): A FeatureInputComponent
                that will give the input to the graph instead of the index selector.
                If not None, hide_index=True. Defaults to None.
            pos_label ({int, str}, optional): initial pos label. 
                        Defaults to explainer.pos_label
            col (str, optional): Feature to display PDP for. Defaults to None.
            index ({int, str}, optional): Index to add ice line to plot. Defaults to None.
            cats (bool, optional): Group categoricals for feature selector. Defaults to True.
            dropna (bool, optional): Drop rows where values equal explainer.na_fill (usually -999). Defaults to True.
            sample (int, optional): Sample size to calculate average partial dependence. Defaults to 100.
            gridlines (int, optional): Number of ice lines to display in plot. Defaults to 50.
            gridpoints (int, optional): Number of breakpoints on horizontal axis Defaults to 10.
            cats_sort (str, optional): how to sort categories: 'alphabet', 
                'freq' or 'shap'. Defaults to 'freq'.
            description (str, optional): Tooltip to display when hover over
                component title. When None default text is shown. 
        """
        super().__init__(explainer, title, name)

        self.index_name = 'pdp-index-'+self.name

        if self.col is None:
            self.col = self.explainer.columns_ranked_by_shap(self.cats)[0]

        if not self.explainer.onehot_cols:
            self.hide_cats = True
            
        if self.feature_input_component is not None:
            self.exclude_callbacks(self.feature_input_component)
            self.hide_index = True
            
        if self.description is None: self.description = f"""
        The partial dependence plot (pdp) show how the model prediction would
        change if you change one particular feature. The plot shows you a sample
        of observations and how these observations would change with this
        feature (gridlines). The average effect is shown in grey. The effect
        of changing the feature for a single {self.explainer.index_name} is
        shown in blue. You can adjust how many observations to sample for the 
        average, how many gridlines to show, and how many points along the
        x-axis to calculate model predictions for (gridpoints).
        """
        self.selector = PosLabelSelector(explainer, name=self.name, pos_label=pos_label)

    def layout(self):
        return dbc.Card([
                make_hideable(
                    dbc.CardHeader([
                            html.Div([
                                html.H3(self.title, id='pdp-title-'+self.name),
                                make_hideable(html.H6(self.subtitle, className='card-subtitle'), hide=self.hide_subtitle),
                                dbc.Tooltip(self.description, target='pdp-title-'+self.name),
                            ]), 
                    ]), hide=self.hide_title),
                dbc.CardBody([
                    dbc.Row([
                        make_hideable(
                            dbc.Col([
                                dbc.Label("Feature:", 
                                        html_for='pdp-col'+self.name, id='pdp-col-label-'+self.name),
                                dbc.Tooltip("Select the feature for which you want to see the partial dependence plot", 
                                            target='pdp-col-label-'+self.name),
                                dbc.Select(id='pdp-col-'+self.name,       
                                    options=[{'label': col, 'value':col} 
                                                for col in self.explainer.columns_ranked_by_shap(self.cats)],
                                    value=self.col),
                            ], md=4), hide=self.hide_col),
                        make_hideable(
                            dbc.Col([
                                dbc.Label(f"{self.explainer.index_name}:", id='pdp-index-label-'+self.name),
                                dbc.Tooltip(f"Select the {self.explainer.index_name} to display the partial dependence plot for", 
                                        target='pdp-index-label-'+self.name),
                                dcc.Dropdown(id='pdp-index-'+self.name, 
                                    options = [{'label': str(idx), 'value':idx} 
                                                    for idx in self.explainer.idxs],
                                    value=self.index)
                            ], md=4), hide=self.hide_index), 
                        make_hideable(
                            dbc.Col([self.selector.layout()
                        ], width=2), hide=self.hide_selector),
                        make_hideable(
                            dbc.Col([
                                dbc.FormGroup([
                                    dbc.Label("Grouping:", id='pdp-group-cats-label-'+self.name),
                                    dbc.Tooltip("Group onehot encoded categorical variables together", 
                                                target='pdp-group-cats-label-'+self.name),
                                    dbc.Checklist(
                                        options=[
                                            {"label": "Group cats", "value": True},
                                        ],
                                        value=[True] if self.cats else [],
                                        id='pdp-group-cats-'+self.name,
                                        inline=True,
                                        switch=True,
                                    ),
                                ]),
                            ], md=2), hide=self.hide_cats),
                    ], form=True),
                    dbc.Row([
                        dbc.Col([
                            dcc.Loading(id='loading-pdp-graph-'+self.name, 
                                children=[dcc.Graph(id='pdp-graph-'+self.name,
                                                    config=dict(modeBarButtons=[['toImage']], displaylogo=False))]),
                        ])
                    ]),
                ]),
                make_hideable(
                dbc.CardFooter([
                    dbc.Row([
                        make_hideable(
                            dbc.Col([
                                dbc.FormGroup([
                                    dbc.Label("Drop fill:"),
                                    dbc.Tooltip("Drop all observations with feature values "
                                            f"equal to {self.explainer.na_fill} from the plot. "
                                            "This prevents the filler values from ruining the x-axis.", 
                                                target='pdp-dropna-'+self.name),
                                    dbc.Checklist(
                                        options=[{"label": "Drop na_fill", "value": True}],
                                        value=[True] if self.dropna else [],
                                        id='pdp-dropna-'+self.name,
                                        inline=True,
                                        switch=True,
                                    ),
                                ]),
                            ]), hide=self.hide_dropna),
                        make_hideable(
                            dbc.Col([ 
                                dbc.Label("Sample:", id='pdp-sample-label-'+self.name ),
                                dbc.Tooltip("Number of observations to use to calculate average partial dependence", 
                                            target='pdp-sample-label-'+self.name ),
                                dbc.Input(id='pdp-sample-'+self.name, value=self.sample,
                                    type="number", min=0, max=len(self.explainer), step=1),
                            ]), hide=self.hide_sample),  
                        make_hideable(   
                            dbc.Col([ #gridlines
                                dbc.Label("Gridlines:", id='pdp-gridlines-label-'+self.name ),
                                dbc.Tooltip("Number of individual observations' partial dependences to show in plot", 
                                            target='pdp-gridlines-label-'+self.name),
                                dbc.Input(id='pdp-gridlines-'+self.name, value=self.gridlines,
                                        type="number", min=0, max=len(self.explainer), step=1),
                            ]), hide=self.hide_gridlines),
                        make_hideable(
                            dbc.Col([ #gridpoints
                                dbc.Label("Gridpoints:", id='pdp-gridpoints-label-'+self.name ),
                                dbc.Tooltip("Number of points to sample the feature axis for predictions."
                                            " The higher, the smoother the curve, but takes longer to calculate", 
                                            target='pdp-gridpoints-label-'+self.name ),
                                dbc.Input(id='pdp-gridpoints-'+self.name, value=self.gridpoints,
                                    type="number", min=0, max=100, step=1),
                            ]), hide=self.hide_gridpoints),
                        make_hideable(
                            html.Div([
                                dbc.Col([
                                        html.Label('Sort categories:', id='pdp-categories-sort-label-'+self.name),
                                        dbc.Tooltip("How to sort the categories: alphabetically, most common "
                                                    "first (Frequency), or highest mean absolute SHAP value first (Shap impact)", 
                                                    target='pdp-categories-sort-label-'+self.name),
                                        dbc.Select(id='pdp-categories-sort-'+self.name,
                                                options = [{'label': 'Alphabetically', 'value': 'alphabet'},
                                                            {'label': 'Frequency', 'value': 'freq'},
                                                            {'label': 'Shap impact', 'value': 'shap'}],
                                                value=self.cats_sort),
                                    ])], 
                                id='pdp-categories-sort-div-'+self.name,
                                style={} if self.col in self.explainer.cat_cols else dict(display="none")
                            ), hide=self.hide_cats_sort),
                    ], form=True),
                ]), hide=self.hide_footer)
        ])
                
    def component_callbacks(self, app):

        @app.callback(
            Output('pdp-categories-sort-div-'+self.name, 'style'),
            Input('pdp-col-'+self.name, 'value')
        )
        def update_pdp_sort_div(col):
            return {} if col in self.explainer.cat_cols else dict(display="none")
        
        @app.callback(
            Output('pdp-col-'+self.name, 'options'),
            [Input('pdp-group-cats-'+self.name, 'value')],
            [State('pos-label-'+self.name, 'value')]
        )
        def update_pdp_graph(cats, pos_label):
            col_options = [{'label': col, 'value':col} 
                                for col in self.explainer.columns_ranked_by_shap(bool(cats), pos_label=pos_label)]
            return col_options
        
        if self.feature_input_component is None:
            @app.callback(
                Output('pdp-graph-'+self.name, 'figure'),
                [Input('pdp-index-'+self.name, 'value'),
                 Input('pdp-col-'+self.name, 'value'),
                 Input('pdp-dropna-'+self.name, 'value'),
                 Input('pdp-sample-'+self.name, 'value'),
                 Input('pdp-gridlines-'+self.name, 'value'),
                 Input('pdp-gridpoints-'+self.name, 'value'),
                 Input('pdp-categories-sort-'+self.name, 'value'),
                 Input('pos-label-'+self.name, 'value')]
            )
            def update_pdp_graph(index, col, drop_na, sample, gridlines, gridpoints, sort, pos_label):
                return self.explainer.plot_pdp(col, index, 
                    drop_na=bool(drop_na), sample=sample, gridlines=gridlines, 
                    gridpoints=gridpoints, sort=sort, pos_label=pos_label)
        else:
            @app.callback(
                Output('pdp-graph-'+self.name, 'figure'),
                [Input('pdp-col-'+self.name, 'value'),
                 Input('pdp-dropna-'+self.name, 'value'),
                 Input('pdp-sample-'+self.name, 'value'),
                 Input('pdp-gridlines-'+self.name, 'value'),
                 Input('pdp-gridpoints-'+self.name, 'value'),
                 Input('pdp-categories-sort-'+self.name, 'value'),
                 Input('pos-label-'+self.name, 'value'),
                 *self.feature_input_component._feature_callback_inputs]
            )
            def update_pdp_graph(col, drop_na, sample, gridlines, gridpoints, sort, pos_label, *inputs):
                X_row = self.explainer.get_row_from_input(inputs, ranked_by_shap=True)
                return self.explainer.plot_pdp(col, X_row=X_row,
                    drop_na=bool(drop_na), sample=sample, gridlines=gridlines, 
                    gridpoints=gridpoints, sort=sort, pos_label=pos_label)


class FeatureInputComponent(ExplainerComponent):
    def __init__(self, explainer, title="Feature Input", name=None,
                    subtitle="Adjust the feature values to change the prediction",
                    hide_title=False,  hide_subtitle=False, hide_index=False, 
                    hide_range=False,
                    index=None, n_input_cols=2, description=None,  **kwargs):
        """Interaction Dependence Component.

        Args:
            explainer (Explainer): explainer object constructed with either
                        ClassifierExplainer() or RegressionExplainer()
            title (str, optional): Title of tab or page. Defaults to 
                        "What if...".
            name (str, optional): unique name to add to Component elements. 
                        If None then random uuid is generated to make sure 
                        it's unique. Defaults to None.
            subtitle (str): subtitle
            hide_title (bool, optional): hide the title
            hide_subtitle (bool, optional): Hide subtitle. Defaults to False.
            hide_index (bool, optional): hide the index selector
            hide_range (bool, optional): hide the range label under the inputs
            index (str, int, optional): default index
            n_input_cols (int): number of columns to split features inputs in. 
                Defaults to 2. 
            description (str, optional): Tooltip to display when hover over
                component title. When None default text is shown. 
            
            
        """
        super().__init__(explainer, title, name)

        assert len(explainer.columns) == len(set(explainer.columns)), \
            "Not all X column names are unique, so cannot launch FeatureInputComponent component/tab!"
            
        self.index_name = 'feature-input-index-'+self.name
        
        self._input_features = self.explainer.columns_ranked_by_shap(cats=True)
        self._feature_inputs = [
            self._generate_dash_input(
                feature, self.explainer.onehot_cols, self.explainer.onehot_dict, self.explainer.categorical_dict) 
                                for feature in self._input_features]
        self._feature_callback_inputs = [Input('feature-input-'+feature+'-input-'+self.name, 'value') for feature in self._input_features]
        self._feature_callback_outputs = [Output('feature-input-'+feature+'-input-'+self.name, 'value') for feature in self._input_features] 
        if self.description is None: self.description = """
        Adjust the input values to see predictions for what if scenarios."""

    def _generate_dash_input(self, col, onehot_cols, onehot_dict, cat_dict):
        if col in cat_dict:
            col_values = cat_dict[col]
            return dbc.FormGroup([
                    dbc.Label(col),
                    dcc.Dropdown(id='feature-input-'+col+'-input-'+self.name, 
                             options=[dict(label=col_val, value=col_val) for col_val in col_values],
                             clearable=False),
                    dbc.FormText(f"Select any {col}") if not self.hide_range else None,
                ])   
        elif col in onehot_cols:
            col_values = onehot_dict[col]
            display_values = [
                col_val[len(col)+1:] if col_val.startswith(col+"_") else col_val
                    for col_val in col_values]
            return dbc.FormGroup([
                    dbc.Label(col),
                    dcc.Dropdown(id='feature-input-'+col+'-input-'+self.name, 
                             options=[dict(label=display, value=col_val) 
                                        for display, col_val in zip(display_values, col_values)],
                             clearable=False),
                    dbc.FormText(f"Select any {col}") if not self.hide_range else None,
                ])   
        else:
            min_range = np.round(self.explainer.X[col][lambda x: x != self.explainer.na_fill].min(), 2)
            max_range = np.round(self.explainer.X[col][lambda x: x != self.explainer.na_fill].max(), 2)
            return dbc.FormGroup([
                    dbc.Label(col),
                    dbc.Input(id='feature-input-'+col+'-input-'+self.name, type="number"),
                    dbc.FormText(f"Range: {min_range}-{max_range}") if not self.hide_range else None
                ])
        
    def get_slices(self, n_inputs, n_cols=2):
        """returns a list of slices to divide n inputs into n_cols columns"""
        if n_inputs < n_cols:
            n_cols = n_inputs
        rows_per_col = ceil(n_inputs / n_cols)
        slices = []
        for col in range(n_cols):
            if col == n_cols-1 and n_inputs % rows_per_col > 0:
                slices.append(slice(col*rows_per_col, col*rows_per_col+(n_inputs % rows_per_col)))
            else:
                slices.append(slice(col*rows_per_col, col*rows_per_col+rows_per_col))
        return slices

    def layout(self):
        return dbc.Card([
            make_hideable(
                dbc.CardHeader([
                        html.Div([
                            html.H3(self.title, id='feature-input-title-'+self.name),
                            make_hideable(html.H6(self.subtitle, className='card-subtitle'), hide=self.hide_subtitle),
                            dbc.Tooltip(self.description, target='feature-input-title-'+self.name),
                        ]), 
                ]), hide=self.hide_title),
            dbc.CardBody([
                dbc.Row([
                    make_hideable(
                            dbc.Col([
                                dbc.Label(f"{self.explainer.index_name}:"),
                                dcc.Dropdown(id='feature-input-index-'+self.name, 
                                    options = [{'label': str(idx), 'value':idx} 
                                                    for idx in self.explainer.idxs],
                                    value=self.index)
                            ], md=4), hide=self.hide_index), 
                    ], form=True),
                dbc.Row([dbc.Col(self._feature_inputs[slicer]) 
                            for slicer in self.get_slices(len(self._feature_inputs), self.n_input_cols)]),
            ])
        ])

    def component_callbacks(self, app):
        
        @app.callback(
            [*self._feature_callback_outputs],
            [Input('feature-input-index-'+self.name, 'value')]
        )
        def update_whatif_inputs(index):
            idx = self.explainer.get_int_idx(index)
            if idx is None:
                raise PreventUpdate
            feature_values = (self.explainer.X_cats
                                [self.explainer.columns_ranked_by_shap(cats=True)]
                                .iloc[[idx]].values[0].tolist())
            return feature_values



                        
