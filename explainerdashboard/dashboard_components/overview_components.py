__all__ = [
    'PredictionSummaryComponent',
    'ImportancesComponent',
    'PdpComponent',
    'WhatIfComponent',
]

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
                    hide_title=False, hide_selector=False,
                    pos_label=None, index=None, percentile=True):
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
        return html.Div([
            make_hideable(
                html.H3("Predictions summary:"), hide=self.hide_title),
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

    def _register_callbacks(self, app):
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
    def __init__(self, explainer, title="Importances", name=None,
                        hide_type=False, hide_depth=False, hide_cats=False,
                        hide_title=False, hide_selector=False,
                        pos_label=None, importance_type="shap", depth=None, cats=True):
        """Display features importances component

        Args:
            explainer (Explainer): explainer object constructed with either
                        ClassifierExplainer() or RegressionExplainer()
            title (str, optional): Title of tab or page. Defaults to 
                        "Importances".
            name (str, optional): unique name to add to Component elements. 
                        If None then random uuid is generated to make sure 
                        it's unique. Defaults to None.
            hide_type (bool, optional): Hide permutation/shap selector toggle. 
                        Defaults to False.
            hide_depth (bool, optional): Hide number of features toggle. 
                        Defaults to False.
            hide_cats (bool, optional): Hide group cats toggle. 
                        Defaults to False.
            hide_title (bool, optional): hide title. Defaults to False.
            hide_selector (bool, optional): hide pos label selectors. 
                        Defaults to False.
            pos_label ({int, str}, optional): initial pos label. 
                        Defaults to explainer.pos_label
            importance_type (str, {'permutation', 'shap'} optional): 
                        initial importance type to display. Defaults to "shap".
            depth (int, optional): Initial number of top features to display. 
                        Defaults to None (=show all).
            cats (bool, optional): Group categoricals. Defaults to True.
        """
        super().__init__(explainer, title, name)

        if self.explainer.cats is None or not self.explainer.cats:
            self.hide_cats = True

        assert importance_type in ['shap', 'permutation'], \
            "importance type must be either 'shap' or 'permutation'!"

        if depth is not None:
            self.depth = min(depth, len(explainer.columns_ranked_by_shap(cats)))

        

        self.selector = PosLabelSelector(explainer, name=self.name, pos_label=pos_label)

        if self.explainer.y_missing:
            self.hide_type = True
            self.importance_type = 'shap'
        self.register_dependencies('shap_values', 'shap_values_cats')
        if not (self.hide_type and self.importance_type == 'shap'):
            self.register_dependencies('permutation_importances', 'permutation_importances_cats')

    def layout(self):
        return html.Div([
            dbc.Row([
                make_hideable(
                    dbc.Col([html.H2('Feature Importances:')]), hide=self.hide_title),
            ]),
            dbc.Row([
                make_hideable(
                    dbc.Col([
                        dbc.FormGroup([
                            dbc.Label("Importances type:"),
                            dbc.RadioItems(
                                options=[
                                    {'label': 'Permutation Importances', 
                                    'value': 'permutation'},
                                    {'label': 'SHAP values', 
                                    'value': 'shap'}
                                ],
                                value=self.importance_type,
                                id='importances-permutation-or-shap-'+self.name,
                                inline=True,
                            ),
                        ]),
                    ]), self.hide_type),
                make_hideable(
                    dbc.Col([
                        html.Label('Depth:'),
                        dcc.Dropdown(id='importances-depth-'+self.name,
                                            options = [{'label': str(i+1), 'value':i+1} 
                                                        for i in range(self.explainer.n_features(self.cats))],
                                            value=self.depth)
                    ], md=3), self.hide_depth),
                make_hideable(
                    dbc.Col([
                        dbc.Label("Grouping:"),
                        dbc.FormGroup(
                        [
                            dbc.RadioButton(
                                id='importances-group-cats-'+self.name, 
                                className="form-check-input",
                                checked=self.cats),
                            dbc.Label("Group Cats",
                                    html_for='importances-group-cats-'+self.name,
                                    className="form-check-label"),
                        ], check=True), 
                    ]),  self.hide_cats),    
                make_hideable(
                        dbc.Col([self.selector.layout()
                    ], width=2), hide=self.hide_selector)    
            ], form=True),

            dbc.Row([
                dbc.Col([
                    dcc.Graph(id='importances-graph-'+self.name,
                                config=dict(modeBarButtons=[['toImage']], displaylogo=False))
                ]),
            ]), 
        ])
        
    def _register_callbacks(self, app, **kwargs):
        @app.callback(  
            Output('importances-graph-'+self.name, 'figure'),
            [Input('importances-depth-'+self.name, 'value'),
             Input('importances-group-cats-'+self.name, 'checked'),
             Input('importances-permutation-or-shap-'+self.name, 'value'),
             Input('pos-label-'+self.name, 'value')],
        )
        def update_importances(depth, cats, permutation_shap, pos_label):
            return self.explainer.plot_importances(
                        kind=permutation_shap, topx=depth, 
                        cats=cats, pos_label=pos_label)


class PdpComponent(ExplainerComponent):
    def __init__(self, explainer, title="Partial Dependence Plot", name=None,
                    hide_col=False, hide_index=False, hide_cats=False,
                    hide_title=False, hide_selector=False,
                    hide_dropna=False, hide_sample=False, 
                    hide_gridlines=False, hide_gridpoints=False,
                    pos_label=None, col=None, index=None, cats=True,
                    dropna=True, sample=100, gridlines=50, gridpoints=10):
        """Show Partial Dependence Plot component

        Args:
            explainer (Explainer): explainer object constructed with either
                        ClassifierExplainer() or RegressionExplainer()
            title (str, optional): Title of tab or page. Defaults to 
                        "Partial Dependence Plot".
            name (str, optional): unique name to add to Component elements. 
                        If None then random uuid is generated to make sure 
                        it's unique. Defaults to None.
            hide_col (bool, optional): Hide feature selector. Defaults to False.
            hide_index (bool, optional): Hide index selector. Defaults to False.
            hide_cats (bool, optional): Hide group cats toggle. Defaults to False.
            hide_title (bool, optional): Hide title, Defaults to False.
            hide_selector (bool, optional): hide pos label selectors. Defaults to False.
            hide_dropna (bool, optional): Hide drop na's toggle Defaults to False.
            hide_sample (bool, optional): Hide sample size input. Defaults to False.
            hide_gridlines (bool, optional): Hide gridlines input. Defaults to False.
            hide_gridpoints (bool, optional): Hide gridpounts input. Defaults to False.
            pos_label ({int, str}, optional): initial pos label. 
                        Defaults to explainer.pos_label
            col (str, optional): Feature to display PDP for. Defaults to None.
            index ({int, str}, optional): Index to add ice line to plot. Defaults to None.
            cats (bool, optional): Group categoricals for feature selector. Defaults to True.
            dropna (bool, optional): Drop rows where values equal explainer.na_fill (usually -999). Defaults to True.
            sample (int, optional): Sample size to calculate average partial dependence. Defaults to 100.
            gridlines (int, optional): Number of ice lines to display in plot. Defaults to 50.
            gridpoints (int, optional): Number of breakpoints on horizontal axis Defaults to 10.
        """
        super().__init__(explainer, title, name)

        self.index_name = 'pdp-index-'+self.name

        if self.col is None:
            self.col = self.explainer.columns_ranked_by_shap(self.cats)[0]

        self.selector = PosLabelSelector(explainer, name=self.name, pos_label=pos_label)

    def layout(self):
        return html.Div([
                make_hideable(
                    html.H3('Partial Dependence Plot:'), hide=self.hide_title),
                dbc.Row([
                    make_hideable(
                        dbc.Col([
                            dbc.Label("Feature:", html_for='pdp-col'),
                            dcc.Dropdown(id='pdp-col-'+self.name, 
                                options=[{'label': col, 'value':col} 
                                            for col in self.explainer.columns_ranked_by_shap(self.cats)],
                                value=self.col),
                        ], md=4), hide=self.hide_col),
                    make_hideable(
                        dbc.Col([
                            dbc.Label(f"{self.explainer.index_name}:"),
                            dcc.Dropdown(id='pdp-index-'+self.name, 
                                options = [{'label': str(idx), 'value':idx} 
                                                for idx in self.explainer.idxs],
                                value=None)
                        ], md=4), hide=self.hide_index), 
                    make_hideable(
                        dbc.Col([self.selector.layout()
                    ], width=2), hide=self.hide_selector),
                    make_hideable(
                        dbc.Col([
                            dbc.Label("Grouping:"),
                            dbc.FormGroup(
                            [
                                dbc.RadioButton(
                                    id='pdp-group-cats-'+self.name, 
                                    className="form-check-input",
                                    checked=self.cats),
                                dbc.Label("Group Cats",
                                        html_for='pdp-group-cats-'+self.name, 
                                        className="form-check-label"),
                            ], check=True)
                        ], md=2), hide=self.hide_cats),
                    
                ], form=True),
                dbc.Row([
                    dbc.Col([
                        dcc.Loading(id='loading-pdp-graph-'+self.name, 
                            children=[dcc.Graph(id='pdp-graph-'+self.name,
                                                config=dict(modeBarButtons=[['toImage']], displaylogo=False))]),
                    ])
                ]),
                dbc.Row([
                    make_hideable(
                        dbc.Col([ #
                            dbc.Label("Drop na:"),
                                dbc.FormGroup(
                                [
                                    dbc.RadioButton(
                                        id='pdp-dropna-'+self.name, 
                                        className="form-check-input",
                                        checked=self.dropna),
                                    dbc.Label("Drop na's",
                                            html_for='pdp-dropna-'+self.name, 
                                            className="form-check-label"),
                                ], check=True)
                        ]), hide=self.hide_dropna),
                    make_hideable(
                        dbc.Col([ 
                            dbc.Label("pdp sample size"),
                            dbc.Input(id='pdp-sample-'+self.name, value=self.sample,
                                type="number", min=0, max=len(self.explainer), step=1),
                        ]), hide=self.hide_sample),  
                    make_hideable(   
                        dbc.Col([ #gridlines
                            dbc.Label("gridlines"),
                            dbc.Input(id='pdp-gridlines-'+self.name, value=self.gridlines,
                                    type="number", min=0, max=len(self.explainer), step=1),
                        ]), hide=self.hide_gridlines),
                    make_hideable(
                        dbc.Col([ #gridpoints
                            dbc.Label("gridpoints"),
                            dbc.Input(id='pdp-gridpoints-'+self.name, value=self.gridpoints,
                                type="number", min=0, max=100, step=1),
                        ]), hide=self.hide_gridpoints),
                ], form=True)
        ])
                
    def _register_callbacks(self, app):
        @app.callback(
            Output('pdp-graph-'+self.name, 'figure'),
            [Input('pdp-index-'+self.name, 'value'),
             Input('pdp-col-'+self.name, 'value'),
             Input('pdp-dropna-'+self.name, 'checked'),
             Input('pdp-sample-'+self.name, 'value'),
             Input('pdp-gridlines-'+self.name, 'value'),
             Input('pdp-gridpoints-'+self.name, 'value'),
             Input('pos-label-'+self.name, 'value')]
        )
        def update_pdp_graph(index, col, drop_na, sample, gridlines, gridpoints, pos_label):
            return self.explainer.plot_pdp(col, index, 
                drop_na=drop_na, sample=sample, gridlines=gridlines, gridpoints=gridpoints, 
                pos_label=pos_label)

        @app.callback(
            Output('pdp-col-'+self.name, 'options'),
            [Input('pdp-group-cats-'+self.name, 'checked')],
            [State('pos-label-'+self.name, 'value')]
        )
        def update_pdp_graph(cats, pos_label):
            col_options = [{'label': col, 'value':col} 
                                for col in self.explainer.columns_ranked_by_shap(cats, pos_label=pos_label)]
            return col_options


class WhatIfComponent(ExplainerComponent):
    def __init__(self, explainer, title="What if...", name=None,
                    hide_title=False, hide_index=False, hide_selector=False,
                    hide_contributions=False, hide_pdp=False,
                    index=None, pdp_col=None, pos_label=None):
        """Interaction Dependence Component.

        Args:
            explainer (Explainer): explainer object constructed with either
                        ClassifierExplainer() or RegressionExplainer()
            title (str, optional): Title of tab or page. Defaults to 
                        "What if...".
            name (str, optional): unique name to add to Component elements. 
                        If None then random uuid is generated to make sure 
                        it's unique. Defaults to None.
            hide_title (bool, optional): hide the title
            hide_index (bool, optional): hide the index selector
            hide_selector (bool, optional): hide the pos_label selector
            hide_contributions (bool, optional): hide the contributions graph
            hide_pdp (bool, optional): hide the pdp graph
            index (str, int, optional): default index
            pdp_col (str, optional): default pdp feature col
            pos_label ({int, str}, optional): initial pos label. 
                        Defaults to explainer.pos_label
            
        """
        super().__init__(explainer, title, name)

        assert len(explainer.columns) == len(set(explainer.columns)), \
            "Not all column names are unique, so cannot launch whatif component/tab!"
        
        if self.pdp_col is None:
            self.pdp_col = self.explainer.columns_ranked_by_shap(cats=True)[0]
            
        self.index_name = 'whatif-index-'+self.name
        
        self._input_features = self.explainer.columns_cats
        self._feature_inputs = [
            self._generate_dash_input(
                feature, self.explainer.cats, self.explainer.cats_dict) 
                                for feature in self._input_features]
        self._feature_callback_inputs = [Input('whatif-'+feature+'-input-'+self.name, 'value') for feature in self._input_features]
        self._feature_callback_outputs = [Output('whatif-'+feature+'-input-'+self.name, 'value') for feature in self._input_features]
        
        self.selector = PosLabelSelector(explainer, name=self.name, pos_label=pos_label)
        
        self.register_dependencies('preds', 'shap_values')
        
        
    def _generate_dash_input(self, col, cats, cats_dict):
        if col in cats:
            col_values = [
                col_val[len(col)+1:] if col_val.startswith(col+"_") else col_val
                    for col_val in cats_dict[col]]
            return html.Div([
                html.P(col),
                dcc.Dropdown(id='whatif-'+col+'-input-'+self.name, 
                             options=[dict(label=col_val, value=col_val) for col_val in col_values],
                             clearable=False)
            ])
        else:
            return  html.Div([
                        html.P(col),
                        dbc.Input(id='whatif-'+col+'-input-'+self.name, type="number"),
                    ])
        
    def layout(self):
        return dbc.Container([
            dbc.Row([
                make_hideable(
                    dbc.Col([
                        html.H1("What if...analysis")
                    ]), hide=self.hide_title)
            ]),
            dbc.Row([
                make_hideable(
                        dbc.Col([
                            dbc.Label(f"{self.explainer.index_name}:"),
                            dcc.Dropdown(id='whatif-index-'+self.name, 
                                options = [{'label': str(idx), 'value':idx} 
                                                for idx in self.explainer.idxs],
                                value=self.index)
                        ], md=4), hide=self.hide_index), 
                make_hideable(
                        dbc.Col([self.selector.layout()
                    ], md=2), hide=self.hide_selector),
                ], form=True),
            dbc.Row([
                dbc.Col([
                    html.H3("Edit Feature input:")
                ])
            ]),
            dbc.Row([
                dbc.Col([
                    *self._feature_inputs[:int((len(self._feature_inputs) + 1)/2)]
                ]),
                dbc.Col([
                    *self._feature_inputs[int((len(self._feature_inputs) + 1)/2):]
                ]),
            ]),
            dbc.Row([
                make_hideable(
                    dbc.Col([
                        html.H3("Prediction and contributions:"),
                        dcc.Graph(id='whatif-contrib-graph-'+self.name,
                                                config=dict(modeBarButtons=[['toImage']], displaylogo=False)),
                    ]), hide=self.hide_contributions),
                make_hideable(
                    dbc.Col([
                        html.H3("Partial dependence:"),
                        dcc.Dropdown(id='whatif-pdp-col-'+self.name, 
                                     options=[dict(label=col, value=col) for col in self._input_features],
                                     value=self.pdp_col),
                        dcc.Graph(id='whatif-pdp-graph-'+self.name, 
                                    config=dict(modeBarButtons=[['toImage']], displaylogo=False)),
                    ]), hide=self.hide_pdp),
            ])
        ], fluid=True)
        
    def _register_callbacks(self, app):
        @app.callback(
            [Output('whatif-contrib-graph-'+self.name, 'figure'),
             Output('whatif-pdp-graph-'+self.name, 'figure')],
            [Input('whatif-pdp-col-'+self.name, 'value'),
             Input('pos-label-'+self.name, 'value'),
             *self._feature_callback_inputs,
             ],
        )
        def update_whatif_plots(pdp_col, pos_label, *input_args):
            X_row = pd.DataFrame(dict(zip(self._input_features, input_args)), index=[0]).fillna(0)
            contrib_plot = self.explainer.plot_shap_contributions(X_row=X_row, pos_label=pos_label)
            pdp_plot = self.explainer.plot_pdp(pdp_col, X_row=X_row, pos_label=pos_label)
            return contrib_plot, pdp_plot
        
        @app.callback(
            [*self._feature_callback_outputs],
            [Input('whatif-index-'+self.name, 'value')]
        )
        def update_whatif_inputs(index):
            idx = self.explainer.get_int_idx(index)
            if idx is None:
                raise PreventUpdate
            feature_values = self.explainer.X_cats.iloc[[idx]].values[0].tolist()
            return feature_values
                        
