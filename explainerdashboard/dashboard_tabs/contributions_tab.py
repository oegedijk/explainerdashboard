__all__ = ['ContributionsTab',
           'IndividualContributionsComponent',
            #'IndexComponent',
            #'RandomIndexComponent',
            'ClassifierRandomIndexComponent',
            #'RegressionRandomIndexComponent',
            'PredictionSummaryComponent',
            'ContributionsGraphComponent',
            'ContributionsTableComponent',
            'PdpComponent'
            ]

import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_table

from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

import numpy as np
import pandas as pd

from .dashboard_methods import *


class IndividualContributionsComponent(ExplainerComponent):
    def __init__(self, explainer, title="Individual Contributions",
                        header_mode="none", name=None):
        super().__init__(explainer, title, header_mode, name)

        self.index = ClassifierRandomIndexComponent(explainer)
        self.summary = PredictionSummaryComponent(explainer)
        self.contributions = ContributionsGraphComponent(explainer)
        self.pdp = PdpComponent(explainer)
        self.contributions_list = ContributionsTableComponent(explainer)

        self.index_connector = IndexConnector(self.index, 
                [self.summary, self.contributions, self.pdp, self.contributions_list])

        self.register_components(self.index, self.summary, self.contributions, self.pdp, self.contributions_list, self.index_connector)

    def _layout(self):
        return html.Div([
            dbc.Container([
                dbc.Row([
                    dbc.Col([
                        self.index.layout()
                    ]),
                    dbc.Col([
                        self.summary.layout()
                    ]),
                ]),
                dbc.Row([
                    dbc.Col([
                        self.contributions.layout()
                    ]),
                    dbc.Col([
                        self.pdp.layout()
                    ]),
                ]),
                dbc.Row([
                    dbc.Col([
                        self.contributions_list.layout()
                    ]),
                    dbc.Col([
                        html.Div([]),

                    ]),
                ])
            ], fluid=True),
        ])


class IndexConnector(ExplainerComponent):
    def __init__(self, input_index, output_indexes):
        self.input_index_name = self.index_name(input_index)
        self.output_index_names = self.index_name(output_indexes, multi=True)

    @staticmethod
    def index_name(indexes, multi=False):
            index_name_list = []
            if isinstance(indexes, str):
                index_name_list.append(indexes)
            elif isinstance(indexes, ExplainerComponent) and hasattr(indexes, "index_name"):
                index_name_list.append(indexes.index_name)
            elif multi and hasattr(indexes, '__iter__'):
                for index in indexes:
                    if isinstance(index, str):
                        index_name_list.append(index)
                    elif isinstance(index, ExplainerComponent) and hasattr(index, "index_name"):
                        index_name_list.append(index.index_name)
                    else:
                        raise ValueError("inputs/outputs should be either str or an ExplainerComponent with a .index_name property!"
                                        f"{index} is neither!")

            if multi:
                return index_name_list
            else:
                return index_name_list[0]

    def register_callbacks(self, app):
        @app.callback(
            [Output(index_name, 'value') for index_name in self.output_index_names],
            [Input(self.input_index_name, 'value')]
        )
        def update_indexes(index):
            return tuple(index for i in range(len(self.output_index_names)))


class ClassifierRandomIndexComponent(ExplainerComponent):
    def __init__(self, explainer, title="Select Random Index",
                        header_mode="none", name=None,
                        hide_index=False, hide_slider=False, 
                        hide_labels=False, hide_pred_or_perc=False,
                        hide_button=False,
                        index=None, slider= None, labels=None, 
                        pred_or_perc='predictions'):
        super().__init__(explainer,title, header_mode, name)

        self.hide_index, self.hide_slider = hide_index, hide_slider
        self.hide_labels, self.hide_pred_or_perc = hide_labels, hide_pred_or_perc
        self.hide_button = hide_button

        self.index, self.slider = index, slider
        self.labels, self.pred_or_perc = labels, pred_or_perc

        self.index_name = 'random-index-clas-index-'+self.name

        if self.slider is None:
            self.slider = [0.0, 1.0]

        if self.labels is None:
            self.labels = self.explainer.labels

        assert (len(self.slider)==2 and 
                self.slider[0]>=0 and self.slider[0]<=1 and 
                self.slider[1]>=0.0 and self.slider[1]<=1.0 and 
                self.slider[0]<=self.slider[1]), \
                    "slider should be e.g. [0.5, 1.0]"

        assert all([lab in self.explainer.labels for lab in self.labels]), \
            f"These labels are not in explainer.labels: {[lab for lab in labs if lab not in explainer.labels]}!"

        assert self.pred_or_perc in ['predictions', 'percentiles'], \
            "pred_or_perc should either be `predictions` or `percentiles`!"

    def _layout(self):
        return html.Div([
            html.H3("Select index:"),
            dbc.Row([
                make_hideable(
                    dbc.Col([
                            #dbc.Label("Index:", html_for='random-index-clas-index-'+self.name),
                            dcc.Dropdown(id='random-index-clas-index-'+self.name, 
                                    options = [{'label': str(idx), 'value':idx} 
                                                    for idx in self.explainer.idxs],
                                    value=self.index)
                        ]), hide=self.hide_index),
                make_hideable(
                    dbc.Col([
                        dbc.Button("Random Index", color="primary", id='random-index-clas-button-'+self.name, block=True)
                    ], md=2), hide=self.hide_button),
            ], form=True),
            dbc.Row([
                make_hideable(
                    dbc.Col([
                        html.Div([
                            dbc.Label(id='random-index-clas-slider-label-'+self.name,
                                children="Predictions range:", 
                                html_for='prediction-range-slider-'+self.name,),
                            dcc.RangeSlider(
                                id='random-index-clas-slider-'+self.name,
                                min=0.0, max=1.0, step=0.01,
                                value=self.slider,  allowCross=False,
                                marks={0.0:'0.0', 0.1:'0.1', 0.2:'0.2', 0.3:'0.3', 
                                        0.4:'0.4', 0.5:'0.5', 0.6:'0.6', 0.7:'0.7', 
                                        0.8:'0.8', 0.9:'0.9', 1.0:'1.0'},
                                tooltip = {'always_visible' : False})
                        ], style={'margin-bottom':25})
                    ], md=5), hide=self.hide_slider),
                make_hideable(
                    dbc.Col([
                        dbc.Label("Include labels (y):"),
                        dcc.Dropdown(
                            id='random-index-clas-labels-'+self.name,
                            options=[{'label': lab, 'value': lab} for lab in self.explainer.labels],
                            multi=True,
                            value=self.labels),
                    ], md=4), hide=self.hide_labels),
                make_hideable(
                    dbc.Col([
                        dbc.RadioItems(
                            id='random-index-clas-pred-or-perc-'+self.name,
                            options=[
                                {'label': 'Use predictions', 'value': 'predictions'},
                                {'label': 'Use percentiles', 'value': 'percentiles'},
                            ],
                            value=self.pred_or_perc,
                            inline=True)
                    ], md=3), hide=self.hide_pred_or_perc),
            ], justify="start"),

        ])
    
    def _register_callbacks(self, app):
        @app.callback(
            Output('random-index-clas-index-'+self.name, 'value'),
            [Input('random-index-clas-button-'+self.name, 'n_clicks')],
            [State('random-index-clas-slider-'+self.name, 'value'),
             State('random-index-clas-labels-'+self.name, 'value'),
             State('random-index-clas-pred-or-perc-'+self.name, 'value'),
             State('pos-label', 'value')])
        def update_index(n_clicks, slider_range, labels, pred_or_perc, pos_label):

            if pred_or_perc == 'predictions':
                return self.explainer.random_index(y_values=labels, 
                    pred_proba_min=slider_range[0], pred_proba_max=slider_range[1], 
                    return_str=True, pos_label=pos_label)
            elif pred_or_perc == 'percentiles':
                return self.explainer.random_index(y_values=labels, 
                    pred_percentile_min=slider_range[0], pred_percentile_max=slider_range[1], 
                    return_str=True, pos_label=pos_label)

        @app.callback(
            Output('random-index-clas-slider-label-'+self.name, 'children'),
            [Input('random-index-clas-pred-or-perc-'+self.name, 'value')]
        )
        def update_slider_label(pred_or_perc):
            if pred_or_perc == 'predictions':
                return "Predictions range:"
            elif pred_or_perc == 'percentiles':
                return "Percentiles range:"
            raise PreventUpdate



        

class PredictionSummaryComponent(ExplainerComponent):
    def __init__(self, explainer, title="Prediction Summary",
                    header_mode="none", name=None,
                    hide_index=False, hide_percentile=False,
                    index=None, percentile=True):
        super().__init__(explainer, title, header_mode, name)

        self.hide_index, self.hide_percentile = hide_index, hide_percentile
        self.index, self.percentile = index, percentile

        self.index_name = 'modelprediction-index-'+self.name

    def _layout(self):
        return html.Div([
            html.H3("Predictions summary:"),
            dbc.Row([
                make_hideable(
                    dbc.Col([
                        dbc.Label("Index:"),
                        dcc.Dropdown(id='modelprediction-index-'+self.name, 
                                options = [{'label': str(idx), 'value':idx} 
                                                for idx in self.explainer.idxs],
                                value=self.index)
                    ]), hide=self.hide_index),
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

            dcc.Loading(id='loading-modelprediction-'+self.name, 
                         children=[dcc.Markdown(id='modelprediction-'+self.name)]),    
        ])

    def _register_callbacks(self, app):
        @app.callback(
            Output('modelprediction-'+self.name, 'children'),
            [Input('modelprediction-index-'+self.name, 'value'),
             Input('modelprediction-percentile-'+self.name, 'checked'),
             Input('pos-label', 'value')])
        def update_output_div(index, include_percentile, pos_label):
            if index is not None:
                return self.explainer.prediction_result_markdown(index, include_percentile=include_percentile, pos_label=pos_label)
            raise PreventUpdate

class ContributionsGraphComponent(ExplainerComponent):
    def __init__(self, explainer, title="Contributions",
                    header_mode="none", name=None,
                    hide_index=False, hide_depth=False, hide_cats=False,
                    index=None, depth=None, cats=True):
        super().__init__(explainer, title, header_mode, name)

        self.hide_index, self.hide_depth, self.hide_cats = \
            hide_index, hide_depth, hide_cats
        self.index, self.depth, self.cats = index, depth, cats

        self.index_name = 'contributions-graph-index-'+self.name

        if self.depth is not None:
            self.depth = min(self.depth, len(self.explainer.columns_ranked_by_shap(self.cats)))

        self.register_dependencies('shap_values', 'shap_values_cats')

    def _layout(self):
        return html.Div([
            html.H3("Contributions to prediction:"),
            dbc.Row([
                make_hideable(
                    dbc.Col([
                        dbc.Label("Index:"),
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
                                            for i in range(len(self.explainer.columns_ranked_by_shap(self.cats)))],
                            value=self.depth)
                    ], md=2), hide=self.hide_depth),
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
                    ], md=3), hide=self.hide_cats),
                ], form=True),
            dbc.Row([
                dbc.Col([
                    dcc.Loading(id='loading-contributions-graph-'+self.name, 
                        children=[dcc.Graph(id='contributions-graph-'+self.name)]),
                ]),
            ]),
        ])
        
    def _register_callbacks(self, app):
        @app.callback(
            [Output('contributions-graph-'+self.name, 'figure'),
             Output('contributions-graph-depth-'+self.name, 'options')],
            [Input('contributions-graph-index-'+self.name, 'value'),
             Input('contributions-graph-depth-'+self.name, 'value'),
             Input('contributions-graph-group-cats-'+self.name, 'checked'),
             Input('pos-label', 'value')])
        def update_output_div(index, depth, cats, pos_label):
            if index is None:
                raise PreventUpdate

            plot = self.explainer.plot_shap_contributions(index, topx=depth, cats=cats, pos_label=pos_label)
            ctx = dash.callback_context
            trigger = ctx.triggered[0]['prop_id'].split('.')[0]
            if trigger == 'contributions-graph-group-cats-'+self.name:
                depth_options = [{'label': str(i+1), 'value': i+1} 
                                        for i in range(len(self.explainer.columns_ranked_by_shap(cats)))]
                return (plot, depth_options)
            else:
                return (plot, dash.no_update)


class ContributionsTableComponent(ExplainerComponent):
    def __init__(self, explainer, title="Contributions",
                    header_mode="none", name=None,
                    hide_index=False, hide_depth=False, hide_cats=False,
                    index=None, depth=None, cats=True):
        super().__init__(explainer, title, header_mode, name)

        self.hide_index, self.hide_depth, self.hide_cats = \
            hide_index, hide_depth, hide_cats
        self.index, self.depth, self.cats = index, depth, cats

        self.index_name = 'contributions-table-index-'+self.name

        if self.depth is not None:
            self.depth = min(self.depth, len(self.explainer.columns_ranked_by_shap(self.cats)))

        self.register_dependencies('shap_values', 'shap_values_cats')

    def _layout(self):
        return html.Div([
            html.H3("Contributions to prediction:"),
            dbc.Row([
                make_hideable(
                    dbc.Col([
                        dbc.Label("Display contributions for:"),
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
                                            for i in range(len(self.explainer.columns_ranked_by_shap(self.cats)))],
                            value=self.depth)
                    ], md=2), hide=self.hide_depth),
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
                    dash_table.DataTable(
                        id='contributions-table-'+self.name, 
                        #style_cell={'fontSize':18, 'font-family':'sans-serif'},
                        columns=[{'id': c, 'name': c} for c in ['Reason', 'Effect']],    
                    ),    
                ]),
            ]),
        ])
        
    def _register_callbacks(self, app):
        @app.callback(
            [Output('contributions-table-'+self.name, 'data'),
             Output('contributions-table-'+self.name, 'tooltip_data'),
             Output('contributions-table-depth-'+self.name, 'options')],
            [Input('contributions-table-index-'+self.name, 'value'),
             Input('contributions-table-depth-'+self.name, 'value'),
             Input('contributions-table-group-cats-'+self.name, 'checked'),
             Input('pos-label', 'value')])
        def update_output_div(index, depth, cats, pos_label):
            if index is None:
                raise PreventUpdate

            contributions_table = self.explainer.contrib_summary_df(index, cats=cats, topx=depth, pos_label=pos_label).to_dict('records')
            tooltip_data = [{'Reason': desc} for desc in self.explainer.description_list(
                                self.explainer.contrib_df(index, cats=cats, topx=depth)['col'])]
            
            ctx = dash.callback_context
            trigger = ctx.triggered[0]['prop_id'].split('.')[0]
            if trigger == 'contributions-table-group-cats-'+self.name:
                depth_options = [{'label': str(i+1), 'value': i+1} 
                                        for i in range(len(self.explainer.columns_ranked_by_shap(cats)))]
                return (contributions_table, tooltip_data, depth_options)
            else:
                return (contributions_table, tooltip_data, dash.no_update)
  

class PdpComponent(ExplainerComponent):
    def __init__(self, explainer, title="Partial Dependence Plot",
                    header_mode="none", name=None,
                    hide_col=False, hide_index=False, hide_cats=False,
                    hide_dropna=False, hide_sample=False, 
                    hide_gridlines=False, hide_gridpoints=False,
                    col=None, index=None, cats=True,
                    dropna=True, sample=100, gridlines=50, gridpoints=10):
        super().__init__(explainer, title, header_mode, name)

        self.hide_col, self.hide_index, self.hide_cats = hide_col, hide_index, hide_cats
        self.hide_dropna, self.hide_sample = hide_dropna, hide_sample
        self.hide_gridlines, self.hide_gridpoints = hide_gridlines, hide_gridpoints

        self.col, self.index, self.cats = col, index, cats
        self.dropna, self.sample, self.gridlines, self.gridpoints = \
            dropna, sample, gridlines, gridpoints

        self.index_name = 'pdp-index-'+self.name

        if self.col is None:
            self.col = self.explainer.columns_ranked_by_shap(self.cats)[0]

    def _layout(self):
        return html.Div([
                html.H3('Partial Dependence Plot:'),
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
                            dbc.Label("Index:"),
                            dcc.Dropdown(id='pdp-index-'+self.name, 
                                options = [{'label': str(idx), 'value':idx} 
                                                for idx in self.explainer.idxs],
                                value=None)
                        ], md=4), hide=self.hide_index), 
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
                        ], md=3), hide=self.hide_cats),
                ], form=True),
                dbc.Row([
                    dbc.Col([
                        dcc.Loading(id='loading-pdp-graph-'+self.name, 
                            children=[dcc.Graph(id='pdp-graph-'+self.name)]),
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
             Input('pos-label', 'value')]
        )
        def update_pdp_graph(index, col, drop_na, sample, gridlines, gridpoints, pos_label):
            return self.explainer.plot_pdp(col, index, 
                drop_na=drop_na, sample=sample, gridlines=gridlines, gridpoints=gridpoints, 
                pos_label=pos_label)

        @app.callback(
            Output('pdp-col-'+self.name, 'options'),
            [Input('pdp-group-cats-'+self.name, 'checked')]
        )
        def update_pdp_graph(cats):
            col_options = [{'label': col, 'value':col} 
                                for col in self.explainer.columns_ranked_by_shap(cats)]
            return col_options

def contributions_callbacks(explainer, app, round=2, **kwargs):

    if explainer.is_classifier:
        @app.callback(
            Output('input-index', 'value'),
            [Input('index-button', 'n_clicks')],
            [State('prediction-range-slider', 'value'),
             State('include-labels', 'value'),
             State('preds-or-ranks', 'value'),
             State('tabs', 'value')]
        )
        def update_input_index(n_clicks, slider_range, include, preds_or_ranks, tab):
            y = None
            if include=='neg': y = 0 
            elif include=='pos': y = 1
            return_str = True if explainer.idxs is not None else False

            if preds_or_ranks == 'preds':
                idx = explainer.random_index(
                    y_values=y, pred_proba_min=slider_range[0], pred_proba_max=slider_range[1],
                    return_str=return_str)
            elif preds_or_ranks == 'ranks':
                idx = explainer.random_index(
                    y_values=y, pred_percentile_min=slider_range[0], pred_percentile_max=slider_range[1],
                    return_str=return_str)

            if idx is not None:
                return idx
            raise PreventUpdate
    else:
        @app.callback(
            Output('input-index', 'value'),
            [Input('index-button', 'n_clicks')],
            [State('prediction-range-slider', 'value')]
        )
        def update_input_index(n_clicks, slider_range,):
            y = None
            return_str = True if explainer.idxs is not None else False
            idx = explainer.random_index(pred_min=slider_range[0], pred_max=slider_range[1], return_str=return_str)
            if idx is not None:
                return idx
            raise PreventUpdate

    @app.callback(
        Output('index-store', 'data'),
        [Input('input-index', 'value')]
    )
    def update_bsn_div(input_index):
        if (explainer.idxs is None 
            and str(input_index).isdigit() 
            and int(input_index) <= len(explainer)):
            return int(input_index)
        elif (explainer.idxs is not None
             and str(input_index) in explainer.idxs):
             return str(input_index)
        raise PreventUpdate

    @app.callback(
        [Output('model-prediction', 'children'),
         Output('contributions-graph', 'figure'),
         Output('contributions_table', 'data'),
         Output('contributions_table', 'tooltip_data')],
        [Input('index-store', 'data'),
         Input('contributions-size', 'value'),
         Input('pos-label', 'value')]
    )
    def update_output_div(index, topx, pos_label):
        if index is None:
            raise PreventUpdate
        prediction_result_md = explainer.prediction_result_markdown(index, pos_label=pos_label)
        plot = explainer.plot_shap_contributions(index, topx=topx, round=round, pos_label=pos_label)
        summary_table = explainer.contrib_summary_df(index, round=round, pos_label=pos_label)
        tooltip_data = [{'Reason': desc} for desc in explainer.description_list(explainer.contrib_df(index)['col'])]
        return (prediction_result_md, plot, summary_table.to_dict('records'), tooltip_data)

    @app.callback(
        Output('pdp-col', 'value'),
        [Input('contributions-graph', 'clickData')])
    def update_pdp_col(clickData):
        if clickData is not None:
            col = clickData['points'][0]['x']
            return col
        raise PreventUpdate

    @app.callback(
        Output('pdp-graph', 'figure'),
        [Input('index-store', 'data'),
         Input('pdp-col', 'value'),
         Input('pos-label', 'value')]
    )
    def update_pdp_graph(idx, col, pos_label):
        return explainer.plot_pdp(col, idx, sample=100, pos_label=pos_label)