__all__ = [
    'ClassifierRandomIndexComponent',
    'CutoffConnector',
    'IndexConnector',
    'HighlightConnector'
]

import numpy as np

import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html

from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from .dashboard_methods import  *

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
                        ], md=8), hide=self.hide_index),
                make_hideable(
                    dbc.Col([
                        dbc.Button("Random Index", color="primary", id='random-index-clas-button-'+self.name, block=True)
                    ], md=4), hide=self.hide_button),
            ], form=True),
            dbc.Row([
                make_hideable(
                    dbc.Col([
                        dbc.Label("Include labels (y):"),
                        dcc.Dropdown(
                            id='random-index-clas-labels-'+self.name,
                            options=[{'label': lab, 'value': lab} for lab in self.explainer.labels],
                            multi=True,
                            value=self.labels),
                    ], md=12), hide=self.hide_labels),
            ]),
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
                    ], md=8), hide=self.hide_slider),
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
                    ], md=4), hide=self.hide_pred_or_perc),
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


class CutoffConnector(ExplainerComponent):
    """
    updates cutoff properties of other components given by component list
    e.g. 'precision-cutoff', 'confusionmatrix-cutoff', etc. 
    """
    def __init__(self, explainer, title="Global cutoff",
                        header_mode="none", name=None,
                        cutoff_components=None,
                        hide_cutoff=False, hide_percentile=False,
                        cutoff=0.5, percentile=None):
        super().__init__(explainer, title, header_mode, name)
        self.cutoff_names = [comp.cutoff_name for comp in cutoff_components]

        self.hide_cutoff = hide_cutoff
        self.hide_percentile = hide_percentile
        self.cutoff, self.percentile = cutoff, percentile
        self.register_dependencies(['preds', 'pred_percentiles'])

    def _layout(self):
        return html.Div([
            dbc.Row([
                make_hideable(
                    dbc.Col([
                        html.Div([
                            html.Label('Cutoff prediction probability:'),
                            dcc.Slider(id='cutoffconnector-cutoff-'+self.name, 
                                        min = 0.01, max = 0.99, step=0.01, value=self.cutoff,
                                        marks={0.01: '0.01', 0.25: '0.25', 0.50: '0.50',
                                                0.75: '0.75', 0.99: '0.99'}, 
                                        included=False,
                                        tooltip = {'always_visible' : False})
                        ], style={'margin-bottom': 15}),
                    ]), hide=self.hide_cutoff),
                make_hideable(
                    dbc.Col([
                        html.Div([
                            html.Label('Cutoff percentile of samples:'),
                            dcc.Slider(id='cutoffconnector-percentile-'+self.name, 
                                        min = 0.01, max = 0.99, step=0.01, value=self.percentile,
                                        marks={0.01: '0.01', 0.25: '0.25', 0.50: '0.50',
                                                0.75: '0.75', 0.99: '0.99'}, 
                                        included=False,
                                        tooltip = {'always_visible' : False})
                        ], style={'margin-bottom': 15}),
                    ]), hide=self.hide_percentile),
            ])
        ])

    def _register_callbacks(self, app):
        @app.callback(
            Output('cutoffconnector-cutoff-'+self.name, 'value'),
            [Input('cutoffconnector-percentile-'+self.name, 'value'),
             Input('pos-label', 'value')]
        )
        def update_cutoff(percentile, pos_label):
            if percentile is not None:
                return np.round(self.explainer.cutoff_from_percentile(percentile, pos_label=pos_label), 2)
            raise PreventUpdate

        @app.callback(
            [Output(cut, 'value') for cut in self.cutoff_names],
            [Input('cutoffconnector-cutoff-'+self.name, 'value')]
        )
        def update_cutoffs(cutoff):
            return tuple(cutoff for i in range(len(self.cutoff_names)))


class IndexConnector(ExplainerComponent):
    def __init__(self, input_index, output_indexes):
        self.connector_init()
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


class HighlightConnector(ExplainerComponent):
    def __init__(self, input_highlight, output_highlights):
        self.connector_init()
        self.input_highlight_name = self.highlight_name(input_highlight)
        self.output_highlight_names = self.highlight_name(output_highlights, multi=True)

    @staticmethod
    def highlight_name(highlights, multi=False):
            highlight_name_list = []
            if isinstance(highlights, str):
                highlight_name_list.append(highlights)
            elif isinstance(highlights, ExplainerComponent) and hasattr(highlights, "highlight_name"):
                highlight_name_list.append(highlights.highlight_name)
            elif multi and hasattr(highlights, '__iter__'):
                for highlight in highlights:
                    if isinstance(highlight, str):
                        highlight_name_list.append(highlight)
                    elif isinstance(highlight, ExplainerComponent) and hasattr(highlight, "highlight_name"):
                        highlight_name_list.append(highlight.highlight_name)
                    else:
                        raise ValueError("inputs/outputs should be either str or an ExplainerComponent with a .highlight_name property!"
                                        f"{highlight} is neither!")

            if multi:
                return highlight_name_list
            else:
                return highlight_name_list[0]

    def register_callbacks(self, app):
        @app.callback(
            [Output(highlight_name, 'value') for highlight_name in self.output_highlight_names],
            [Input(self.input_highlight_name, 'value')])
        def update_highlights(highlight):
            return tuple(highlight for i in range(len(self.output_highlight_names)))
