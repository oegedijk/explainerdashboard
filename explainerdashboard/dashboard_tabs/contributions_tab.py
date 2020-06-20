__all__ = ['ContributionsTab']

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

class ContributionsTab:
    def __init__(self, explainer, 
                    standalone=False, hide_title=False,
                    tab_id="contributions", title='Contributions',
                    n_features=15, round=2, **kwargs):
        self.explainer = explainer
        self.standalone = standalone
        self.tab_id = tab_id
        self.title = title
        
        self.n_features = n_features
        self.round = round
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
            contributions_layout(self.explainer,  
                    n_features=self.n_features, round=self.round, **self.kwargs)
        ], fluid=True)
    
    def register_callbacks(self, app):
        self.label_selector.register_callbacks(app)
        contributions_callbacks(self.explainer, app, round=self.round)


def contributions_layout(explainer, n_features=15, round=2, **kwargs):
    """returns layout for individual contributions tabs
    
    :param explainer: ExplainerBunch to build layout for
    :type explainer: ExplainerBunch
    :param n_features: Default number of features to display in contributions graph, defaults to 15
    :type n_features: int, optional
    :param round: Precision of floats to display, defaults to 2
    :type round: int, optional
    :rtype: [dbc.Container
    """

    if explainer.is_classifier:
        index_choice_form = dbc.Form([

                dbc.FormGroup([
                    html.Div([
                        dbc.Label('Range to select from (prediction probability or prediction percentile):', 
                                    html_for='prediction-range-slider'),
                        dcc.RangeSlider(
                        id='prediction-range-slider',
                        min=0.0, max=1.0, step=0.01,
                        value=[0.5, 1.0],  allowCross=False,
                        marks={0.0:'0.0', 0.1:'0.1', 0.2:'0.2', 0.3:'0.3', 
                                0.4:'0.4', 0.5:'0.5', 0.6:'0.6', 0.7:'0.7', 
                                0.8:'0.8', 0.9:'0.9', 1.0:'1.0'},
                        tooltip = {'always_visible' : False})
                    ], style={'margin-bottom':25})
                    
                ]),
                dbc.FormGroup([
                    dbc.RadioItems(
                        id='include-labels',
                        options=[
                            {'label': explainer.pos_label_str, 'value': 'pos'},
                            {'label': 'Not ' + explainer.pos_label_str, 'value': 'neg'},
                            {'label': 'Both/either', 'value': 'any'},
                        ],
                        value='any',
                        inline=True),
                    dbc.RadioItems(
                        id='preds-or-ranks',
                        options=[
                            {'label': 'Use predictions', 'value': 'preds'},
                            {'label': 'Use percentiles', 'value': 'ranks'},
                        ],
                        value='preds',
                        inline=True)
                ])        
            ])
    else:
        index_choice_form =  dbc.Form([
                dbc.FormGroup([
                    html.Div([
                        html.Div([
                            dbc.Label('Range of predicted outcomes to select from:', 
                                    html_for='prediction-range-slider'),
                            dcc.RangeSlider(
                                id='prediction-range-slider',
                                min=min(explainer.preds), max=max(explainer.preds), 
                                step=np.float_power(10, -round),
                                value=[min(explainer.preds), max(explainer.preds)], 
                                marks={min(explainer.preds):str(np.round(min(explainer.preds), round)),
                                        max(explainer.preds):str(np.round(max(explainer.preds), round))}, 
                                allowCross=False,
                                tooltip = {'always_visible' : False}
                            )

                        ], style={'margin-bottom':25})
                        
                    ]),  
                ], style={'margin-bottom':25}),
            ])

    return dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H2('Display prediction for:'),
            dbc.Input(id='input-index', 
                        placeholder="Fill in index here...",
                        debounce=True),
            index_choice_form,
            dbc.Button("Random Index", color="primary", id='index-button'),
            dcc.Store(id='index-store'),
        ], md=6),
 
        dbc.Col([
             dcc.Loading(id="loading-model-prediction", 
                         children=[dcc.Markdown(id='model-prediction')]),      
        ], md=6),
        
    ], align="start", justify="between"),

    dbc.Row([
        dbc.Col([
            html.H3('Contributions to prediction'),
            dbc.Label('Number of features to display:', html_for='contributions-size'),
            html.Div([
                dcc.Slider(id='contributions-size', 
                    min = 1, max = len(explainer.columns), 
                    marks = {int(i) : str(int(i)) 
                                for i in np.linspace(
                                        1, len(explainer.columns_cats), 6)},
                    step = 1, value=min(n_features, len(explainer.columns_cats)),
                    tooltip = {'always_visible' : False}
                ),
            ], style={'margin-bottom':25}),
            
            dbc.Label('(click on a bar to display pdp graph)'),
            dcc.Loading(id="loading-contributions-graph", 
                        children=[dcc.Graph(id='contributions-graph')]),
            
            html.Div(id='contributions-size-display', style={'margin-top': 20}),
            html.Div(id='contributions-clickdata'),
        ], md=6),

        dbc.Col([
            html.H3('Partial Dependence Plot'),
            dbc.Label("Plot partial dependence plot (\'what if?\') for column:", html_for='pdp-col'),
            dcc.Dropdown(id='pdp-col', 
                options=[{'label': col, 'value':col} 
                            for col in explainer.mean_abs_shap_df(cats=True)\
                                                        .Feature.tolist()],
                value=explainer.mean_abs_shap_df(cats=True).Feature[0]),
            dcc.Loading(id="loading-pdp-graph", 
                    children=[dcc.Graph(id='pdp-graph')]),
        ], md=6)
    ]),
    dbc.Row([
        dbc.Col([
            html.H3('Contributions to prediction'),
            dash_table.DataTable(
                id='contributions_table',
                style_cell={'fontSize':20, 'font-family':'sans-serif'},
                columns=[{'id': c, 'name': c} 
                            for c in ['Reason', 'Effect']],
                      
            ),    
        ], md=6),
    ]),
    ], fluid=True)


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
         Input('label-store', 'data')]
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
         Input('label-store', 'data')]
    )
    def update_pdp_graph(idx, col, pos_label):
        return explainer.plot_pdp(col, idx, sample=100, pos_label=pos_label)