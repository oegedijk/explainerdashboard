__all__ = ['ContributionsTab']

import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_table

from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

import numpy as np

from .dashboard_methods import *

class ContributionsTab:
    def __init__(self, explainer, standalone=False, tab_id="contributions", title='Contributions',
                 n_features=15, round=2, **kwargs):
        self.explainer = explainer
        self.standalone = standalone
        self.tab_id = tab_id
        self.title = title
        
        self.n_features = n_features
        self.round = round
        self.kwargs = kwargs
           
    def layout(self):
        if self.standalone:
            return contributions_layout(self.explainer, title=self.title, standalone=self.standalone, 
                                     n_features=self.n_features, round=self.round)
        else:
            return contributions_layout(self.explainer,  
                                     n_features=self.n_features, round=self.round)
    
    def register_callbacks(self, app):
        contributions_callbacks(self.explainer, app, standalone=self.standalone)


def contributions_layout(explainer, 
            title=None, standalone=False, hide_selector=False, 
            n_features=15, round=2, **kwargs):
    """returns layout for individual contributions tabs
    
    :param explainer: ExplainerBunch to build layout for
    :type explainer: ExplainerBunch
    :type title: str
    :param standalone: when standalone layout, include a a label_store, defaults to False
    :type standalone: bool
    :param hide_selector: if model is a classifier, optionally hide the positive label selector, defaults to False
    :type hide_selector: bool
    :param n_features: Default number of features to display in contributions graph, defaults to 15
    :type n_features: int, optional
    :param round: Precision of floats to display, defaults to 2
    :type round: int, optional
    :rtype: [dbc.Container
    """

    if explainer.is_classifier:
        index_choice_form = dbc.Form([
                dbc.FormGroup([
                    dcc.RangeSlider(
                        id='prediction-range-slider',
                        min=0.0, max=1.0, step=0.01,
                        value=[0.5, 1.0],  allowCross=False,
                        marks={0.0:'0.0', 0.1:'0.1', 0.2:'0.2', 0.3:'0.3', 
                                0.4:'0.4', 0.5:'0.5', 0.6:'0.6', 0.7:'0.7', 
                                0.8:'0.8', 0.9:'0.9', 1.0:'1.0'},)
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
                        inline=True)
                ])        
            ])
    else:
        index_choice_form =  dbc.Form([
                dbc.FormGroup([
                    dcc.RangeSlider(
                        id='prediction-range-slider',
                        min=min(explainer.preds), max=max(explainer.preds), 
                        step=np.float_power(10, -round),
                        value=[min(explainer.preds), max(explainer.preds)],  
                        allowCross=False)
                ]),  
            ])

    return dbc.Container([
    title_and_label_selector(explainer, title, standalone, hide_selector),
    dbc.Row([
        dbc.Col([
            html.H2('Display prediction for:'),
            dbc.Input(id='input-index', 
                        placeholder="Fill in index here...",
                        debounce=True),
            index_choice_form,
            dbc.Button("Random Index", color="primary", id='index-button'),
            dcc.Store(id='index-store'),
        ], width=4),
 
        dbc.Col([
             dcc.Loading(id="loading-model-prediction", 
                         children=[dcc.Markdown(id='model-prediction')]),      
        ], width=4),
        
    ], align="center", justify="between"),

    dbc.Row([
        dbc.Col([
            html.H3('Contributions to prediction'),
            dbc.Label('Number of features to display:', html_for='contributions-size'),
            dcc.Slider(id='contributions-size', 
                min = 1, max = len(explainer.columns), 
                marks={int(i) : str(int(i)) 
                            for i in np.linspace(
                                    1, len(explainer.columns), 6)},
                step = 1, value=min(n_features, len(explainer.columns)),
                ),
            dbc.Label('(click on a bar to display pdp graph)'),
            dcc.Loading(id="loading-contributions-graph", 
                        children=[dcc.Graph(id='contributions-graph')]),
            
            html.Div(id='contributions-size-display', style={'margin-top': 20}),
            html.Div(id='contributions-clickdata'),
        ], width=6),

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
        ], width=6)
    ]),
    dbc.Row([
        dbc.Col([
            html.H3('Contributions to prediction'),
            dbc.Label('(table format)'),
            dash_table.DataTable(
                id='contributions_table',
                style_cell={'fontSize':20, 'font-family':'sans-serif'},
                columns=[{'id': c, 'name': c} 
                            for c in ['Reason', 'Effect']],
            ),    
        ], width=10),
    ]),
    ], fluid=True)


def contributions_callbacks(explainer, app, 
            standalone=False, round=2, **kwargs):


    if explainer.is_classifier:
        if standalone:
            label_selector_register_callback(explainer, app)
        @app.callback(
            Output('input-index', 'value'),
            [Input('index-button', 'n_clicks')],
            [State('prediction-range-slider', 'value'),
             State('include-labels', 'value')]
        )
        def update_input_index(n_clicks, slider_range, include):
            y = None
            if include=='neg': y = 0 
            elif include=='pos': y = 1
            return_str = True if explainer.idxs is not None else False
            idx = explainer.random_index(
                    y_values=y, pred_proba_min=slider_range[0], pred_proba_max=slider_range[1],
                    return_str=return_str)
            if idx is not None:
                return idx
            raise PreventUpdate
    else:
        @app.callback(
            Output('input-index', 'value'),
            [Input('index-button', 'n_clicks')]
        )
        def update_input_index(n_clicks):
            y = None
            return_str = True if explainer.idxs is not None else False
            idx = explainer.random_index(return_str=return_str)
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
        Output('contributions-size-display', 'children'),
        [Input('contributions-size', 'value')])
    def display_value(contributions_size):
        return f"Displaying top {contributions_size} features."

    @app.callback(
        [Output('model-prediction', 'children'),
         Output('contributions-graph', 'figure'),
         Output('contributions_table', 'data')],
        [Input('index-store', 'data'),
         Input('contributions-size', 'value'),
         Input('label-store', 'data')]
    )
    def update_output_div(index, topx, pos_label):
        if index is None:
            raise PreventUpdate
        int_idx = explainer.get_int_idx(index)
        if explainer.is_classifier:
            def display_probas(pred_probas_raw, labels, round=2):
                assert len(pred_probas_raw.shape)==1 and len(pred_probas_raw) ==len(labels)
                for i in range(len(labels)):
                    yield '##### ' + labels[i] + ': ' + str(np.round(100*pred_probas_raw[i], round))+ '%\n'

            model_prediction = f"# Prediction for {index}:\n" 
            for pred in display_probas(explainer.pred_probas_raw[int_idx], explainer.labels, round):
                model_prediction += pred
            if isinstance(explainer.y[0], int) or isinstance(explainer.y[0], np.int64):
                model_prediction += f"##### Actual Outcome: {explainer.labels[explainer.y[int_idx]]}"
        else:
            model_prediction = f"# Prediction for {index}:\n" \
                                + f"##### Prediction: {np.round(explainer.preds[int_idx], round)}\n"
            model_prediction += f"##### Actual Outcome: {np.round(explainer.y[int_idx], round)}"

        plot = explainer.plot_shap_contributions(index, topx=topx, round=round)
        summary_table = explainer.contrib_summary_df(index, topx=topx, round=round).to_dict('records')
        return (model_prediction, plot, summary_table)

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
        return explainer.plot_pdp(col, idx, sample=100)