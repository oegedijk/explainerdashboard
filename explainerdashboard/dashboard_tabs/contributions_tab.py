__all__ = ['contributions_tab', 'contributions_tab_register_callbacks']

import dash
import dash_core_components as dcc
import dash_daq as daq
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_table

from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

import numpy as np

def contributions_tab(explainer, n_features=15, **kwargs):
    return dbc.Container([
    dbc.Row([
        dbc.Col([
            dbc.Label('Fill specific index'),
            dbc.Input(id='input-index', placeholder="Fill in index here...",
                        debounce=True),
            html.Div([
                dcc.RangeSlider(
                    id='prediction-range-slider',
                    min=0.0, max=1.0, step=0.01,
                    value=[0.5, 1.0],  allowCross=False,
                    marks={0.0:'0.0', 0.1:'0.1', 0.2:'0.2', 0.3:'0.3', 
                            0.4:'0.4', 0.5:'0.5', 0.6:'0.6', 0.7:'0.7', 
                            0.8:'0.8', 0.9:'0.9', 1.0:'1.0'})
            ], style={'margin': 20}),
            html.Div([
                dcc.RadioItems(
                    id='include-labels',
                    options=[
                        {'label': explainer.labels[1], 'value': 'pos'},
                        {'label': explainer.labels[0], 'value': 'neg'},
                        {'label': 'Both/either', 'value': 'any'},
                    ],
                    value='any',
                    labelStyle={'display': 'inline-block'}
                )
            ], style={'margin': 20}),
            html.Div([
                html.Button('random index', id='index-button'),
            ], style={'margin': 30})
        ], width=4),
        dbc.Col([
             dcc.Loading(id="loading-model-prediction", 
                         children=[dcc.Markdown(id='model-prediction')]),      
        ]),
        dcc.Store(id='index-store'),
    ], justify="between"),

    dbc.Row([
        dbc.Col([
            html.Div([
                html.Label('Number of features to display:'),
                dcc.Slider(id='contributions-size', 
                    min = 1, max = len(explainer.columns), 
                    marks={int(i) : str(int(i)) 
                                for i in np.linspace(
                                        1, len(explainer.columns), 6)},
                    step = 1, value=min(n_features, len(explainer.columns))),
            ]),
            html.Div(id='contributions-size-display', style={'margin-top': 20})
        ], width=4),
    ]),

    dbc.Row([
        dbc.Col([
            html.Div([
                html.H3('Contributions to prediction'),
                html.Label('(click on a bar to display pdp graph)'),
                dcc.Loading(id="loading-contributions-graph", 
                         children=[dcc.Graph(id='contributions-graph')])
            ], style={'margin': 30}),
            html.Div(id='contributions-clickdata')

        ], width=6),
        dbc.Col([
            html.Div([
                html.H3('Partial Dependence Plot'),
                html.Label("Plot partial dependence plot (\'what if?\') for column:"),
                dcc.Dropdown(id='pdp-col', 
                    options=[{'label': col, 'value':col} 
                                for col in explainer.mean_abs_shap_df(cats=True)\
                                                            .Feature.tolist()],
                    value=explainer.mean_abs_shap_df(cats=True).Feature[0]),
                dcc.Loading(id="loading-pdp-graph", 
                        children=[dcc.Graph(id='pdp-graph')]),
            ], style={'margin': 30})
        ], width=6)
    ]),
    dbc.Row([
        dbc.Col([
            html.H3('Contributions to prediction'),
            html.Label('(table format)'),
            dash_table.DataTable(
                id='contributions_table',
                style_cell={'fontSize':20, 'font-family':'sans-serif'},
                columns=[{'id': c, 'name': c} 
                            for c in ['Reason', 'Effect']],
            ),
            
        ], width=10),
    ]),
    ], fluid=True)


def contributions_tab_register_callbacks(explainer, app):
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
        Input('contributions-size', 'value')]
    )
    def update_output_div(idx, topx):
        int_idx = explainer.get_int_idx(idx)
        model_prediction = f"##### Index: {idx}\n"\
                            + f"## Prediction: {np.round(100*explainer.pred_probas[int_idx],2)}% {explainer.labels[1]}\n"
        if isinstance(explainer.y[0], int) or isinstance(explainer.y[0], np.int64):
            model_prediction += f"## Actual Outcome: {explainer.labels[explainer.y[int_idx]]}"
        plot = explainer.plot_shap_contributions(idx, topx=topx)
        summary_table = explainer.contrib_summary_df(idx, topx=topx).to_dict('records')
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
        Input('pdp-col', 'value')]
    )
    def update_pdp_graph(idx, col):
        return explainer.plot_pdp(col, idx, sample=100)