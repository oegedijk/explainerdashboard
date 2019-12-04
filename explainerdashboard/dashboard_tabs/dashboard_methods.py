import inspect

import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html

from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate


# Stolen from https://www.fast.ai/2019/08/06/delegation/
# then extended to deal with multiple inheritance
def delegates(to=None, keep=False):
    "Decorator: replace `**kwargs` in signature with params from `to`"
    def _f(f):
        from_f = f.__init__ if to is None else f
        sig = inspect.signature(from_f)
        sigd = dict(sig.parameters)
        k = sigd.pop('kwargs')
        if to is None:
            for base_cls in f.__bases__:
                to_f = base_cls.__init__
                s2 = {k:v for k,v in inspect.signature(to_f).parameters.items()
                    if v.default != inspect.Parameter.empty and k not in sigd}
                sigd.update(s2)
        else:
            to_f = to
            s2 = {k:v for k,v in inspect.signature(to_f).parameters.items()
                if v.default != inspect.Parameter.empty and k not in sigd}
            sigd.update(s2)
        if keep: sigd['kwargs'] = k
        from_f.__signature__ = sig.replace(parameters=sigd.values())
        return f
    return _f


def title_and_label_selector(explainer, title=None, include_label_store=False, hide_selector=False):
    """
    Returns a title dbc.Row with positive label selector in case of a classifier.
    
    """
    if not include_label_store and title is None:
        return None
    
    title_col = dbc.Col([html.H1(title)], width='auto') if title is not None else None
    
    if not include_label_store:
        return dbc.Row([title_col], justify="start", align="center")
    
    if explainer.is_classifier: 
        #classifier can show label selector, but still optionally hide it
        hidden_style = {'display': 'none'} if hide_selector else None
        return dbc.Row([
            dbc.Col([html.H1(title)], width='auto'),
            dbc.Col([
                html.Div([
                    dcc.Dropdown(
                        id='pos-label-selector',
                        options = [{'label': label, 'value': label} 
                                        for label in explainer.labels],
                        value = explainer.pos_label_str
                     )
                ], style=hidden_style)
            ], width=2),
            dcc.Store(id='label-store')
        ], justify="start", align="center")
    else: 
        # regression never shows label selector, but still includes the label store 
        # for callback compatability
        return dbc.Row([
            dbc.Col([html.H1(title)], width='auto'),
            dcc.Store(id='label-store')
        ], justify="start", align="center")

        
def label_selector_register_callback(explainer, app, **kwargs):
    if explainer.is_classifier:
        @app.callback(
            Output("label-store", "data"),
            [Input('pos-label-selector', 'value')]
        )
        def change_positive_label(pos_label):
            print('pos label:', pos_label)
            if pos_label is not None:
                explainer.pos_label = pos_label
                print('updated pos label')
                return pos_label
            raise PreventUpdate
            