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


class EmptyLayout:
    def __init__(self):
        pass

    def layout(self):
        return None

    def register_callbacks(self, app):
        pass

class TitleAndLabelSelector:
    def __init__(self, explainer, title="Explainer Dashboard", 
                    title_only=False, label_store_only=False, dummy=False):
        self.explainer = explainer
        self.title = title
        self.title_only, self.label_store_only = title_only, label_store_only
        self.dummy = dummy

    def layout(self):
        if self.dummy: return None
    
        title_col = dbc.Col([html.H1(self.title)], width='auto')
    
        if self.title_only:
            return dbc.Container([
                dbc.Row([
                    title_col,
                ], justify="start", align="center")
            ], fluid=True)
        elif self.label_store_only or not self.explainer.is_classifier:
            return dbc.Container([
                dbc.Row([
                    title_col,
                    dcc.Store(id='label-store')
                ], justify="start", align="center")
            ], fluid=True)
        elif self.explainer.is_classifier: 
            #classifier can show label selector, but still optionally hide it
            return dbc.Container([
                dbc.Row([
                    title_col,
                    dbc.Col([
                            dcc.Dropdown(
                                id='pos-label-selector',
                                options = [{'label': label, 'value': label} 
                                                for label in self.explainer.labels],
                                value = self.explainer.pos_label_str
                            )
                    ], width=2),
                    dcc.Store(id='label-store')
                ], justify="start", align="center")
            ], fluid=True)
        else:
            return None
 

    def register_callbacks(self, app, **kwargs):
        if self.explainer.is_classifier and not self.label_store_only:
            @app.callback(
                Output("label-store", "data"),
                [Input('pos-label-selector', 'value')]
            )
            def change_positive_label(pos_label):
                if pos_label is not None:
                    self.explainer.pos_label = pos_label
                    return pos_label
                return self.explainer.pos_label
