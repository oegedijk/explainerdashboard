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

class DummyComponent:
    def __init__(self):
        pass

    def layout(self):
        return None

    def register_callbacks(self, app):
        pass

class TitleAndLabelSelector:
    def __init__(self, explainer, title="Explainer Dashboard", 
                    title_only=False, hidden=False, 
                    dummy_label_store=False, dummy_tabs=False):
        self.explainer = explainer
        self.title = title
        self.title_only = title_only
        self.hidden = hidden
        self.dummy_label_store = dummy_label_store
        self.dummy_tabs = dummy_tabs

    def layout(self):
        if self.hidden: 
            return html.Div([
                html.Div(id='tabs', style="none") if self.dummy_tabs else None,
                dcc.Store(id='label-store'),
            ], style="none")
    
        title_col = dbc.Col([html.H1(self.title)], width='auto')
    
        if self.title_only:
            return dbc.Container([
                dbc.Row([
                    title_col,
                    html.Div(id='tabs', style="none") if self.dummy_tabs else None,
                ], justify="start", align="center")
            ], fluid=True)
        elif self.dummy_label_store or not self.explainer.is_classifier:
            return dbc.Container([
                dbc.Row([
                    title_col,
                    html.Div(id='tabs', style="none") if self.dummy_tabs else None,
                    dcc.Store(id='label-store')
                ], justify="start", align="center")
            ], fluid=True)
        elif self.explainer.is_classifier: 
            return dbc.Container([
                dbc.Row([
                    title_col,
                    dbc.Col([
                            dcc.Dropdown(
                                id='pos-label-selector',
                                options = [{'label': label, 'value': i} 
                                                for i, label in enumerate(self.explainer.labels)],
                                value = self.explainer.pos_label
                            )
                    ], width=2),
                    html.Div(id='tabs', style="none") if self.dummy_tabs else None,
                    dcc.Store(id='label-store')
                ], justify="start", align="center")
            ], fluid=True)
        else:
            return None
 
    def register_callbacks(self, app, **kwargs):
        if (self.explainer.is_classifier and
            not self.dummy_label_store and
            not self.title_only and
            not self.hidden):
            @app.callback(
                Output("label-store", "data"),
                [Input('pos-label-selector', 'value')]
            )
            def change_positive_label(pos_label):
                if pos_label is not None:
                    return pos_label
                return self.explainer.pos_label
