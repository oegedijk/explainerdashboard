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

class ExplainerHeader:
    def __init__(self, explainer, mode="dashboard", title="Explainer Dashboard"):
        assert mode in ["dashboard", "standalone", "hidden", "none"]
        self.explainer = explainer
        self.mode = mode
        self.title = title

    def layout(self):
        dummy_pos_label = html.Div(
                [dcc.Input(id="pos-label")], style=dict(display="none"))
        dummy_tabs = html.Div(id="tabs", style=dict(display="none"))

        title_col = dbc.Col([html.H1(self.title)], width='auto')

        if self.explainer.is_classifier: 
            pos_label_group = dcc.Dropdown(
                    id='pos-label',
                    options = [{'label': label, 'value': i} 
                            for i, label in enumerate(self.explainer.labels)],
                    value = self.explainer.pos_label)
        else:
            pos_label_group = dummy_pos_label
        
        if self.mode=="dashboard":
            return dbc.Container([
                dbc.Row([title_col, dbc.Col([pos_label_group], width=2)],
                justify="start", align="center")
            ], fluid=True)
        elif self.mode=="standalone":
            return dbc.Container([
                dbc.Row([title_col, dbc.Col([pos_label_group], width=2), dummy_tabs], 
                justify="start", align="center")
            ], fluid=True)
        elif self.mode=="hidden":
            return html.Div(
                [pos_label_group, dummy_tabs], 
                style=dict(display="none"))
        elif self.mode=="none":
            return None
