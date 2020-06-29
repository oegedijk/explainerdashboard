from abc import ABC
import inspect

import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html

import shortuuid


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
                s2 = {k: v for k, v in inspect.signature(to_f).parameters.items()
                    if v.default != inspect.Parameter.empty and k not in sigd}
                sigd.update(s2)
        else:
            to_f = to
            s2 = {k: v for k, v in inspect.signature(to_f).parameters.items()
                if v.default != inspect.Parameter.empty and k not in sigd}
            sigd.update(s2)
        if keep: 
            sigd['kwargs'] = k
        from_f.__signature__ = sig.replace(parameters=sigd.values())
        return f
    return _f


class DummyComponent:
    def __init__(self):
        pass

    def layout(self):
        pass

    def register_callbacks(self, app):
        pass


class ExplainerHeader:
    """
    Generates header layout with a title and, for classification models, a
        positive label selector. The callbacks for most of the ExplainerComponents
        expect the presence of both a 'pos-label' and a 'tabs' component, so you
        have to make sure there is at least a component with that name.

        If you have  standalone page without a dcc.Tabs component, the header
        inserts a dummy 'tabs' hidden div.

        If you set the header mode to "hidden", the header will insert a 'pos-label'
        and 'tabs' dummy component in a hidden div.

    Four modes:
        "dashboard": Display both title and label selector.
            Used for ExplainerDashboard.
        "standalone": display title and label selector, insert dummy 'tabs'.
            Used for ExplainerTabs.
        "hidden": don't display title but insert 'pos-label' and 'tabs'.
            Used for InlineExplainer.
        "none": don't generate any header at all.
            Used for ExplainerComponents.
    """
    def __init__(self, explainer, mode="dashboard", title="Explainer Dashboard"):
        assert mode in ["dashboard", "standalone", "hidden", "none"]
        self.explainer = explainer
        self.mode = mode
        self.title = title

    def layout(self):
        dummy_pos_label = html.Div(
                [dcc.Input(id="pos-label")], style=dict(display="none"))
        dummy_tabs = html.Div(dcc.Input(id="tabs"), style=dict(display="none"))

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


class ExplainerComponent(ABC):
    def __init__(self, explainer, title=None, header_mode="none", name=None):
        self.explainer = explainer
        self.title = title
        self.header = ExplainerHeader(explainer, title=title, mode=header_mode)
        self.name = name
        if self.name is None:
            self.name = shortuuid.ShortUUID().random(length=10)
        self._components = []
        self._dependencies = []

    def register_components(self, *components):
        for comp in components:
            if isinstance(comp, ExplainerComponent):
                self._components.append(comp)
            elif hasattr(comp, '__iter__'):
                for subcomp in comp:
                    if isinstance(subcomp, ExplainerComponent):
                        self._components.append(subcomp)
                    else:
                        print(f"{subcomp.__name__} is not an ExplainerComponent so not adding to self.components")
            else:
                print(f"{comp.__name__} is not an ExplainerComponent so not adding to self.components")

    def register_dependencies(self, *dependencies):
        for dep in dependencies:
            if isinstance(dep, str):
                self._dependencies.append(dep)
            elif hasattr(dep, '__iter__'):
                for subdep in dep:
                    if isinstance(subdep, str):
                        self._dependencies.append(subdep)
                    else:
                        print(f"{subdep.__name__} is not a str so not adding to self.dependencies")
            else:
                print(f"{dep.__name__} is not a str or list of str so not adding to self.dependencies")

    @property
    def dependencies(self):
        deps = self._dependencies
        for comp in self._components:
            deps.extend(comp.dependencies)
        deps = list(set(deps))
        return deps

    def calculate_dependencies(self):
        for dep in self.dependencies:
            try:
                _ = getattr(self.explainer, dep)
            except:
                ValueError(f"Failed to generate dependency '{dep}': "
                    "Failed to calculate or retrieve explainer property explainer.{dep}...")

    def _layout(self):
        return None

    def layout(self):
        return html.Div([
            self.header.layout(),
            self._layout()
        ])

    def _register_callbacks(self, app):
        pass

    def register_callbacks(self, app):
        for comp in self._components:
            comp.register_callbacks(app)
        self._register_callbacks(app)

def make_hideable(element, hide=False):
    """helper function to optionally not display an element in a layout.

    if hide=True: return a hidden div containing element
    else: return element
    
    if element is a dbc.Col([]), put col.children in hidden div instead.
    """ 
    if hide:
        if isinstance(element, dbc.Col) or isinstance(element, dbc.FormGroup):
            return html.Div(element.children, style=dict(display="none"))
        else:
            return html.Div(element, style=dict(display="none"))
    else:
        return element


