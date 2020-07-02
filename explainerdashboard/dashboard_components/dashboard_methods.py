__all__ = [
    'DummyComponent',
    'ExplainerHeader',
    'ExplainerComponent',
    'make_hideable',
]

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
        expect the presence of both a 'pos-label' and a 'tabs' dash element, so you
        have to make sure there is at least an element with that name.

        If you have standalone page without tabs ans so without a dcc.Tabs 
        component, an ExplainerHeader(mode='standalone') will insert a dummy 
        'tabs' hidden div.

        Even if you don't need or want a positive label selector in your layout, 
        you still need a 'pos-label' input element in order for the callbacks
        to be validated. an ExplainerHeader(mode='hidden'), will add a hidden
        div with both a 'tabs' and a 'pos-label' element.

        For a subcomponent, you don't need to define either a 'pos-label' or
        a 'tabs' element (these are taken care off by the overall layout),
        so ExplainerHeader(mode='none'), will simply add a dummy hidden layout.
    """
    def __init__(self, explainer, mode="dashboard", title="Explainer Dashboard"):
        """Insert appropriate header layout into explainable dashboard layout.

        Args:
            explainer ([type]): [description]
            mode (str, {'dashboard', 'standalone', 'hidden', 'none', optional): 
                "dashboard": Display both title and label selector.
                    Used for ExplainerDashboard.
                "standalone": display title and label selector, insert dummy 'tabs'.
                    Used for ExplainerTabs or standalone single layouts.
                "hidden": don't display header but insert 'pos-label' and 'tabs'.
                    Used for InlineExplainer.
                "none": don't generate any header at all.
                    Used for ExplainerComponents.. Defaults to "dashboard".
            title (str, optional): Title to display in header. 
                    Defaults to "Explainer Dashboard".
        """
        assert mode in ["dashboard", "standalone", "hidden", "none"]
        self.explainer = explainer
        self.mode = mode
        self.title = title

    def layout(self):
        """html.Div() with elements depending on self.mode:
            'dashboard': title+positive label selector
            'standalone': title+positive label selector + hidden 'tabs' element
            'hidden': hidden 'post-label' and 'tabs' element
            'none': empty layout
        """
        dummy_pos_label = html.Div(
                [dcc.Input(id="pos-label")], style=dict(display="none"))
        dummy_tabs = html.Div(dcc.Input(id="tabs"), style=dict(display="none"))

        title_col = dbc.Col([html.H1(self.title)], width='auto')

        if self.explainer.is_classifier:
            pos_label_group = html.Div([
                dbc.Label("Positive class:", html_for="pos-label"),
                dcc.Dropdown(
                    id='pos-label',
                    options = [{'label': label, 'value': i}
                            for i, label in enumerate(self.explainer.labels)],
                    value = self.explainer.pos_label)
            ])
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
    """ExplainerComponent is a bundle of a dash layout and callbacks that
    make use of an Explainer object. 

    An ExplainerComponent can have ExplainerComponent subcomponents.

    ExplainerComponent makes sure that:
    1. an appropriate header is inserted
    2. callbacks are registered
    3. lazily calculated dependencies are registered (even nested ones!)
    
    Each ExplainerComponent adds a unique uuid name string to all elements, so 
    that there is never a name clash even with multiple ExplanerComponents of 
    the same type in a layout. 

    Important:
        define your layout in _layout() and your callbacks in _register_callbacks()
        ExplainerComponent will return a layout() consisting of both
        header and _layout(), and register callbacks of subcomponents in addition
        to _register_callbacks() when calling register_callbacks()
    """
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
        """register subcomponents so that their callbacks will be registered
        and dependencies can be tracked"""
        if not hasattr(self, '_components'):
            self._components = []
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
        """register dependencies: lazily calculated explainer properties that
        you want to calculate *before* starting the dashboard"""
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
        """returns a list of unique dependencies of the component 
        and all subcomponents"""
        if not hasattr(self, '_dependencies'):
            self._dependencies = []
        if not hasattr(self, '_components'):
            self._components = []
        deps = self._dependencies
        for comp in self._components:
            deps.extend(comp.dependencies)
        deps = list(set(deps))
        return deps

    def calculate_dependencies(self):
        """calls all properties in self.dependencies so that they get calculated
        up front. This is useful to do before starting a dashboard, so you don't
        compute properties multiple times in parallel."""
        for dep in self.dependencies:
            try:
                _ = getattr(self.explainer, dep)
            except:
                ValueError(f"Failed to generate dependency '{dep}': "
                    "Failed to calculate or retrieve explainer property explainer.{dep}...")

    def _layout(self):
        """layout to be defined by the particular ExplainerComponent instance.
        All element id's should append +self.name to make sure they are unique."""
        return None

    def layout(self):
        "returns a combination of the header and _layout()"
        return html.Div([
            self.header.layout(),
            self._layout()
        ])

    def _register_callbacks(self, app):
        """register callbacks particular to a ExplainerComponent"""
        pass

    def register_callbacks(self, app):
        """register callbacks of this component and all subcomponents"""
        for comp in self._components:
            comp.register_callbacks(app)
        self._register_callbacks(app)

def make_hideable(element, hide=False):
    """helper function to optionally not display an element in a layout.

    This is used for all the hide_ flags in ExplainerComponent constructors.
    e.g. hide_cutoff=True to hide a cutoff slider from a layout:
    
    Example:
        make_hideable(dbc.Col([cutoff.layout()]), hide=hide_cutoff)

    Args:
        hide(bool): wrap the element inside a hidden html.div. If the element 
                    is a dbc.Col or a dbc.FormGroup, wrap element.children in
                    a hidden html.Div instead. Defaults to False.
    """ 
    if hide:
        if isinstance(element, dbc.Col) or isinstance(element, dbc.FormGroup):
            return html.Div(element.children, style=dict(display="none"))
        else:
            return html.Div(element, style=dict(display="none"))
    else:
        return element


