__all__ = [
    'delegates_kwargs',
    'delegates_doc',
    'DummyComponent',
    'ExplainerHeader',
    'ExplainerComponent',
    'make_hideable',
    'ExplainerTabsLayout',
    'ExplainerPageLayout'
]

from abc import ABC
import inspect
import types

import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html

import shortuuid


# Stolen from https://www.fast.ai/2019/08/06/delegation/
# then extended to deal with multiple inheritance
def delegates_kwargs(to=None, keep=False):
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


def delegates_doc(to=None, keep=False):
    "Decorator: replace `__doc__` with `__doc__` from `to`"
    def _f(f):
        from_f = f.__init__ if to is None else f
        if to is None:
            for base_cls in f.__bases__:
                to_f = base_cls.__init__
        else:
            if isinstance(to, types.FunctionType):
                to_f = to
            else:
                to_f = to.__init__
        from_f.__doc__ = to_f.__doc__
        return f
    return _f


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

class DummyComponent:
    def __init__(self):
        pass

    def layout(self):
        pass

    def register_callbacks(self, app):
        pass


class ExplainerHeader:
    """
    Generates a header layout with a title and, for classification models, a
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
    def __init__(self, explainer, mode="none", title="Explainer Dashboard",
                    hide_title=False, hide_selector=False, fluid=True):
        """Insert appropriate header layout into explainable dashboard layout.

        Args:
            explainer ([type]): [description]
            mode (str, {'dashboard', 'standalone', 'hidden', 'none', optional): 
                "dashboard": Display both title and label selector.
                    Used for ExplainerDashboard.
                "dashboard_titleonly": Display title and add dummy label selector
                "standalone": display title and label selector, insert dummy 'tabs'.
                    Used for ExplainerTabs or standalone single layouts.
                "hidden": don't display header but insert dummy elements 'pos-label' and 'tabs'. 
                    Used for InlineExplainer.
                "none": don't generate any header at all.
                    Used for sub ExplainerComponents.. Defaults to "none".
            title (str, optional): Title to display in header. 
                    Defaults to "Explainer Dashboard".
            hide_title(bool, optional): Hide the title. Defaults to False.
            hide_selector(bool, optional): Hide the selector. Defaults to False.
            fluid(bool, optional): whether to "stretch" (fluid=True) the layout
                    of the dbc.Container([])
            
        """
        assert mode in ["dashboard", "standalone", "hidden", "none"]
        self.explainer = explainer
        self.mode = mode
        self.title = title
        self.hide_title, self.hide_selector = hide_title, hide_selector
        self.fluid = fluid

    def layout(self):
        """html.Div() with elements depending on self.mode:

        - 'dashboard': title+positive label selector
        - 'standalone': title+positive label selector + hidden 'tabs' element
        - 'hidden': hidden 'pos-label' and 'tabs' element
        - 'none': empty layout
        """
        dummy_pos_label = html.Div(
                [dcc.Input(id="pos-label")], style=dict(display="none"))
        dummy_tabs = html.Div(dcc.Input(id="tabs"), style=dict(display="none"))

        title_col = make_hideable(
            dbc.Col([html.H1(self.title)], width='auto'), hide=self.hide_title)


        if self.explainer.is_classifier:
            pos_label_group = make_hideable(
                dbc.Col([
                    dbc.Label("Positive class:", html_for="pos-label"),
                    dcc.Dropdown(
                        id='pos-label',
                        options = [{'label': label, 'value': i}
                                for i, label in enumerate(self.explainer.labels)],
                        value = self.explainer.pos_label)
                ], width=2), 
                hide=self.hide_selector)
        else:
            pos_label_group = html.Div(
                [dcc.Input(id="pos-label")], style=dict(display="none"))

        if self.mode=="dashboard":
            return dbc.Container([
                dbc.Row([title_col, pos_label_group],
                justify="start", align="center")
            ], fluid=self.fluid)
        elif self.mode=="standalone":
            return dbc.Container([
                dbc.Row([title_col, pos_label_group, dummy_tabs],
                justify="start", align="center")
            ], fluid=self.fluid)
        elif self.mode=="hidden":
            return html.Div(
                    dbc.Row([pos_label_group, dummy_tabs])
            , style=dict(display="none"))
        elif self.mode=="none":
            return None


class ExplainerComponent(ABC):
    """ExplainerComponent is a bundle of a dash layout and callbacks that
    make use of an Explainer object. 

    An ExplainerComponent can have ExplainerComponent subcomponents, that
    you register with register_components(). If the component depends on 
    certain lazily calculated Explainer properties, you can register these
    with register_dependencies().

    ExplainerComponent makes sure that:

    1. An appropriate header is inserted into .layout()
    2. Callbacks of subcomponents are registered.
    3. Lazily calculated dependencies (even of subcomponents) can be calculated.
    
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
        """initialize the ExplainerComponent

        Args:
            explainer (Explainer): explainer object constructed with e.g.
                        ClassifierExplainer() or RegressionExplainer()
            title (str, optional): Title of tab or page. Defaults to None.
            header_mode (str,{"standalone", "hidden" or "none"}, optional):
                        determines what kind of ExplainerHeader to insert into
                        the layout. For a component that forms the entire page, 
                        use 'standalone'. If you don't want to see the title and
                        pos label selector, use 'hidden'. If this is a subcomponent
                        of a larger page, use 'none'.
                        Defaults to "none".
            name (str, optional): unique name to add to Component elements. 
                        If None then random uuid is generated to make sure 
                        it's unique. Defaults to None.

        """
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
        if not hasattr(self, 'header'):
            return self._layout()
        else:
            return html.Div([
                self.header.layout(),
                self._layout()
            ])

    def _register_callbacks(self, app):
        """register callbacks specific to this ExplainerComponent"""
        pass

    def register_callbacks(self, app):
        """First register callbacks of all subcomponents, then call
        _register_callbacks(app)
        """
        if not hasattr(self, '_components'):
            self._components = []
        for comp in self._components:
            comp.register_callbacks(app)
        self._register_callbacks(app)




def instantiate_component(component, explainer, header_mode="none", **kwargs):
    """Returns an instantiated ExplainerComponent.
    
    If the component input is just a class definition, instantiate it with
    explainer, header_mode and kwargs.
    If it is already an ExplainerComponent instance, set header_mode and return it.
    If it is any other instance with layout and regiuster_components methods,
    then add a name property and return it. 

    Args:
        component ([type]): Either a class definition or instance
        explainer ([type]): An Explainer object that will be used to instantiate class definitions
        header_mode (str, optional): Header_mode to assign to component. Defaults to "none".
        kwargs: kwargs will be passed on to the instance

    Raises:
        ValueError: if component is not a subclass or instance of ExplainerComponent,
                or is an instance without layout and register_callbacks methods

    Returns:
        [type]: instantiated component
    """

    if inspect.isclass(component) and issubclass(component, ExplainerComponent):
        return component(explainer, header_mode=header_mode, **kwargs)
    elif isinstance(component, ExplainerComponent):
        component.header.mode = header_mode
        return component
    elif (not inspect.isclass(component)
          and hasattr(component, "layout")):
        if not (hasattr(component, "name") and isinstance(component.name, str)):
            try:
                component_name  = component.__name__
            except:
                component_name = shortuuid.ShortUUID().random(length=10)
            print(f"Warning: setting {component}.name to {component_name}")
            component.name = component_name
        if not hasattr(component, "title"):
            print(f"Warning: setting {component}.title to 'CustomTab'")
            component.title = "CustomTab"
        try:
            component.header.mode = header_mode
        except AttributeError:
            print(f"Warning couldn't find a header to set mode to {header_mode}")
        return component
    else:
        raise ValueError(f"{component} is not a valid component...")


class ExplainerTabsLayout:
    def __init__(self, explainer, tabs,
                 title='Model Explainer',
                 hide_title=False,
                 hide_selector=False,
                 fluid=True,
                 **kwargs):
        """Generates a multi tab layout from a a list of ExplainerComponents.
        If the component is a class definition, it gets instantiated first. If 
        the component is not derived from an ExplainerComponent, then attempt
        with duck typing to nevertheless instantiate a layout.

        Args:
            explainer ([type]): explainer
            tabs (list[ExplainerComponent class or instance]): list of
                ExplainerComponent class definitions or instances.
            title (str, optional): [description]. Defaults to 'Model Explainer'.
            hide_title (bool, optional): Hide the title. Defaults to False.
            hide_selector (bool, optional): Hide the positive label selector. 
                        Defaults to False.
            fluid (bool, optional): Stretch layout to fill space. Defaults to False.
        """
        self.fluid = fluid

        self.header = ExplainerHeader(explainer, "dashboard", title,
                                    hide_title=hide_title, 
                                    hide_selector=hide_selector,
                                    fluid=fluid)
        self.tabs  = [instantiate_component(tab, explainer, "none", **kwargs) for tab in tabs]
        assert len(self.tabs) > 0, 'When passing a list to tabs, need to pass at least one valid tab!'
        
    def layout(self):
        """returns a multitab layout plus ExplainerHeader"""
        return dbc.Container([
            self.header.layout(),
            dcc.Tabs(id="tabs", value=self.tabs[0].name, 
                        children=[dcc.Tab(label=tab.title, id=tab.name, value=tab.name,
                                        children=tab.layout()) for tab in self.tabs]),
        ], fluid=self.fluid)

    def register_callbacks(self, app):
        """Registers callbacks for all tabs"""
        for tab in self.tabs:
            try:
                tab.register_callbacks(app)
            except AttributeError:
                print(f"Warning: {tab} does not have a register_callbacks method!")

    def calculate_dependencies(self):
        """Calculates dependencies for all tabs"""
        for tab in self.tabs:
            try:
                tab.calculate_dependencies()
            except AttributeError:
                print(f"Warning: {tab} does not have a calculate_dependencies method!")


class ExplainerPageLayout(ExplainerComponent):
    def __init__(self, explainer, component,
                 title='Model Explainer',
                 hide_title=False,
                 hide_selector=False,
                 fluid=False,
                 **kwargs):
        """Generates a single page layout from a single ExplainerComponent.
        If the component is a class definition, it gets instantiated. 

        If the component is not derived from an ExplainerComponent, then tries
        with duck typing to nevertheless instantiate a layout.


        Args:
            explainer ([type]): explainer
            component (ExplainerComponent class or instance): ExplainerComponent 
                        class definition or instance.
            title (str, optional): [description]. Defaults to 'Model Explainer'.
            hide_title (bool, optional): Hide the title. Defaults to False.
            hide_selector (bool, optional): Hide the positive label selector. 
                        Defaults to False.
            fluid (bool, optional): Stretch layout to fill space. Defaults to False.
        """
        
        self.fluid = fluid
        self.header = ExplainerHeader(explainer, "standalone", title,
                                    hide_title=hide_title, 
                                    hide_selector=hide_selector,
                                    fluid=fluid)
        self.page  = instantiate_component(component, explainer, "none", **kwargs)
        
    def layout(self):
        """returns single page layout with an ExplainerHeader"""
        return dbc.Container([
            self.header.layout(),
            self.page.layout()
        ], fluid=self.fluid)

    def register_callbacks(self, app):
        """Register callbacks of page"""
        try:
            self.page.register_callbacks(app)
        except AttributeError:
            print(f"Warning: {self.page} does not have a register_callbacks method!")

    def calculate_dependencies(self):
        """Calculate dependencies of page"""
        try:
            self.page.calculate_dependencies()
        except AttributeError:
            print(f"Warning: {self.page} does not have a calculate_dependencies method!")

