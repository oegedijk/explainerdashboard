__all__ = [
    'delegates_kwargs',
    'delegates_doc',
    'DummyComponent',
    'ExplainerComponent',
    'PosLabelSelector',
    'make_hideable',
    'get_dbc_tooltips',
    'update_params',
    'update_kwargs',
]

import sys
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


def update_params(kwargs, **params):
    """kwargs override params"""
    return dict(params, **kwargs)


def update_kwargs(kwargs, **params):
    """params override kwargs"""
    return dict(kwargs, **params)


def get_dbc_tooltips(dbc_table, desc_dict, hover_id, name):
                """Return a dbc.Table and a list of dbc.Tooltips.

                Args:
                    dbc_table (dbc.Table): Table with first column consisting of label
                    desc_dict (dict): dict that map labels to a description (str)
                    hover_id (str): dash component_id base: tooltips will have 
                        component_id=f"{hover_id}-{label}-{name}"
                    name (str): name to be used in hover_id

                Returns:
                    dbc.Table, List[dbc.Tooltip]
                """
                tooltips_dict = {}
                for tr in dbc_table.children[1].children:
                    tds = tr.children
                    label = tds[0].children
                    if label in desc_dict:
                        tr.id = f'{hover_id}-{label}-'+name
                        tooltips_dict[label] = desc_dict[label]

                tooltips = [dbc.Tooltip(desc,
                    target=f'{hover_id}-{label}-'+name, 
                    placement="top") for label, desc in tooltips_dict.items()]

                return dbc_table, tooltips


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
        return None

    def register_callbacks(self, app):
        pass


class ExplainerComponent(ABC):
    """ExplainerComponent is a bundle of a dash layout and callbacks that
    make use of an Explainer object. 

    An ExplainerComponent can have ExplainerComponent subcomponents, that
    you register with register_components(). If the component depends on 
    certain lazily calculated Explainer properties, you can register these
    with register_dependencies().

    ExplainerComponent makes sure that:

    1. Callbacks of subcomponents are registered.
    2. Lazily calculated dependencies (even of subcomponents) can be calculated.
    3. Pos labels selector id's of all subcomponents can be calculated. 
    
    Each ExplainerComponent adds a unique uuid name string to all elements, so 
    that there is never a name clash even with multiple ExplanerComponents of 
    the same type in a layout. 

    Important:
        define your callbacks in component_callbacks() and
        ExplainerComponent will register callbacks of subcomponents in addition
        to component_callbacks() when calling register_callbacks()
    """
    def __init__(self, explainer, title=None, name=None):
        """initialize the ExplainerComponent

        Args:
            explainer (Explainer): explainer object constructed with e.g.
                        ClassifierExplainer() or RegressionExplainer()
            title (str, optional): Title of tab or page. Defaults to None.
            name (str, optional): unique name to add to Component elements. 
                        If None then random uuid is generated to make sure 
                        it's unique. Defaults to None.
        """
        self.explainer = explainer
        self.title = title
        self.name = name
        if self.name is None:
            self.name = shortuuid.ShortUUID().random(length=5)

        self._store_child_params(
                        no_attr=['explainer', 'title', 'name'], 
                        no_param=['explainer'])
        self._components = []
        self._dependencies = []

    def _store_child_params(self, no_store=None, no_attr=None, no_param=None):
             
        if not hasattr(self, '_stored_params'): 
            self._stored_params = {}
        child_frame = sys._getframe(2)
        child_args = child_frame.f_code.co_varnames[1:child_frame.f_code.co_argcount]
        child_dict = {arg: child_frame.f_locals[arg] for arg in child_args}
        
        if isinstance(no_store, bool) and no_store:
            return
        else:
            if no_store is None: no_store = tuple()
        
        if isinstance(no_attr, bool) and no_attr: dont_attr = True
        else:
            if no_attr is None: no_attr = tuple()
            dont_attr = False 
            
        if isinstance(no_param, bool) and no_param: dont_param = True
        else:
            if no_param is None: no_param = tuple()
            dont_param = False 

        for name, value in child_dict.items():
            if not dont_attr and name not in no_store and name not in no_attr:
                setattr(self, name, value)
            if not dont_param and name not in no_store and name not in no_param:
                self._stored_params[name] = value

    def register_components(self, *components):
        """register subcomponents so that their callbacks will be registered
        and dependencies can be tracked
        
        Args:
            scan_self (bool, optional): scan self.__dict__ and add all
            ExplainerComponent attributes to _components. Defaults to True
        """
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
        
        for k, v in self.__dict__.items():
            if k != '_components' and isinstance(v, ExplainerComponent) and v not in self._components:
                self._components.append(v)

    def has_pos_label_connector(self):
        if not hasattr(self, '_components'):
            self._components = []
        for comp in self._components:
            if str(type(comp)).endswith("PosLabelConnector'>"):
                return True
            elif comp.has_pos_label_connector():
                return True
        return False
            
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

    @property
    def component_imports(self):
        """returns a list of ComponentImport namedtuples("component", "module")
         all components and and subcomponents"""
        if not hasattr(self, '_components'):
            self._components = []
        _component_imports = [(self.__class__.__name__, self.__class__.__module__)]
        for comp in self._components:
            _component_imports.extend(comp.component_imports)
        return list(set(_component_imports))


    @property
    def pos_labels(self):
        """returns a list of unique pos label selector elements 
        of the component and all subcomponents"""
        
        if not hasattr(self, '_components'):
            self._components = []
        pos_labels = []
        if hasattr(self, 'selector') and isinstance(self.selector, PosLabelSelector):
            pos_labels.append('pos-label-'+self.selector.name)
        for comp in self._components:
            pos_labels.extend(comp.pos_labels)
        pos_labels = list(set(pos_labels))
        return pos_labels

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

    def layout(self):
        """layout to be defined by the particular ExplainerComponent instance.
        All element id's should append +self.name to make sure they are unique."""
        return None

    def component_callbacks(self, app):
        """register callbacks specific to this ExplainerComponent."""
        if hasattr(self, "_register_callbacks"):
            print("Warning: the use of _register_callbacks() will be deprecated!"
                  " Use component_callbacks() from now on...")
            self._register_callbacks(app)

    def register_callbacks(self, app):
        """First register callbacks of all subcomponents, then call
        component_callbacks(app)
        """
        if not hasattr(self, '_components'):
            self._components = []
        for comp in self._components:
            comp.register_callbacks(app)
        self.component_callbacks(app)


class PosLabelSelector(ExplainerComponent):
    """For classifier models displays a drop down menu with labels to be selected
    as the positive class.
    """
    def __init__(self, explainer, title='Pos Label Selector', name=None,
                     pos_label=None):
        """Generates a positive label selector with element id 'pos_label-'+self.name

        Args:
            explainer (Explainer): explainer object constructed with e.g.
                        ClassifierExplainer() or RegressionExplainer()
            title (str, optional): Title of tab or page. Defaults to None.
            name (str, optional): unique name to add to Component elements. 
                        If None then random uuid is generated to make sure 
                        it's unique. Defaults to 'Pos Label Selector'.
            pos_label (int, optional): Initial pos label. Defaults to 
                        explainer.pos_label.
        """
        super().__init__(explainer, title, name)
        if pos_label is not None:
            self.pos_label = explainer.get_pos_label_index(pos_label)
        else:
            self.pos_label = explainer.pos_label

    def layout(self):
        if self.explainer.is_classifier:
            return html.Div([
                    dbc.Label("Positive class:", 
                        html_for="pos-label-"+self.name, 
                        id="pos-label-label-"+self.name),
                    dbc.Tooltip("Select the label to be set as the positive class",
                        target="pos-label-label-"+self.name),
                    dcc.Dropdown(
                        id='pos-label-'+self.name,
                        options = [{'label': label, 'value': i}
                                for i, label in enumerate(self.explainer.labels)],
                        value = self.pos_label)
            ])
        else:
            return html.Div([dcc.Input(id="pos-label-"+self.name)], style=dict(display="none"))