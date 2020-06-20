# -*- coding: utf-8 -*-

__all__ = ['ExplainerDashboard', 
            'JupyterExplainerDashboard',
            'ExplainerTab',
            'JupyterExplainerTab',
            'InlineExplainer',
            'ModelSummaryTab', 
            'ContributionsTab', 
            'ShapDependenceTab',
            'ShapInteractionsTab', 
            'DecisionTreesTab']


import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc

from jupyter_dash import JupyterDash

import plotly.io as pio

from .dashboard_tabs.dashboard_methods import *
from .dashboard_tabs.model_summary_tab import *
from .dashboard_tabs.contributions_tab import *
from .dashboard_tabs.shap_dependence_tab import *
from .dashboard_tabs.shap_interactions_tab import *
from .dashboard_tabs.decision_trees_tab import *
from explainerdashboard.dashboard_tabs.model_summary_tab import ModelSummaryTab


class ExplainerDashboard:
    """Constructs a dashboard out of an Explainer object. You can indicate
    which tabs to include, and pass kwargs on to individual tabs.
    """
    def __init__(self, explainer, title='Model Explainer',   
                tabs=None,
                model_summary=True,  
                contributions=True,
                shap_dependence=True,
                shap_interaction=True,
                decision_trees=True,
                plotly_template="none",
                **kwargs):
        """Constructs an ExplainerDashboard.
        
        :param explainer: an ExplainerBunch object
        :param title: Title of the dashboard, defaults to 'Model Explainer'
        :type title: str, optional
        :param model_summary: display model_summary tab or not, defaults to True
        :type model_summary: bool, optional
        :param contributions: display individual contributions tab or not, defaults to True
        :type contributions: bool, optional
        :param shap_dependence: display shap dependence tab or not, defaults to True
        :type shap_dependence: bool, optional
        :param shap_interaction: display tab interaction tab or not. 
        :type shap_interaction: bool, optional
        :param decision_trees: display tab with individual decision tree of random forest, defaults to False
        :type decision_trees: bool, optional
        """
        self.explainer=explainer
        self.title = title
        self.model_summary = model_summary
        self.contributions = contributions
        self.shap_dependence = shap_dependence
        self.shap_interaction = shap_interaction
        self.decision_trees = decision_trees
        self.plotly_template = plotly_template
        self.kwargs = kwargs

        # calculate lazily loaded properties before starting dashboard:
        if shap_dependence or contributions or model_summary:
            _ = explainer.shap_values, explainer.preds, explainer.pred_percentiles
            if explainer.cats is not None:
                _ = explainer.shap_values_cats
            if explainer.is_classifier:
                _ = explainer.pred_probas
        if model_summary:
            _ = explainer.permutation_importances
            if explainer.cats is not None:
                _ = explainer.permutation_importances_cats
        if shap_interaction:
            try:
                _ = explainer.shap_interaction_values
                if explainer.cats is not None:
                    _ = explainer.shap_interaction_values_cats
            except:
                print("Note: calculating shap interaction failed, so turning off interactions tab")
                self.shap_interaction=False
        if decision_trees:
            if hasattr(self.explainer, 'decision_trees'):
                _ = explainer.graphviz_available
                _ = explainer.decision_trees
            else:
                self.decision_trees = False
            
        self.app = self.get_dash_app()
        self.app.title = title
        
        pio.templates.default = self.plotly_template

        # layout
        self.title_and_label_selector = TitleAndLabelSelector(explainer, title=title)
        self.tabs = [] if tabs is None else tabs
        self._insert_tabs()
        assert len(self.tabs) > 0, 'need to pass at least one tab! e.g. model_summary=True'
        
        self.tab_layouts = [
            dcc.Tab(children=tab.layout(), label=tab.title, id=tab.tab_id, value=tab.tab_id) 
                for tab in self.tabs]

        self.app.layout = dbc.Container([
            self.title_and_label_selector.layout(),
            dcc.Tabs(id="tabs", value=self.tabs[0].tab_id, children=self.tab_layouts),
        ], fluid=True)

        #register callbacks
        self.title_and_label_selector.register_callbacks(self.app)
        
        for tab in self.tabs:
            tab.register_callbacks(self.app)

    def get_dash_app(self):
        app = dash.Dash(__name__)
        app.config['suppress_callback_exceptions']=True
        app.css.config.serve_locally = True
        app.scripts.config.serve_locally = True
        return app

    def _insert_tabs(self):
        if self.model_summary:
            self.tabs.append(ModelSummaryTab(self.explainer, **self.kwargs))
        if self.contributions:
            self.tabs.append(ContributionsTab(self.explainer, **self.kwargs))
        if self.shap_dependence:
            self.tabs.append(ShapDependenceTab(self.explainer, **self.kwargs))
        if self.shap_interaction:
            self.tabs.append(ShapInteractionsTab(self.explainer, **self.kwargs))
        if self.decision_trees:
            assert hasattr(self.explainer, 'decision_trees'), \
                """the explainer object has no shadow_trees property. This tab 
                only works with a RandomForestClassifierBunch or RandomForestRegressionBunch""" 
            self.tabs.append(DecisionTreesTab(self.explainer, **self.kwargs))
        
    def run(self, port=8050, **kwargs):
        """Starts the dashboard using the built-in Flask server on localhost:port
        
        :param port: the port to run the dashboard on, defaults to 8050
        :type port: int, optional
        """
        print(f"Running {self.title} on http://localhost:{port}")
        pio.templates.default = self.plotly_template
        self.app.run_server(port=port, **kwargs)


class JupyterExplainerDashboard(ExplainerDashboard):
    """
    ExplainerDashboard that uses launches a JupyterDash app instead of a
    default dash app.
    """ 
    def get_dash_app(self):
        app = JupyterDash(__name__)
        return app

    def run(self, port=8050, mode='inline', width=800, height=650, **kwargs):
        """Starts the dashboard using the built-in Flask server on localhost:port

        :param port: the port to run the dashboard on, defaults to 8050
        :type port: int, optional
        :param mode: either 'inline', 'jupyterlab' or 'external'
        :type mode: str, optional
        :param width: width in pixels of inline iframe
        :param height: height in pixels of inline iframe
        """
        pio.templates.default = self.plotly_template
        if mode in ['inline', 'jupyterlab']:
            self.app.run_server(mode=mode, width=width, height=height, port=port)
        elif mode == 'external':
             self.app.run_server(mode=mode, port=port, **kwargs)
        else:
            raise ValueError("mode should either be 'inline', 'jupyterlab'  or 'external'!")


class ExplainerTab:
    """Constructs a dashboard with a single tab out of an Explainer object. 
    You either pass the class definition of the tab to include, or a string
    identifier. 
    """
    def __init__(self, explainer, tab, title='Model Explainer', 
                    plotly_template="none", **kwargs):
        """Constructs an ExplainerDashboard.
        
        :param explainer: an ExplainerBunch object
        :param tab: Tab or string identifier for tab to be displayed: 
                    'model_summary', 'contributions', 'shap_dependence', 
                    'shap_interaction', 'decision_trees'
        :type tab: either an ExplainerTab or str
        :param title: Title of the dashboard, defaults to 'Model Explainer'
        :type title: str, optional
        :param tab: single tab to be run as dashboard
        """
        
        self.explainer = explainer
        self.title = title

        self.plotly_template = plotly_template
        self.kwargs = kwargs

        if isinstance(tab, str):
            if tab == 'model_summary':
                tab = ModelSummaryTab
            elif tab == 'contributions':
                tab = ContributionsTab
            elif tab == 'shap_dependence':
                tab = ShapDependenceTab
            elif tab == 'shap_interaction':
                tab = ShapInteractionsTab
            elif tab == 'decision_trees':
                tab = DecisionTreesTab
            else:
                raise ValueError("If tab is str then it should be in "
                        "['model_summary', 'contributions', 'shap_dependence', "
                        "'shap_interactions', 'decision_trees']")

        self.tab = tab(self.explainer, standalone=True, **self.kwargs)

        self.app = self.get_dash_app()
        self.app.title = title
        
        pio.templates.default = self.plotly_template

        self.app.layout = self.tab.layout()
        self.tab.register_callbacks(self.app)

    def get_dash_app(self):
        app = dash.Dash(__name__)
        app.config['suppress_callback_exceptions']=True
        app.css.config.serve_locally = True
        app.scripts.config.serve_locally = True
        return app

    def run(self, port=8050, **kwargs):
        """Starts the dashboard using the built-in Flask server on localhost:port
        
        :param port: the port to run the dashboard on, defaults to 8050
        :type port: int, optional
        """
        print(f"Running {self.title} on http://localhost:{port}")
        pio.templates.default = self.plotly_template
        self.app.run_server(port=port, **kwargs)


class JupyterExplainerTab(ExplainerTab):
    def get_dash_app(self):
        app = JupyterDash(__name__)
        return app

    def run(self, port=8050, mode='inline', width=800, height=650, **kwargs):
        """Starts the dashboard using the built-in Flask server on localhost:port
        :param port: the port to run the dashboard on, defaults to 8050
        :type port: int, optional
        :param mode: either 'inline', 'jupyterlab' or 'external' 
        :type mode: str, optional
        :param width: width in pixels of inline iframe
        :param height: height in pixels of inline iframe
        """
        pio.templates.default = self.plotly_template
        if mode in ['inline', 'jupyterlab']:
            self.app.run_server(mode=mode, width=width, height=height, port=port)
        elif mode == 'jupyterlab':
             self.app.run_server(mode=mode, port=port, **kwargs)
        else:
            raise ValueError("mode should either be 'inline', 'jupyterlab' or 'external'!")

class InlineExplainer:
    """
    Run a single tab inline in a Jupyter notebook using specific method calls.
    """
    def __init__(self, explainer, mode='inline', width=800, height=650, 
                    port=8050, plotly_template="none", **kwargs):
        """
        :param explainer: an Explainer object
        :param mode: either 'inline', 'jupyterlab' or 'external' 
        :type mode: str, optional
        :param width: width in pixels of inline iframe
        :param height: height in pixels of inline iframe
        :param port: port to run if mode='external'
        """
        assert mode in ['inline', 'external', 'jupyterlab'], \
            "mode should either be 'inline', 'external' or 'jupyterlab'!"
        self.explainer = explainer
        self.mode = mode
        self.width = width
        self.height = height
        self.port = port
        self.plotly_template = plotly_template
        self.kwargs = kwargs

    def _run_app(self, app, **kwargs):
        """Starts the dashboard either inline or in a seperate tab

        :param app: the JupyterDash app to be run
        :type mode: JupyterDash app instance
        """
        pio.templates.default = self.plotly_template
        if self.mode in ['inline', 'jupyterlab']:
            app.run_server(mode=self.mode, width=self.width, height=self.height, port=self.port)
        elif self.mode == 'external':
             app.run_server(mode=self.mode, port=self.port, **self.kwargs)
        else:
            raise ValueError("mode should either be 'inline', 'jupyterlab'  or 'external'!")

    def _run_tab(self, tab, title):
        app = JupyterDash(__name__)
        app.title = title
        app.layout = tab.layout()
        tab.register_callbacks(app)
        self._run_app(app)

    def model_summary(self, title='Model Summary', 
                        bin_size=0.1, quantiles=10, cutoff=0.5, 
                        round=2, logs=False, vs_actual=False, ratio=False):
        """Runs model_summary tab inline in notebook"""
        tab = ModelSummaryTab(self.explainer, standalone=True, hide_title=True,
                    bin_size=bin_size, quantiles=quantiles, cutoff=cutoff, 
                    round=round, logs=logs, vs_actual=vs_actual, ratio=ratio)
        self._run_tab(tab, title)

    def importances(self, title='Importances', **kwargs):
        """Runs model_summary tab inline in notebook"""
        tab = ImportancesStats(self.explainer, 
                standalone=True, hide_title=True, **kwargs)
        self._run_tab(tab, title)

    def model_stats(self, title='Models Stats', **kwargs):
        """Runs model_stats inline in notebook"""
        if self.explainer.is_classifier:
            tab = ClassifierModelStats(self.explainer, 
                standalone=True, hide_title=True, **kwargs)
        elif self.explainer.is_regression:
            tab = RegressionModelStats(self.explainer, 
                standalone=True, hide_title=True, **kwargs)
        self._run_tab(tab, title)

    def contributions(self,  title='Contributions', **kwargs):
        """Show contributions (permutation or shap) inline in notebook"""
        tab = ContributionsTab(self.explainer, 
                standalone=True, hide_title=True, **kwargs)
        self._run_tab(tab, title)

    def shap_dependence(self, title='Shap Dependence', **kwargs):
        """Runs shap_dependence tab inline in notebook"""
        tab = ShapDependenceTab(self.explainer, 
                standalone=True, hide_title=True, **kwargs)
        self._run_tab(tab, title)

    def shap_interaction(self, title='Shap Interactions', **kwargs):
        """Runs shap_interaction tab inline in notebook"""
        tab = ShapInteractionsTab(self.explainer, 
            standalone=True, hide_title=True, **kwargs)
        self._run_tab(tab, title)

    def decision_trees(self, title='Decision Trees', **kwargs):
        """Runs decision_trees tab inline in notebook"""
        tab = DecisionTreesTab(self.explainer, 
                standalone=True, hide_title=True, **kwargs)
        self._run_tab(tab, title)

