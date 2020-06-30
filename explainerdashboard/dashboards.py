# -*- coding: utf-8 -*-

__all__ = ['ExplainerDashboard', 
            'JupyterExplainerDashboard',
            'ExplainerTab',
            'JupyterExplainerTab',
            'InlineExplainer']


import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc

from jupyter_dash import JupyterDash

import plotly.io as pio

from .dashboard_components import *
from .dashboard_tabs import *


class ExplainerDashboard:
    """Constructs a dashboard out of an Explainer object. You can indicate
    which tabs to include, and pass kwargs on to individual tabs.
    """
    def __init__(self, explainer, tabs=None,
                 title='Model Explainer',   
                importances=True,
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
        
        if tabs is None:
            tabs = []
            if shap_interaction and not explainer.interactions_should_work:
                print("For this type of model and model_output interactions don't "
                          "work, so setting shap_interaction=False...")
                shap_interactions = False
            if decision_trees and not hasattr(explainer, 'decision_trees'):
                print("the explainer object has no decision_trees property. so "
                        "setting decision_trees=False...")
                decision_trees = False
        
            if importances:
                tabs.append(ImportancesTab)
            if model_summary:
                tabs.append(ModelSummaryTab)
            if contributions:
                tabs.append(ContributionsTab)
            if shap_dependence:
                tabs.append(ShapDependenceTab)
            if shap_interaction:
                tabs.append(ShapInteractionsTab)
            if decision_trees:
                tabs.append(DecisionTreesTab)

        tabs  = [tab(explainer, header_mode="none") for tab in tabs if issubclass(tab, ExplainerComponent)]
        assert len(tabs) > 0, 'need to pass at least one valid tab! e.g. model_summary=True'

        self.app = self.get_dash_app()
        self.app.title = title
        self.app.layout = dbc.Container([
            ExplainerHeader(explainer, title=title, mode="dashboard").layout(),
            dcc.Tabs(id="tabs", value=tabs[0].name, 
                     children=[dcc.Tab(label=tab.title, id=tab.name, value=tab.name,
                                       children=tab.layout()) for tab in tabs]),
        ], fluid=True)
        
        for tab in tabs:
            tab.calculate_dependencies()
            tab.register_callbacks(self.app)

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
        print(f"Running {self.app.title} on http://localhost:{port}")
        pio.templates.default = "none"
        self.app.run_server(port=port, **kwargs)


class JupyterExplainerDashboard(ExplainerDashboard):
    """
    ExplainerDashboard that uses launches a JupyterDash app instead of a
    default dash app.
    """ 
    def get_dash_app(self):
        app = JupyterDash(__name__)
        return app

    def run(self, mode='inline', port=8050, width=800, height=650, **kwargs):
        """Starts the dashboard using the built-in Flask server on localhost:port

        :param mode: either 'inline', 'jupyterlab' or 'external'
        :type mode: str, optional
        :param port: the port to run the dashboard on, defaults to 8050
        :type port: int, optional
        :param width: width in pixels of inline iframe
        :param height: height in pixels of inline iframe
        """
        pio.templates.default = "none"
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
                    'importances', 'model_summary', 'contributions', 'shap_dependence', 
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
            if tab == 'importances':
                tab = ImportancesTab
            elif tab == 'model_summary':
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
                raise ValueError("If parameter tab is str then it should be in "
                        "['importances', 'model_summary', 'contributions', 'shap_dependence', "
                        "'shap_interactions', 'decision_trees']")

        self.tab = tab(self.explainer, header_mode="standalone", **self.kwargs)

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

    def run(self, port=8050, mode='inline', width=1000, height=650, **kwargs):
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
    def __init__(self, explainer, mode='inline', width=1000, height=800, 
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

    def _run_component(self, component, title):
        app = JupyterDash(__name__)
        app.title = title
        app.layout = component.layout()
        component.register_callbacks(app)
        self._run_app(app)

    def modelsummary_tab(self, title='Model Summary', 
                        bin_size=0.1, quantiles=10, cutoff=0.5, 
                        logs=False, pred_or_actual="vs_pred", ratio=False, col=None,
                        importance_type="shap", depth=None, cats=True):
        """Runs model_summary tab inline in notebook"""
        tab = ModelSummaryTab(self.explainer, header_mode="hidden",
                    bin_size=bin_size, quantiles=quantiles, cutoff=cutoff, 
                    logs=logs, pred_or_actual=pred_or_actual, ratio=ratio,
                    importance_type=importance_type, depth=depth, cats=cats)
        self._run_component(tab, title)

    def importances_tab(self,  title='Importances', **kwargs):
        """Show contributions (permutation or shap) inline in notebook"""
        tab = ImportancesTab(self.explainer, 
                header_mode="standalone", **kwargs)
        self._run_component(tab, title)

    def contributions_tab(self,  title='Contributions', **kwargs):
        """Show contributions (permutation or shap) inline in notebook"""
        tab = ContributionsTab(self.explainer, 
                header_mode="standalone", **kwargs)
        self._run_component(tab, title)

    def dependence_tab(self, title='Shap Dependence', **kwargs):
        """Runs shap_dependence tab inline in notebook"""
        tab = ShapDependenceTab(self.explainer, 
                header_mode="standalone", **kwargs)
        self._run_component(tab, title)

    def interactions_tab(self, title='Shap Interactions', **kwargs):
        """Runs shap_interactions tab inline in notebook"""
        tab = ShapInteractionsTab(self.explainer, 
                header_mode="standalone", **kwargs)
        self._run_component(tab, title)

    def decisiontrees_tab(self, title='Decision Trees', **kwargs):
        """Runs decision_trees tab inline in notebook"""
        tab = DecisionTreesTab(self.explainer, 
                header_mode="standalone", **kwargs)
        self._run_component(tab, title)

    def importances(self, title='Importances', **kwargs):
        """Runs model_summary tab inline in notebook"""
        comp = ImportancesComponent(self.explainer, header_mode="hidden", **kwargs)
        self._run_component(comp, title)

    def model_stats(self, title='Models Stats', **kwargs):
        """Runs model_stats inline in notebook"""
        if self.explainer.is_classifier:
            comp = ClassifierModelStatsComposite(self.explainer, 
                    header_mode="hidden", **kwargs)
        elif self.explainer.is_regression:
            comp = RegressionModelStatsComposite(self.explainer, 
                header_mode="hidden", **kwargs)
        self._run_component(comp, title)

    def precision(self, title="Precision Plot", **kwargs):
        """shows precision plot"""
        assert self.explainer.is_classifier
        comp = PrecisionComponent(self.explainer, header_mode="hidden", **kwargs)
        self._run_component(comp, title)

    def confusion_matrix(self, title="Confusion Matrix", **kwargs):
        """shows precision plot"""
        comp= ConfusionMatrixComponent(self.explainer, header_mode="hidden", **kwargs)
        self._run_component(comp, title)

    def lift_curve(self, title="Lift Curve", **kwargs):
        """shows precision plot"""
        assert self.explainer.is_classifier
        comp = LiftCurveComponent(self.explainer, header_mode="hidden", **kwargs)
        self._run_component(comp, title)

    def classification(self, title="Classification", **kwargs):
        """shows precision plot"""
        assert self.explainer.is_classifier
        comp = ClassificationComponent(self.explainer, header_mode="hidden", **kwargs)
        self._run_component(comp, title)

    def roc_auc(self, title="ROC AUC Curve", **kwargs):
        """shows precision plot"""
        assert self.explainer.is_classifier
        comp = RocAucComponent(self.explainer, header_mode="hidden", **kwargs)
        self._run_component(comp, title)

    def pr_auc(self, title="PR AUC Curve", **kwargs):
        """shows precision plot"""
        assert self.explainer.is_classifier
        comp = PrAucComponent(self.explainer, header_mode="hidden", **kwargs)
        self._run_component(comp, title)

    def pred_vs_actual(self, title="Predicted vs Actual", **kwargs):
        "shows predicted vs actual for regression"
        assert self.explainer.is_regression
        comp = PredictedVsActualComponent(self.explainer, header_mode="hidden", **kwargs)
        self._run_component(comp, title)

    def residuals(self, title="Residuals", **kwargs):
        "shows residuals for regression"
        assert self.explainer.is_regression
        comp = ResidualsComponent(self.explainer, header_mode="hidden", **kwargs)
        self._run_component(comp, title)

    def residuals_vs_col(self, title="Residuals vs col", **kwargs):
        "shows residuals vs col for regression"
        assert self.explainer.is_regression
        comp = ResidualsVsColComponent(self.explainer, header_mode="hidden", **kwargs)
        self._run_component(comp, title)

    def prediction(self,  title='Prediction', **kwargs):
        """Show contributions (permutation or shap) inline in notebook"""
        comp = PredictionSummaryComponent(self.explainer, 
                header_mode="hidden", **kwargs)
        self._run_component(comp, title)

    def random_index(self, title='Random Index', **kwargs):
        """show random index selector inline in notebook"""
        if self.explainer.is_classifier:
            comp = ClassifierRandomIndexComponent(self.explainer, 
                    header_mode="hidden", **kwargs)
        elif self.explainer.is_regression:
            comp = RegressionRandomIndexComponent(self.explainer, 
                    header_mode="hidden", **kwargs)
        self._run_component(comp, title)

    def contributions_graph(self,  title='Contributions', **kwargs):
        """Show contributions (permutation or shap) inline in notebook"""
        comp = ShapContributionsGraphComponent(self.explainer, 
                header_mode="hidden", **kwargs)
        self._run_component(comp, title)

    def contributions_table(self,  title='Contributions', **kwargs):
        """Show contributions (permutation or shap) inline in notebook"""
        comp = ShapContributionsTableComponent(self.explainer, 
                header_mode="hidden", **kwargs)
        self._run_component(comp, title)

    def pdp(self, title="Partial Dependence Plots", **kwargs):
        """Show contributions (permutation or shap) inline in notebook"""
        comp = PdpComponent(self.explainer, 
                header_mode="hidden", **kwargs)
        self._run_component(comp, title)

    def individual_predictions(self,  title='Contributions', **kwargs):
        """Show contributions (permutation or shap) inline in notebook"""
        comp = IndividualPredictionsComposite(self.explainer, 
                header_mode="hidden", **kwargs)
        self._run_component(comp, title)

    def shap_overview(self, title='Shap Overview', **kwargs):
        """Runs shap_dependence tab inline in notebook"""
        comp = ShapDependenceComposite(self.explainer, 
                header_mode="hidden", **kwargs)
        self._run_component(comp, title)

    def shap_summary(self, title='Shap Summary', **kwargs):
        """Show shap summary inline in notebook"""
        comp = ShapSummaryComponent(self.explainer, 
                header_mode="hidden", **kwargs)
        self._run_component(comp, title)

    def shap_dependence(self, title='Shap Dependence', **kwargs):
        """Show shap summary inline in notebook"""
        
        comp = ShapDependenceComponent(self.explainer, 
                header_mode="hidden", **kwargs)
        self._run_component(comp, title)

    def interaction_overview(self, title='Interactions Overview', **kwargs):
        """Runs shap_dependence tab inline in notebook"""
        comp = ShapInteractionsComposite(self.explainer, 
                header_mode="hidden", **kwargs)
        self._run_component(comp, title)

    def interaction_summary(self, title='Shap Interaction Summary', **kwargs):
        """show shap interaction summary inline in notebook"""
        comp =InteractionSummaryComponent(self.explainer, header_mode="hidden", **kwargs)
        self._run_component(comp, title)

    def interaction_dependence(self, title='Shap Interaction Dependence', **kwargs):
        """show shap interaction dependence inline in notebook"""
        comp =InteractionDependenceComponent(self.explainer, header_mode="hidden", **kwargs)
        self._run_component(comp, title)

    def decisiontrees_overview(self, title="Decision Trees", **kwargs):
        """shap decision tree composite inline in notebook"""
        comp = DecisionTreesComposite(self.explainer, header_mode="hidden", **kwargs)
        self._run_component(comp, title)

    def decisiontrees(self, title='Decision Trees', **kwargs):
        """Runs decision_trees tab inline in notebook"""
        comp = DecisionTreesComponent(self.explainer, 
                header_mode="hidden", **kwargs)
        self._run_component(comp, title)

    def decisionpath_table(self, title='Decision path', **kwargs):
        """Runs decision_trees tab inline in notebook"""
        comp = DecisionPathTableComponent(self.explainer, 
                header_mode="hidden", **kwargs)
        self._run_component(comp, title)

    def decisionpath_graph(self, title='Decision path', **kwargs):
        """Runs decision_trees tab inline in notebook"""
        comp = DecisionPathTableComponent(self.explainer, 
                header_mode="hidden", **kwargs)
        self._run_component(comp, title)

