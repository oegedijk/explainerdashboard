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

    def run(self, port=8050, mode='inline', width=800, height=650, **kwargs):
        """Starts the dashboard using the built-in Flask server on localhost:port

        :param port: the port to run the dashboard on, defaults to 8050
        :type port: int, optional
        :param mode: either 'inline', 'jupyterlab' or 'external'
        :type mode: str, optional
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
                    header_mode="standalone", **kwargs):
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

        self.tab = tab(self.explainer, header_mode=header_mode, **self.kwargs)

        self.app = self.get_dash_app()
        self.app.title = title
        
        #pio.templates.default = self.plotly_template

        self.app.layout = self.tab.layout()
        self.tab.register_callbacks(self.app)
        self.tab.calculate_dependencies()

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
        #pio.templates.default = self.plotly_template
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
        #pio.templates.default = self.plotly_template
        if mode in ['inline', 'jupyterlab']:
            self.app.run_server(mode=mode, width=width, height=height, port=port)
        elif mode in ['external']:
             self.app.run_server(mode=mode, port=port, **kwargs)
        else:
            raise ValueError("mode should either be 'inline', 'jupyterlab' or 'external'!")


class InlineExplainer:
    """
    Run a single tab inline in a Jupyter notebook using specific method calls.
    """
    def __init__(self, explainer, mode='inline', width=1000, height=800, 
                    port=8050, **kwargs):
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
        self._explainer = explainer
        self._mode = mode
        self._width = width
        self._height = height
        self._port = port
        self._kwargs = kwargs
        self.tab = InlineExplainerTabs(self, "tabs") 
        """subclass with InlineExplainerTabs layouts, e.g. InlineExplainer(explainer).tab.modelsummary()"""
        self.shap = InlineShapExplainer(self, "shap") 
        """subclass with InlineShapExplainer layouts, e.g. InlineExplainer(explainer).shap.dependence()"""
        self.classifier = InlineClassifierExplainer(self, "classifier") 
        """subclass with InlineClassifierExplainer plots, e.g. InlineExplainer(explainer).classifier.confusion_matrix()"""
        self.regression = InlineRegressionExplainer(self, "regression") 
        """subclass with InlineRegressionExplainer plots, e.g. InlineExplainer(explainer).regression.residuals()"""
        self.decisiontrees =InlineDecisionTreesExplainer(self, "decisiontrees") 
        """subclass with InlineDecisionTreesExplainer plots, e.g. InlineExplainer(explainer).decisiontrees.decisiontrees()"""

    def _run_app(self, app, **kwargs):
        """Starts the dashboard either inline or in a seperate tab

        :param app: the JupyterDash app to be run
        :type mode: JupyterDash app instance
        """
        pio.templates.default = "none"
        if self._mode in ['inline', 'jupyterlab']:
            app.run_server(mode=self._mode, width=self._width, height=self._height, port=self._port)
        elif self._mode == 'external':
             app.run_server(mode=self._mode, port=self._port, **self._kwargs)
        else:
            raise ValueError("mode should either be 'inline', 'jupyterlab'  or 'external'!")

    def _run_component(self, component, title):
        app = JupyterDash(__name__)
        app.title = title
        app.layout = component.layout()
        component.register_callbacks(app)
        self._run_app(app)
    
    def importances(self, title='Importances', **kwargs):
        """Runs model_summary tab inline in notebook"""
        comp = ImportancesComponent(self._explainer, header_mode="hidden", **kwargs)
        self._run_component(comp, title)

    def model_stats(self, title='Models Stats', **kwargs):
        """Runs model_stats inline in notebook"""
        if self._explainer.is_classifier:
            comp = ClassifierModelStatsComposite(self._explainer, 
                    header_mode="hidden", **kwargs)
        elif self._explainer.is_regression:
            comp = RegressionModelStatsComposite(self._explainer, 
                header_mode="hidden", **kwargs)
        self._run_component(comp, title)

    def prediction(self,  title='Prediction', **kwargs):
        """Show contributions (permutation or shap) inline in notebook"""
        comp = PredictionSummaryComponent(self._explainer, 
                header_mode="hidden", **kwargs)
        self._run_component(comp, title)

    def random_index(self, title='Random Index', **kwargs):
        """show random index selector inline in notebook"""
        if self._explainer.is_classifier:
            comp = ClassifierRandomIndexComponent(self._explainer, 
                    header_mode="hidden", **kwargs)
        elif self._explainer.is_regression:
            comp = RegressionRandomIndexComponent(self._explainer, 
                    header_mode="hidden", **kwargs)
        self._run_component(comp, title)

    def pdp(self, title="Partial Dependence Plots", **kwargs):
        """Show contributions (permutation or shap) inline in notebook"""
        comp = PdpComponent(self._explainer, 
                header_mode="hidden", **kwargs)
        self._run_component(comp, title)

    
class InlineExplainerComponent:
    def __init__(self, inline_explainer, name):
        self._inline_explainer = inline_explainer
        self._explainer = inline_explainer._explainer
        self._name = name

    def _run_component(self, component, title):
        self._inline_explainer._run_component(component, title)

    def __repr__(self):
        component_methods = [method_name for method_name in dir(self)
                  if callable(getattr(self, method_name)) and not method_name.startswith("_")]

        return f"InlineExplainer.{self._name} has the following components: {', '.join(component_methods)}"


class InlineExplainerTabs(InlineExplainerComponent):
    
    @delegates_kwargs(ImportancesTab)
    @delegates_doc(ImportancesTab)
    def importances(self,  title='Importances', **kwargs):
        """Show contributions (permutation or shap) inline in notebook"""
        tab = ImportancesTab(self._explainer, 
                header_mode="standalone", **kwargs)
        self._run_component(tab, title)

    @delegates_kwargs(ModelSummaryTab)
    @delegates_doc(ModelSummaryTab)
    def modelsummary(self, title='Model Summary', 
                        bin_size=0.1, quantiles=10, cutoff=0.5, 
                        logs=False, pred_or_actual="vs_pred", ratio=False, col=None,
                        importance_type="shap", depth=None, cats=True):
        """Runs model_summary tab inline in notebook"""
        tab = ModelSummaryTab(self._explainer, header_mode="hidden",
                    bin_size=bin_size, quantiles=quantiles, cutoff=cutoff, 
                    logs=logs, pred_or_actual=pred_or_actual, ratio=ratio,
                    importance_type=importance_type, depth=depth, cats=cats)
        self._run_component(tab, title)

    @delegates_kwargs(ContributionsTab)
    @delegates_doc(ContributionsTab)
    def contributions(self,  title='Contributions', **kwargs):
        """Show contributions (permutation or shap) inline in notebook"""
        tab = ContributionsTab(self._explainer, 
                header_mode="standalone", **kwargs)
        self._run_component(tab, title)

    @delegates_kwargs(ShapDependenceTab)
    @delegates_doc(ShapDependenceTab)
    def dependence(self, title='Shap Dependence', **kwargs):
        """Runs shap_dependence tab inline in notebook"""
        tab = ShapDependenceTab(self._explainer, 
                header_mode="standalone", **kwargs)
        self._run_component(tab, title)

    @delegates_kwargs(ShapInteractionsTab)
    @delegates_doc(ShapInteractionsTab)
    def interactions(self, title='Shap Interactions', **kwargs):
        """Runs shap_interactions tab inline in notebook"""
        tab = ShapInteractionsTab(self._explainer, 
                header_mode="standalone", **kwargs)
        self._run_component(tab, title)

    @delegates_kwargs(ShapDecisionTreesTab)
    @delegates_doc(ShapDecisionTreesTab)
    def decisiontrees(self, title='Decision Trees', **kwargs):
        """Runs shap_interactions tab inline in notebook"""
        tab = ShapDecisionTreesTab(self._explainer, 
                header_mode="standalone", **kwargs)
        self._run_component(tab, title)


class InlineShapExplainer(InlineExplainerComponent):

    @delegates_kwargs(ShapDependenceComposite)
    @delegates_doc(ShapDependenceComposite)
    def overview(self, title='Shap Overview', **kwargs):
        """Runs shap_dependence tab inline in notebook"""
        comp = ShapDependenceComposite(self._explainer, 
                header_mode="hidden", **kwargs)
        self._run_component(comp, title)

    @delegates_kwargs(ShapSummaryComponent)
    @delegates_doc(ShapSummaryComponent)
    def summary(self, title='Shap Summary', **kwargs):
        """Show shap summary inline in notebook"""
        comp = ShapSummaryComponent(self._explainer, 
                header_mode="hidden", **kwargs)
        self._run_component(comp, title)

    @delegates_kwargs(ShapDependenceComponent)
    @delegates_doc(ShapDependenceComponent)
    def dependence(self, title='Shap Dependence', **kwargs):
        """Show shap summary inline in notebook"""
        
        comp = ShapDependenceComponent(self._explainer, 
                header_mode="hidden", **kwargs)
        self._run_component(comp, title)

    @delegates_kwargs(ShapInteractionsComposite)
    @delegates_doc(ShapInteractionsComposite)
    def interaction_overview(self, title='Interactions Overview', **kwargs):
        """Runs shap_dependence tab inline in notebook"""
        comp = ShapInteractionsComposite(self._explainer, 
                header_mode="hidden", **kwargs)
        self._run_component(comp, title)

    @delegates_kwargs(InteractionSummaryComponent)
    @delegates_doc(InteractionSummaryComponent)
    def interaction_summary(self, title='Shap Interaction Summary', **kwargs):
        """show shap interaction summary inline in notebook"""
        comp =InteractionSummaryComponent(self._explainer, header_mode="hidden", **kwargs)
        self._run_component(comp, title)

    @delegates_kwargs(InteractionDependenceComponent)
    @delegates_doc(InteractionDependenceComponent)
    def interaction_dependence(self, title='Shap Interaction Dependence', **kwargs):
        """show shap interaction dependence inline in notebook"""
        comp =InteractionDependenceComponent(self._explainer, header_mode="hidden", **kwargs)
        self._run_component(comp, title)

    @delegates_kwargs(ShapContributionsGraphComponent)
    @delegates_doc(ShapContributionsGraphComponent)
    def contributions_graph(self,  title='Contributions', **kwargs):
        """Show contributions (permutation or shap) inline in notebook"""
        comp = ShapContributionsGraphComponent(self._explainer, 
                header_mode="hidden", **kwargs)
        self._run_component(comp, title)

    @delegates_kwargs(ShapContributionsTableComponent)
    @delegates_doc(ShapContributionsTableComponent)
    def contributions_table(self,  title='Contributions', **kwargs):
        """Show contributions (permutation or shap) inline in notebook"""
        comp = ShapContributionsTableComponent(self._explainer, 
                header_mode="hidden", **kwargs)
        self._run_component(comp, title)


class InlineClassifierExplainer(InlineExplainerComponent):
    @delegates_kwargs(ClassifierModelStatsComposite)
    @delegates_doc(ClassifierModelStatsComposite)
    def model_stats(self, title='Models Stats', **kwargs):
        """Runs model_stats inline in notebook"""
        comp = ClassifierModelStatsComposite(self._explainer, 
                    header_mode="hidden", **kwargs)
        self._run_component(comp, title)

    @delegates_kwargs(PrecisionComponent)
    @delegates_doc(PrecisionComponent)
    def precision(self, title="Precision Plot", **kwargs):
        """shows precision plot"""
        assert self._explainer.is_classifier
        comp = PrecisionComponent(self._explainer, header_mode="hidden", **kwargs)
        self._run_component(comp, title)

    @delegates_kwargs(ConfusionMatrixComponent)
    @delegates_doc(ConfusionMatrixComponent)
    def confusion_matrix(self, title="Confusion Matrix", **kwargs):
        """shows precision plot"""
        comp= ConfusionMatrixComponent(self._explainer, header_mode="hidden", **kwargs)
        self._run_component(comp, title)

    @delegates_kwargs(LiftCurveComponent)
    @delegates_doc(LiftCurveComponent)
    def lift_curve(self, title="Lift Curve", **kwargs):
        """shows precision plot"""
        assert self._explainer.is_classifier
        comp = LiftCurveComponent(self._explainer, header_mode="hidden", **kwargs)
        self._run_component(comp, title)

    @delegates_kwargs(ClassificationComponent)
    @delegates_doc(ClassificationComponent)
    def classification(self, title="Classification", **kwargs):
        """shows precision plot"""
        assert self._explainer.is_classifier
        comp = ClassificationComponent(self._explainer, header_mode="hidden", **kwargs)
        self._run_component(comp, title)

    @delegates_kwargs(RocAucComponent)
    @delegates_doc(RocAucComponent)
    def roc_auc(self, title="ROC AUC Curve", **kwargs):
        """shows precision plot"""
        assert self._explainer.is_classifier
        comp = RocAucComponent(self._explainer, header_mode="hidden", **kwargs)
        self._run_component(comp, title)

    @delegates_kwargs(PrAucComponent)
    @delegates_doc(PrAucComponent)
    def pr_auc(self, title="PR AUC Curve", **kwargs):
        """shows precision plot"""
        assert self._explainer.is_classifier
        comp = PrAucComponent(self._explainer, header_mode="hidden", **kwargs)
        self._run_component(comp, title)


class InlineRegressionExplainer(InlineExplainerComponent):
    
    @delegates_kwargs(RegressionModelStatsComposite)
    @delegates_doc(RegressionModelStatsComposite)
    def model_stats(self, title='Models Stats', **kwargs):
        """Runs model_stats inline in notebook"""

        comp = RegressionModelStatsComposite(self._explainer, 
                    header_mode="hidden", **kwargs)
        self._run_component(comp, title)
    
    @delegates_kwargs(PredictedVsActualComponent)
    @delegates_doc(PredictedVsActualComponent)
    def pred_vs_actual(self, title="Predicted vs Actual", **kwargs):
        "shows predicted vs actual for regression"
        assert self.explainer.is_regression
        comp = PredictedVsActualComponent(self._explainer, header_mode="hidden", **kwargs)
        self._run_component(comp, title)

    @delegates_kwargs(ResidualsComponent)
    @delegates_doc(ResidualsComponent)
    def residuals(self, title="Residuals", **kwargs):
        "shows residuals for regression"
        assert self.explainer.is_regression
        comp = ResidualsComponent(self._explainer, header_mode="hidden", **kwargs)
        self._run_component(comp, title)

    @delegates_kwargs(ResidualsVsColComponent)
    @delegates_doc(ResidualsVsColComponent)
    def residuals_vs_col(self, title="Residuals vs col", **kwargs):
        "shows residuals vs col for regression"
        assert self.explainer.is_regression
        comp = ResidualsVsColComponent(self._explainer, header_mode="hidden", **kwargs)
        self._run_component(comp, title)


class InlineDecisionTreesExplainer(InlineExplainerComponent):
    @delegates_kwargs(DecisionTreesComposite)
    @delegates_doc(DecisionTreesComposite)
    def overview(self, title="Decision Trees", **kwargs):
        """shap decision tree composite inline in notebook"""
        comp = DecisionTreesComposite(self._explainer, header_mode="hidden", **kwargs)
        self._run_component(comp, title)

    @delegates_kwargs(DecisionTreesComponent)
    @delegates_doc(DecisionTreesComponent)
    def decisiontrees(self, title='Decision Trees', **kwargs):
        """Runs decision_trees tab inline in notebook"""
        comp = DecisionTreesComponent(self._explainer, 
                header_mode="hidden", **kwargs)
        self._run_component(comp, title)

    @delegates_kwargs(DecisionPathTableComponent)
    @delegates_doc(DecisionPathTableComponent)
    def decisionpath_table(self, title='Decision path', **kwargs):
        """Runs decision_trees tab inline in notebook"""
        comp = DecisionPathTableComponent(self._explainer, 
                header_mode="hidden", **kwargs)
        self._run_component(comp, title)

    @delegates_kwargs(DecisionPathTableComponent)
    @delegates_doc(DecisionPathTableComponent)
    def decisionpath_graph(self, title='Decision path', **kwargs):
        """Runs decision_trees tab inline in notebook"""
        comp = DecisionPathTableComponent(self._explainer, 
                header_mode="hidden", **kwargs)
        self._run_component(comp, title)

   

