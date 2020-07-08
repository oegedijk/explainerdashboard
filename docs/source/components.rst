ExplainerComponents
*******************

The dashboard is constructed out of ``ExplainerComponents``: self-contained
elements usually consisting of a plot or table and various dropdowns, sliders 
and toggles to manipulate that plot. Components can be connected with connectors,
so that when you select an index in one component that automatically updates the
index in another component for example.

When you run ``ExplainerDashboard`` you get the default dashboard with basically
every component listed below with every toggle and slider visible. 

The ``ExplainerComponents`` make it very easy to construct your own dashboard
with your own layout, with specific explanations for the workings and results
of your model. So you can select which components to use, where to put them
in the layout, which toggles and sliders to display, and what the initial values
for component should be. This way you can also control which interactive 
aspects your end users can and cannot control.  

You import the components with ``from explainerdashboard.dashboard_components import *``

A simple example, where you build a page with only a Shap Dependence component,
but with no group cats or highlight toggle, and initial feature set to 'Fare'::

    from jupyter_dash import JupyterDash
    import dash_bootstrap_components as dbc
    import dash_html_components as html

    from explainerdashboard.dashboard_components import *

    header = ExplainerHeader(explainer, mode="standalone")
    shap_dependence = ShapDependenceComponent(explainer, 
                            hide_title=True, hide_cats=True, hide_highlight=True,
                            cats=True, col='Fare')
            
    layout = dbc.Container([
        dbc.Col([
            header.layout(),
            shap_dependence.layout()
        ])  
    ])
    
    app = JupyterDash()
    app.title = "Titanic Explainer"
    app.layout = layout
    shap_dependence.register_callbacks(app)
    app.run_server() 


Using the awesome ``dash_bootstrap_components`` library it is very easy to quickly
design a modern looking web layout. Then you can simply add ``component.layout()`` 
to the layout and call ``component.register_callbacks(app)`` to register the 
callbacks to the app, and start the server. 

ExplainerComponent and ExplainerHeader
======================================

Each component subclasses ``ExplainerComponent`` which provides the basic
functionality of registering subcomponents, dependencies, adding a header to 
the layout, registering callbacks of subcomponents, and calculating dependencies.

ExplainerComponent
------------------

.. autoclass:: explainerdashboard.dashboard_components.dashboard_methods.ExplainerComponent
   :members:

ExplainerHeader
---------------

.. autoclass:: explainerdashboard.dashboard_components.dashboard_methods.ExplainerHeader
   :members:

shap_components
===============

ShapSummaryComponent
--------------------

.. autoclass:: explainerdashboard.dashboard_components.shap_components.ShapSummaryComponent
   :members:

ShapDependenceComponent
-----------------------

.. autoclass:: explainerdashboard.dashboard_components.shap_components.ShapDependenceComponent
   :members:

ShapSummaryDependenceConnector
------------------------------

.. autoclass:: explainerdashboard.dashboard_components.shap_components.ShapSummaryDependenceConnector
   :members:

InteractionSummaryComponent
---------------------------

.. autoclass:: explainerdashboard.dashboard_components.shap_components.InteractionSummaryComponent
   :members:

InteractionDependenceComponent
------------------------------

.. autoclass:: explainerdashboard.dashboard_components.shap_components.InteractionDependenceComponent
   :members:

InteractionSummaryDependenceConnector
-------------------------------------

.. autoclass:: explainerdashboard.dashboard_components.shap_components.InteractionSummaryDependenceConnector
   :members:

ShapContributionsTableComponent
-------------------------------

.. autoclass:: explainerdashboard.dashboard_components.shap_components.ShapContributionsTableComponent
   :members:

ShapContributionsGraphComponent
-------------------------------

.. autoclass:: explainerdashboard.dashboard_components.shap_components.ShapContributionsGraphComponent
   :members:



overview_components
===================

PredictionSummaryComponent
--------------------------

.. autoclass:: explainerdashboard.dashboard_components.overview_components.PredictionSummaryComponent
   :members:

ImportancesComponent
--------------------

.. autoclass:: explainerdashboard.dashboard_components.overview_components.ImportancesComponent
   :members:

PdpComponent
------------

.. autoclass:: explainerdashboard.dashboard_components.overview_components.PdpComponent
   :members:


classifier_components
=====================

PrecisionComponent
------------------

.. autoclass:: explainerdashboard.dashboard_components.classifier_components.PrecisionComponent
   :members:

ConfusionMatrixComponent
------------------------

.. autoclass:: explainerdashboard.dashboard_components.classifier_components.ConfusionMatrixComponent
   :members:

LiftCurveComponent
------------------

.. autoclass:: explainerdashboard.dashboard_components.classifier_components.LiftCurveComponent
   :members:

ClassificationComponent
-----------------------

.. autoclass:: explainerdashboard.dashboard_components.classifier_components.ClassificationComponent
   :members:

RocAucComponent
---------------

.. autoclass:: explainerdashboard.dashboard_components.classifier_components.RocAucComponent
   :members:

PrAucComponent
--------------

.. autoclass:: explainerdashboard.dashboard_components.classifier_components.PrAucComponent
   :members:


regression_components
=====================

PredictedVsActualComponent
--------------------------

.. autoclass:: explainerdashboard.dashboard_components.regression_components.PredictedVsActualComponent
   :members:

ResidualsComponent
------------------

.. autoclass:: explainerdashboard.dashboard_components.regression_components.ResidualsComponent
   :members:

ResidualsVsColComponent
-----------------------

.. autoclass:: explainerdashboard.dashboard_components.regression_components.ResidualsVsColComponent
   :members:

RegressionModelSummaryComponent
-------------------------------

.. autoclass:: explainerdashboard.dashboard_components.regression_components.RegressionModelSummaryComponent
   :members:



decisiontree_components
=======================

DecisionTreesComponent
----------------------

.. autoclass:: explainerdashboard.dashboard_components.decisiontree_components.DecisionTreesComponent
   :members:

DecisionPathTableComponent
--------------------------

.. autoclass:: explainerdashboard.dashboard_components.decisiontree_components.DecisionPathTableComponent
   :members:

DecisionPathGraphComponent
--------------------------

.. autoclass:: explainerdashboard.dashboard_components.decisiontree_components.DecisionPathGraphComponent
   :members:




connectors
==========

ClassifierRandomIndexComponent
------------------------------

.. autoclass:: explainerdashboard.dashboard_components.connectors.ClassifierRandomIndexComponent
   :members:

RegressionRandomIndexComponent
------------------------------

.. autoclass:: explainerdashboard.dashboard_components.connectors.RegressionRandomIndexComponent
   :members:

CutoffPercentileComponent
-------------------------

.. autoclass:: explainerdashboard.dashboard_components.connectors.CutoffPercentileComponent
   :members:

CutoffConnector
---------------

.. autoclass:: explainerdashboard.dashboard_components.connectors.CutoffConnector
   :members:

IndexConnector
--------------

.. autoclass:: explainerdashboard.dashboard_components.connectors.IndexConnector
   :members:

HighlightConnector
------------------

.. autoclass:: explainerdashboard.dashboard_components.connectors.HighlightConnector
   :members:



   
