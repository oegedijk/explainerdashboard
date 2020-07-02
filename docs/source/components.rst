ExplainerComponents
*******************

The dashboard is constructed out of ``ExplainerComponents``: self-contained
elements usually consisting of a plot or table and various dropdowns, sliders 
and toggles to manipulate that plot.

When you run ``ExplainerDashboard`` you get the default dashboard with basically
every component listed below with every toggle and slider visible. 

The ``ExplainerComponents`` make it very easy to construct your own dashboard
with your own layout, with specific explanations for the workings and results
of your model. So you can select which components to use, where to put them
in the layout, which toggles and sliders to display, and what the initial values
for component should be. This way you can also control which interactive 
aspects your end users can and cannot control.  


ExplainerComponent
==================

Each component subclasses ``ExplainerComponent`` which provides the basic
functionality of registering subcomponents, dependencies, adding a header to 
the layout, registering callbacks of subcomponents, and calculating dependencies.


.. autoclass:: explainerdashboard.dashboard_components.dashboard_methods.ExplainerComponent
   :members:

.. autoclass:: explainerdashboard.dashboard_components.dashboard_methods.ExplainerHeader
   :members:

shap_components
===============

.. automodule:: explainerdashboard.dashboard_components.shap_components
   :members:


overview_components
===================

.. automodule:: explainerdashboard.dashboard_components.overview_components
   :members:


classifier_components
=====================

.. automodule:: explainerdashboard.dashboard_components.classifier_components
   :members:


regression_components
=====================

.. automodule:: explainerdashboard.dashboard_components.regression_components
   :members:


decisiontree_components
=======================

.. automodule:: explainerdashboard.dashboard_components.decisiontree_components
   :members:

connectors
==========

.. automodule:: explainerdashboard.dashboard_components.connectors
   :members:

composites
==========

.. automodule:: explainerdashboard.dashboard_components.composites
   :members:

   
