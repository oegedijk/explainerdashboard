ExplainerDashboard
******************

In order to start an ``ExplainerDashboard`` you first need to contruct an ``Explainer`` instance.
On the basis of this explainer you can then quickly start an interactive dashboard. (note: if you 
are  working inside a jupyter notebook, then it's probably better to use 
``JupyterExplainerDashboard`` instead. See below)

You can pass a list of ``ExplainerComponents`` to the ``tabs`` parameter, or
alternatively switch off individual tabs to display using booleans. 
By default all tabs that are feasible will be displayed. Warning: the interactions tab can
take quite some time to compute, so you may want to switch it off if you're not particularly 
interested in interaction effects between features.

Some example code::

    from explainerdashboard.dashboards import ExplainerDashboard

    ExplainerDashboard(explainer, title="Titanic Explainer",
                        model_summary=True,  
                        contributions=True,
                        shap_dependence=False,
                        shap_interaction=False,
                        decision_trees=False).run(port=8051)



ExplainerDashboard
==================

.. currentmodule:: explainerdashboard.dashboards

.. autoclass:: ExplainerDashboard
   :members: run 
   :member-order: bysource


JupyterExplainerDashboard
=========================
If you are working within a notebook there is also an alternative ``JupyterExplainerDashboard`` that uses the ``JupyterDash`` 
backend instead of ``dash.Dash()``. You can either pass to ``run()`` ``mode='inline'`` or ``mode='jupyterlab'`` or 
``mode='external'`` to start the dashboard inline in a notebook, in seperate pane 
in jupyterlab, or in seperate browser tab respectively. This has the benefit
that you can keep working inside the notebook while the dashboard is running.

.. autoclass:: explainerdashboard.dashboards.JupyterExplainerDashboard
   :members: run
   :member-order: bysource

ExplainerTab
============

To run a single page of a particular ``ExplainerComponent`` or tab, there is ``ExplainerTab``. 
You can either pass the appropriate tab class definition, or a string identifier::

    ExplainerTab(explainer, "model_summary").run()
    ExplainerTab(explainer, "shap_dependence").run(port=8051)
    ExplainerTab(explainer, ShapInteractionsTab).run()


.. autoclass:: explainerdashboard.dashboards.ExplainerTab
   :members: run
   :member-order: bysource

JupyterExplainerTab
===================

Equivalent to JupyterExplainerDashboard, runs a single tab using the
JupyterDash() instead of dash.Dash(). You can either pass to ``run()`` 
``mode='inline'`` or ``mode='jupyterlab'`` or ``mode='external'`` to start 
the dashboard inline in a notebook, in seperate pane in jupyterlab, or in 
seperate browser tab respectively. 

.. autoclass:: explainerdashboard.dashboards.JupyterExplainerTab
   :members: run
   :member-order: bysource

