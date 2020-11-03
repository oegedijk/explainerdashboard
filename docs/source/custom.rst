Custom Dashboards
*****************

By re-using the modular  :ref:`ExplainerComponents and connectors<ExplainerComponents>` 
it becomes very easy to build your own custom dashboards, without needing to know much about 
web development or even much about `plotly dash <https://dash.plotly.com/>`_, 
which is the underlying technology that ``explainerdashboard`` is built on.

You can get some inspiration from the `explainerdashboard composites <https://github.com/oegedijk/explainerdashboard/blob/master/explainerdashboard/dashboard_components/composites.py>`_
that build the layout of the default dashboard tabs.

Simplest Example
================

A very simple example of a custom dashboard would be::

    import dash_html_components as html
    from explainerdashboard.custom import *

    class CustomDashboard(ExplainerComponent):
        def __init__(self, explainer):
            super().__init__(explainer)
            self.dependence = ShapDependenceComponent(explainer, 
                    hide_selector=True, hide_cats=True, hide_index=True, col="Fare")
            self.register_components()

        def layout(self):
            return html.Div([self.dependence.layout()])

    ExplainerDashboard(explainer, CustomDashboard).run()

So you need to import all ExplainerComponents from ``explainerdashboard.custom``.
Then you define your custom dashboard wrapped inside a class that derives from 
``ExplainerComponent``. You need to call the init of the parent class with
``super().__init__(explainer)``. You instantiate the components that you wish
to include as attributes in your init: ``self.dependence = ShapDependenceComponent(explainer, ...)``.
You set the options of the component as you see fit (hiding toggles and dropdowns for example).
By calling ``self.register_components()`` you automatically register all 
ExplainerComponents that have been declared in the init, so that their 
callbacks will also be registered by the app. Then you define the layout in the 
``def layout(self)`` method. In this case we only include the 
``self.dependence.layout()`` wrapped inside a ``html.Div``.

You then pass this ``CustomDashboard`` to ``ExplainerDashboard`` and run it as usual.

You can find the list of all ``ExplainerComponents`` in the :ref:`documentation<ExplainerComponents>`.

.. note::
    To save on boilerplate code, parameters in the init will automagically be 
    stored to attributes by ``super().__init__()``. So in the example below 
    you do not have to explicitly call ``self.a = a`` in the init::

        class CustomDashboard(ExplainerComponent):
            def __init__(self, explainer, a=1):
                super().__init__(explainer)

        custom = CustomDashboard(explainer)
        assert custom.a == 1


Constructing layout
===================

You construct the layout using ``dash_bootstrap_components`` and
``dash_html_components``:

dash_bootstrap_components
-------------------------
Using the ``dash_bootstrap_components`` library it is very easy to construct
a modern looking web, responsive interface with just a few lines of python code. 

The basis of any layout is that you divide your layout
into ``dbc.Rows`` and then divide each row into a number of ``dbc.Cols`` where the total 
column widths should add up to 12. (e.g. two columns of width 6 each)

Then ``dash_bootstrap_components`` offer a lot of other modern web design 
elements such as cards, modals, etc that you can find more information on in
their documentation: `https://dash-bootstrap-components.opensource.faculty.ai/ <https://dash-bootstrap-components.opensource.faculty.ai/>`_

dash_html_components
--------------------

If you know a little bit of html then using ``dash_html_components`` you
can add further elements to your design. For example in order to insert a header
add ``html.H1("This is my header!")``, etc.


Elaborate Example
=================

CustomModelTab
--------------

A more elaborate example is below where we include three components: the 
precision graph, the shap summary and the shap dependence component, and
add explanatory text on either side of each component. The ``ShapSummaryDependenceConnector``
connects a ShapSummaryComponent and a ShapDependenceComponent so that when you 
select a feature in the summary, it automatically gets selected in the dependence plot::

    import dash_html_components as html
    import dash_bootstrap_components as dbc

    from explainerdashboard.custom import *
    from explainerdashboard import ExplainerDashboard

    class CustomModelTab(ExplainerComponent):
        def __init__(self, explainer):
            super().__init__(explainer, title="Titanic Explainer")
            self.precision = PrecisionComponent(explainer, 
                                    hide_cutoff=True, hide_binsize=True, 
                                    hide_binmethod=True, hide_multiclass=True,
                                    hide_selector=True,
                                    cutoff=None)
            self.shap_summary = ShapSummaryComponent(explainer, 
                                    hide_title=True, hide_selector=True,
                                    hide_depth=True, depth=8, 
                                    hide_cats=True, cats=True)
            self.shap_dependence = ShapDependenceComponent(explainer, 
                                    hide_title=True, hide_selector=True,
                                    hide_cats=True, cats=True, 
                                    hide_index=True,
                                    col='Fare', color_col="PassengerClass")
            self.connector = ShapSummaryDependenceConnector(
                    self.shap_summary, self.shap_dependence)
            
            self.register_components()
            
        def layout(self):
            return dbc.Container([
                html.H1("Titanic Explainer"),
                dbc.Row([
                    dbc.Col([
                        html.H3("Model Performance"),
                        html.Div("As you can see on the right, the model performs quite well."),
                        html.Div("The higher the predicted probability of survival predicted by"
                                "the model on the basis of learning from examples in the training set"
                                ", the higher is the actual percentage for a person surviving in "
                                "the test set"),
                    ], width=4),
                    dbc.Col([
                        html.H3("Model Precision Plot"),
                        self.precision.layout()
                    ])
                ]),
                dbc.Row([
                    dbc.Col([
                        html.H3("Feature Importances Plot"),
                        self.shap_summary.layout()
                    ]),
                    dbc.Col([
                        html.H3("Feature importances"),
                        html.Div("On the left you can check out for yourself which parameters were the most important."),
                        html.Div(f"{self.explainer.columns_ranked_by_shap(cats=True)[0]} was the most important"
                                f", followed by {self.explainer.columns_ranked_by_shap(cats=True)[1]}"
                                f" and {self.explainer.columns_ranked_by_shap(cats=True)[2]}."),
                        html.Div("If you select 'detailed' you can see the impact of that variable on "
                                "each individual prediction. With 'aggregate' you see the average impact size "
                                "of that variable on the finale prediction."),
                        html.Div("With the detailed view you can clearly see that the the large impact from Sex "
                                "stems both from males having a much lower chance of survival and females a much "
                                "higher chance.")
                    ], width=4)
                ]),
                dbc.Row([
                    dbc.Col([
                        html.H3("Relations between features and model output"),
                        html.Div("In the plot to the right you can see that the higher the priace"
                                "of the Fare that people paid, the higher the chance of survival. "
                                "Probably the people with more expensive tickets were in higher up cabins, "
                                "and were more likely to make it to a lifeboat."),
                        html.Div("When you color the impacts by the PassengerClass, you can clearly see that "
                                "the more expensive tickets were mostly 1st class, and the cheaper tickets "
                                "mostly 3rd class."),
                        html.Div("On the right you can check out for yourself how different features impact "
                                "the model output."),
                    ], width=4),
                    dbc.Col([
                        html.H3("Feature impact plot"),
                        self.shap_dependence.layout()
                    ]),
                ])
            ])
    
    ExplainerDashboard(explainer, CustomModelTab, hide_header=True).run()


CustomPredictionsTab
--------------------

We can also add another tab to investigate individual predictions, that 
includes an index selector, a SHAP contributions graph and a Random Forest
individual trees graph. The ``IndexConnector`` connects the index selected
in ``ClassifierRandomIndexComponent`` with the index dropdown in the 
contributions graph and trees components. We also pass a 
custom `dbc theme <https://dash-bootstrap-components.opensource.faculty.ai/docs/themes/>`_ 
called FLATLY as a custom css file::

    class CustomPredictionsTab(ExplainerComponent):
        def __init__(self, explainer):
            super().__init__(explainer, title="Predictions")
            
            self.index = ClassifierRandomIndexComponent(explainer, 
                                                        hide_title=True, hide_index=False, 
                                                        hide_slider=True, hide_labels=True, 
                                                        hide_pred_or_perc=True, 
                                                        hide_selector=True, hide_button=False)
            
            self.contributions = ShapContributionsGraphComponent(explainer, 
                                                                hide_title=True, hide_index=True, 
                                                                hide_depth=True, hide_sort=True, 
                                                                hide_orientation=True, hide_cats=True, 
                                                                hide_selector=True,  
                                                                sort='importance')
            
            self.trees = DecisionTreesComponent(explainer, 
                                                hide_title=True, hide_index=True, 
                                                hide_highlight=True, hide_selector=True)

            
            self.connector = IndexConnector(self.index, [self.contributions, self.trees])
            
            self.register_components()
            
        def layout(self):
            return dbc.Container([
                dbc.Row([
                    dbc.Col([
                        html.H3("Enter name:"),
                        self.index.layout()
                    ])
                ]),
                dbc.Row([
                    dbc.Col([
                        html.H3("Contributions to prediction:"),
                        self.contributions.layout()
                    ]),

                ]),
                dbc.Row([

                    dbc.Col([
                        html.H3("Every tree in the Random Forest:"),
                        self.trees.layout()
                    ]),
                ])
            ])

    ExplainerDashboard(explainer, [CustomModelTab, CustomPredictionsTab], 
                   title='Titanic Explainer',
                   header_hide_selector=True, 
                   external_stylesheets=[dbc.themes.FLATLY]).run()


.. image:: screenshots/custom_dashboard.*







