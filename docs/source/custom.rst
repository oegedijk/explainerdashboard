Custom Dashboards
*****************

Using the ``ExplainerComponents`` and connectors it becomes very easy to build 
your own custom dashboards, without needing to know much about web development 
or even much about `plotly dash <https://dash.plotly.com/>`_. 
(although I would recommend anyone to learn the latter)

You can get some inspiration from ``explainerdashboard.dashbord_components.composites``
on how you can construct simple layouts, combining different components and
connectors.

Constructing layout
===================

dash_bootstrap_components
-------------------------
Using the ``dash_bootstrap_components`` library it is very easy to construct
a modern looking web interface with just a few lines of python code. 

The basis of any layout is that you divide your layout
into rows and then divide each row into a number of columns where the total 
column widths should add upto 12. (e.g. two columns of width 6 each)

Then there are a lot of other components that you can find more information
on on the dbc site: https://dash-bootstrap-components.opensource.faculty.ai/

dash_html_components
--------------------

If you know a little bit of html then using ``dash_html_components`` you
can add further elements to your design. For example to insert a header
add ``html.H1("This is my header!")``.

Adding ExplainerComponents
--------------------------

The final element is ofcourse the ``ExplainerComponents``. To add them
to your design you need to simply instantiate them, add the ``component.layout()`` 
to your custom layout and then ``component.register_callbacks(app)``.

A very simple example would be::

    from jupyter_dash import JupyterDash
    import dash_bootstrap_components as dbc
    import dash_html_components as html

    from explainerdashboard.dashboard_components import *

    shap_dependence = ShapDependenceComponent(explainer)
            
    layout = html.Div([
            shap_dependence.layout() 
    ])
    
    app = JupyterDash()
    app.title = "Titanic Explainer"
    app.layout = layout
    shap_dependence.register_callbacks(app)
    app.run_server() 


You can add options to the component when you instantiate them. So for example
if you wish to hide the group cats toggle and start with the 'Fare' feature, you
would instantiate as ``shap_dependence = ShapDependenceComponent(explainer, hide_cats=True, col='Fare')``.
For all the options for the different ExplainerComponent check the :ref:`documentation<shap_components>`.

If you wrap your components into a custom ``ExplainerComponent``, and remember to
``register_components()``, then you don't need to seperately call `register_callbacks()`
on each component, but only on the composite component. If you then start the 
``ExplainerComponent`` using ``ExplainerDashboard()``,  ``calculate_dependencies()`` 
will automatically be run as well.

Below you can see three different design patterns on how to construct your
own custom dashboard.

Design patterns
===============

Standard flat dash design
-------------------------

The standard default dash way to define a custom dashboard is by instantiating
the components, then adding these into a layout, instantiating a dash app,
registering the callbacks, and starting the app::

    precision = PrecisionComponent(explainer, 
                            hide_cutoff=True, hide_binsize=True, 
                            hide_binmethod=True, hide_multiclass=True,
                            hide_selector=True,
                            cutoff=None)
    shap_summary = ShapSummaryComponent(explainer, 
                            hide_title=True, hide_selector=True,
                            hide_depth=True, depth=8, 
                            hide_cats=True, cats=True)
    shap_dependence = ShapDependenceComponent(explainer, 
                            hide_title=True, hide_selector=True,
                            hide_cats=True, cats=True, 
                            hide_index=True,
                            col='Fare', color_col="PassengerClass")
    connector = ShapSummaryDependenceConnector(shap_summary, shap_dependence)
            
    layout = dbc.Container([
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
                        precision.layout()
                    ])
                ]),
                dbc.Row([
                    dbc.Col([
                        html.H3("Feature Importances Plot"),
                        shap_summary.layout()
                    ]),
                    dbc.Col([
                        html.H3("Feature importances"),
                        html.Div("On the left you can check out for yourself which parameters were the most important."),
                        html.Div(f"{explainer.columns_ranked_by_shap(cats=True)[0]} was the most important"
                                f", followed by {explainer.columns_ranked_by_shap(cats=True)[1]}"
                                f" and {explainer.columns_ranked_by_shap(cats=True)[2]}."),
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
                        shap_dependence.layout()
                    ]),
                ])
            ])


    app = dash.Dash(__name__)
    app.title = "Titanic Explainer"
    app.layout = layout

    precision.register_callbacks(app)
    shap_summary.register_callbacks(app)
    shap_dependence.register_callbacks(app)
    connector.register_callbacks(app)

    app.run_server()



Wrapping dashboard into a class
-------------------------------

A slightly cleaner design consists of wrapping the layout into a CustomDashboard
class. 

Here we also calculate dependencies before we start the dashboard. 

ExplainerDashboard does the expensive calculations of e.g. shap values only when 
they are needed for an output, and then saves the result for all subsequent calls. 
However when starting a dashboard multiple components might request shap values
in parallel resulting in wasted cpu cycles and a slow boot. The solution is 
making sure these properties are all calculated before starting the dashboard.
ExplainerComponents come with a nice method ``calculate_dependencies()`` 
that does exactly this::

    class CustomDashboard():
        def __init__(self, explainer):
            self.explainer = explainer
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
            self.connector = ShapSummaryDependenceConnector(self.shap_summary, self.shap_dependence)
            
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
        
        def register_callbacks(self, app):
            self.precision.register_callbacks(app)
            self.shap_summary.register_callbacks(app)
            self.shap_dependence.register_callbacks(app)
            self.connector.register_callbacks(app)

        def calculate_dependencies(self):
            self.precision.calculate_dependencies()
            self.shap_summary.calculate_dependencies()
            self.shap_dependence.calculate_dependencies()
            self.connector.calculate_dependencies()

    db = CustomDashboard(explainer)
    
    app = JupyterDash(external_stylesheets=[dbc.themes.FLATLY], assets_url_path="")
    app.title = "Titanic Explainer"
    app.layout = db.layout()
    db.register_callbacks(app)
    db.calculate_dependencies()
    app.run_server(mode='external')

Custom ExplainerComponent and use ExplainerDashboard
----------------------------------------------------

A third method consists of inheriting from ExplainerComponent and then
running the page with ``ExplainerDashboard``. The main difference is calling the
``super().__init__()`` and calling ``register_components()`` inside the init.

The benefit is that you don't have to explicitly write the ``register_callbacks`` or
``calculate_dependencies`` method, as these get generated automatically 
when calling ``register_components``, and you don't have to write the ``dash`` 
boilerplate code. This means you can fully concentrate on just designing your 
layout and components::

    class CustomDashboard(ExplainerComponent):
        def __init__(self, explainer)
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
            self.connector = ShapSummaryDependenceConnector(self.shap_summary, self.shap_dependence)
            
            self.register_components(self.precision, self.shap_summary, self.shap_dependence, self.connector)
            
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
    
    ExplainerDashboard(explainer, CustomComponent, hide_header=True).run()





