![GitHub Workflow Status (branch)](https://img.shields.io/github/workflow/status/oegedijk/explainerdashboard/explainerdashboard/master?style=plastic)
![https://pypi.python.org/pypi/explainerdashboard/](https://img.shields.io/pypi/v/explainerdashboard.svg)
![https://anaconda.org/conda-forge/explainerdashboard/](https://anaconda.org/conda-forge/explainerdashboard/badges/version.svg)
[![codecov](https://codecov.io/gh/oegedijk/explainerdashboard/branch/master/graph/badge.svg?token=0XU6HNEGBK)](undefined)

# explainerdashboard
by: Oege Dijk

This package makes it convenient to quickly deploy a dashboard web app
that explains the workings of a (scikit-learn compatible) machine 
learning model. The dashboard provides interactive plots on model performance, 
feature importances, feature contributions to individual predictions, 
"what if" analysis,
partial dependence plots, SHAP (interaction) values, visualisation of individual
decision trees, etc. 

You can also interactively explore components of the dashboard in a 
notebook/colab environment (or just launch a dashboard straight from there). 
Or design a dashboard with your own custom layout and explanations (thanks
to the modular design of the library).

 Examples deployed at: [titanicexplainer.herokuapp.com](http://titanicexplainer.herokuapp.com), 
 detailed documentation at [explainerdashboard.readthedocs.io](http://explainerdashboard.readthedocs.io), 
 example notebook on how to launch dashboard for different models [here](https://github.com/oegedijk/explainerdashboard/blob/master/dashboard_examples.ipynb), and an example notebook on how to interact with the explainer object [here](https://github.com/oegedijk/explainerdashboard/blob/master/explainer_examples.ipynb).

 Works with scikit-learn, xgboost, catboost, lightgbm and others.

 ## Installation

You can install the package through pip:

`pip install explainerdashboard`

or conda-forge:

`conda install -c conda-forge explainerdashboard`

## Demonstration GIF:

![explainerdashboard.gif](explainerdashboard.gif)

<!-- [![Dashboard Screenshot](https://i.postimg.cc/Gm8RnKVb/Screenshot-2020-07-01-at-13-25-19.png)](https://postimg.cc/PCj9mWd7) -->

## Background

In a lot of organizations, especially governmental, but with the GDPR also increasingly in private sector, it is becoming more and more important to be able to explain the inner workings of your machine learning algorithms. Customers have to some extent a right to an explanation why they received a certain prediction, and more and more internal and external regulators require it. With recent innovations in explainable AI (e.g. SHAP values) the old black box trope is no longer valid, but it can still take quite a bit of data wrangling and plot manipulation to get the explanations out of a model. This library aims to make this easy.

The goal is manyfold:
- Make it easy for data scientists to quickly inspect the workings and performance of their model in a few lines of code
- Make it possible for non data scientist stakeholders such as managers, directors, internal and external watchdogs to interactively inspect the inner workings of the model without having to depend on a data scientist to generate every plot and table
- Make it easy to build an application that explains individual predictions of your model for customers that ask for an explanation
- Explain the inner workings of the model to the people working (human-in-the-loop) with it so that they gain understanding what the model does and doesn't do. This is important so that they can gain an intuition for when the model is likely missing information and may have to be overruled. 


The library includes:
- *Shap values* (i.e. what is the contributions of each feature to each individual prediction?)
- *Permutation importances* (how much does the model metric deteriorate when you shuffle a feature?)
- *Partial dependence plots* (how does the model prediction change when you vary a single feature?
- *Shap interaction values* (decompose the shap value into a direct effect an interaction effects)
- For Random Forests and xgboost models: visualisation of individual decision trees
- Plus for classifiers: precision plots, confusion matrix, ROC AUC plot, PR AUC plot, etc
- For regression models: goodness-of-fit plots, residual plots, etc. 

The library is designed to be modular so that it should be easy to design your own interactive dashboards with plotly dash, with most of the work of calculating and formatting data, and rendering plots and tables handled by `explainerdashboard`, so that you can focus on the layout
and project specific textual explanations. (i.e. design it so that it will be interpretable for business users in your organization, not just data scientists)

Alternatively, there is a built-in standard dashboard with pre-built tabs (that you can switch off individually)

## Examples of use

Fitting a model, building the explainer object, building the dashboard, and then running it can be as simple as:

```python
ExplainerDashboard(ClassifierExplainer(RandomForestClassifier().fit(X_train, y_train), X_test, y_test)).run()
```

Below a multi-line example, adding a few extra paramaters. 
You can group onehot encoded categorical variables together using the `cats` 
parameter. You can either pass a dict specifying a list of onehot cols per
categorical feature, or if you encode using e.g. 
`pd.get_dummies(df.Name, prefix=['Name'])` (resulting in column names `'Name_Adam', 'Name_Bob'`) 
you can simply pass the prefix `'Name'`:

```python
from sklearn.ensemble import RandomForestClassifier
from explainerdashboard import ClassifierExplainer, ExplainerDashboard
from explainerdashboard.datasets import titanic_survive, titanic_names

feature_descriptions = {
    "Sex": "Gender of passenger",
    "Gender": "Gender of passenger",
    "Deck": "The deck the passenger had their cabin on",
    "PassengerClass": "The class of the ticket: 1st, 2nd or 3rd class",
    "Fare": "The amount of money people paid", 
    "Embarked": "the port where the passenger boarded the Titanic. Either Southampton, Cherbourg or Queenstown",
    "Age": "Age of the passenger",
    "No_of_siblings_plus_spouses_on_board": "The sum of the number of siblings plus the number of spouses on board",
    "No_of_parents_plus_children_on_board" : "The sum of the number of parents plus the number of children on board",
}

X_train, y_train, X_test, y_test = titanic_survive()
train_names, test_names = titanic_names()
model = RandomForestClassifier(n_estimators=50, max_depth=5)
model.fit(X_train, y_train)

explainer = ClassifierExplainer(model, X_test, y_test, 
                                cats=['Deck', 'Embarked',
                                    {'Gender': ['Sex_male', 'Sex_female', 'Sex_nan']}],
                                descriptions=feature_descriptions, # defaults to None
                                labels=['Not survived', 'Survived'], # defaults to ['0', '1', etc]
                                idxs = test_names, # defaults to X.index
                                index_name = "Passenger", # defaults to X.index.name
                                target = "Survival", # defaults to y.name
                                )

db = ExplainerDashboard(explainer, 
                        title="Titanic Explainer", # defaults to "Model Explainer"
                        whatif=False, # you can switch off tabs with bools
                        )
db.run(port=8050)
```

or for a regression model:

```python
X_train, y_train, X_test, y_test = titanic_fare()
model = RandomForestRegressor().fit(X_train, y_train)

explainer = RegressionExplainer(model, X_test, y_test, 
                                cats=['Deck', 'Embarked', 'Sex'],
                                descriptions=feature_descriptions, # defaults to None
                                idxs = test_names, # defaults to X.index
                                index_name = "Passenger", # defaults to X.index.name
                                target = "Fare", # defaults to y.name
                                units = "$", # defaults to ""
                                )

db = ExplainerDashboard(explainer).run()
```

(`y` is actually optional, although some parts of the dashboard will obviously 
not work: `ExplainerDashboard(ClassifierExplainer(model, X_test)).run()`)


### From within a notebook

When working inside jupyter or Google Colab you can use 
`ExplainerDashboard(mode='inline')`, `ExplainerDashboard(mode='external')` or
`ExplainerDashboard(mode='jupyterlab')`, to run the dashboard inline in the notebook,
or in a seperate tab but keep the notebook interactive. 

There is also a specific interface for quickly displaying interactive components
inline in your notebook: `InlineExplainer()`. For example you can use 
`InlineExplainer(explainer).shap.dependence()` to display the shap dependence
component interactively in your notebook output cell.

## Command line tool

You can store explainers to disk with `explainer.dump("explainer.joblib")`
and then run them from the command-line:

```bash
$ explainerdashboard run explainer.joblib
```

Or store the full configuration of a dashboard to `.yaml` with e.g.
`dashboard.to_yaml("dashboard.yaml")` and run it with:

```bash
$ explainerdashboard run dashboard.yaml
```

See [explainerdashboard CLI documentation](https://explainerdashboard.readthedocs.io/en/latest/cli.html)
for details. 

## Customizing your dashboard

The dashboard is highly modular and customizable so that you can adjust it your
own needs and project. 

### switching off tabs

You can switch off individual tabs using boolean flags, e.g.:

```python
ExplainerDashboard(explainer,
                    importances=False,
                    model_summary=True,
                    contributions=True,
                    whatif=True,
                    shap_dependence=True,
                    shap_interaction=False,
                    decision_trees=True)
```

### passing parameters as `**kwargs`

The dashboard consists of independent `ExplainerComponents` that take their
own parameters. For example hiding certain toggles (e.g. `hide_cats=True`) or
setting default values (e.g. `col='Fare'`). When you start your `ExplainerDashboard` 
all the `**kwargs` will be passed down to all each `ExplainerComponents`.
All the components with their parameters can be found [in the documentation](https://explainerdashboard.readthedocs.io/en/latest/components.html).
Some examples of useful parameters to pass:

```python
ExplainerDashboard(explainer, 
                    no_permutations=True, # do not show or calculate permutation importances
                    higher_is_better=False, # flip green and red in contributions graph
                    hide_cats=True, # hide the group cats toggles
                    hide_depth=True, # hide the depth (no of features) dropdown
                    hide_sort=True, # hide sort type dropdown in contributions graph/table
                    hide_orientation=True, # hide orientation dropdown in contributions graph/table
                    hide_type=True, # hide shap/permutation toggle on ImportancesComponent 
                    hide_dropna=True, # hide dropna toggle on pdp component
                    hide_sample=True, # hide sample size input on pdp component
                    hide_gridlines=True, # hide gridlines on pdp component
                    hide_gridpoints=True, # hide gridpoints input on pdp component
                    hide_cutoff=True, # hide cutoff selector on classification components
                    hide_percentage=True, # hide percentage toggle on classificaiton components
                    hide_log_x=True, # hide x-axis logs toggle on regression plots
                    hide_log_y=True, # hide y-axis logs toggle on regression plots
                    hide_ratio=True, # hide the residuals type dropdown
                    hide_points=True, # hide the show violin scatter markers toggle
                    hide_winsor=True, # hide the winsorize input
                    # setting default values
                    col='Fare', # initial feature in shap graphs
                    color_col='Age', # color feature in shap dependence graph
                    interact_col='Age', # interaction feature in shap interaction
                    cats=False, # do not group categorical onehot features
                    depth=5, # only show top 5 features
                    sort = 'low-to-high', # sort features from lowest shap to highest in contributions graph/table
                    orientation='horizontal', # horizontal bars in contributions graph
                    index='Rugg, Miss. Emily', # initial index to display
                    pdp_col='Fare', # initial pdp feature
                    cutoff=0.8, # cutoff for classification plots
                    round=2 # rounding to apply to floats
                    )
```

### designing own layout

All the components in the dashboard are modular and re-usable, which means that 
you can build your own custom [dash](https://dash.plotly.com/) dashboards 
around them.

By using the built-in `ExplainerComponent` class it is easy to build your
own layouts, with just a bare minimum of knowledge of HTML and [bootstrap](https://dash-bootstrap-components.opensource.faculty.ai/docs/quickstart/). For
example if you only wanted to display the `ConfusionMatrixComponent` and 
`ShapContributionsGraphComponent`, but hide
a few toggles:

```python
from explainerdashboard.custom import *

class CustomDashboard(ExplainerComponent):
    def __init__(self, explainer, **kwargs):
        super().__init__(explainer, title="Custom Dashboard")
        self.confusion = ConfusionMatrixComponent(explainer,
                            hide_selector=True, hide_percentage=True,
                            cutoff=0.75)
        self.contrib = ShapContributionsGraphComponent(explainer,
                            hide_selector=True, hide_cats=True, 
                            hide_depth=True, hide_sort=True,
                            index='Rugg, Miss. Emily')
        self.register_components()
        
    def layout(self):
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("Custom Demonstration:"),
                    html.H3("How to build your own layout using ExplainerComponents.")
                ])
            ]),
            dbc.Row([
                dbc.Col([
                    self.confusion.layout(),
                ]),
                dbc.Col([
                    self.contrib.layout(),
                ])
            ])
        ])

db = ExplainerDashboard(explainer, CustomDashboard, hide_header=True).run()
```

You can use this to define your own layouts, specifically tailored to your
own model, project and needs. See [custom dashboard documentation](https://explainerdashboard.readthedocs.io/en/latest/custom.html)
for more details. 

## Deployment

If you wish to use e.g. ``gunicorn`` to deploy the dashboard you should add 
`server = db.app.server` to your code to expose the Flask server. You can then 
start the server with e.g. `gunicorn dashboard:server` 
(assuming the file you defined the dashboard in was called `dashboard.py`). 
See also the [ExplainerDashboard section](https://explainerdashboard.readthedocs.io/en/latest/dashboards.html) 
and the [deployment section of the documentation](https://explainerdashboard.readthedocs.io/en/latest/deployment.html).


## Documentation

Documentation can be found at [explainerdashboard.readthedocs.io](https://explainerdashboard.readthedocs.io/en/latest/).

Example notebook on how to launch dashboards for different model types here: [dashboard_examples.ipynb](https://github.com/oegedijk/explainerdashboard/blob/master/dashboard_examples.ipynb).

Example notebook on how to interact with the explainer object here: [explainer_examples.ipynb](https://github.com/oegedijk/explainerdashboard/blob/master/explainer_examples.ipynb).

Example notebook on how to design a custom dashboard: [custom_examples.ipynb](https://github.com/oegedijk/explainerdashboard/blob/master/custom_examples.ipynb).



## Deployed example:

You can find an example dashboard at [titanicexplainer.herokuapp.com](http://titanicexplainer.herokuapp.com) 

(source code at [https://github.com/oegedijk/explainingtitanic](https://github.com/oegedijk/explainingtitanic))
