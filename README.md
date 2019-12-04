# explainerdashboard
by: Oege Dijk

This package makes it convenient to quickly explain the workings of a
fitted machine learning model using either interactive plots in e.g. Jupyter Notebook or 
deploying an interactive dashboard (based on Flask/Dash) that allows you to quickly explore
the impact of different features on model predictions.

This includes:
- *Shap values* (i.e. what is the contributions of each feature to each individual prediction?)
- *Permutation importances* (how much does the model metric deteriorate when you shuffle a feature?)
- *Partial dependence plots* (how does the model prediction change when you vary a single feature?
- *Shap interaction values* (decompose the shap value into a direct effect an interaction effects)
- For Random Forests: what is the prediction of each *individual decision tree*, and what is the path through each tree?
- Plus for classifiers: precision plots, confusion matrixm, ROC AUC plot, PR AUC plot, etc

## Installation

You can install the package through pip:

`pip install explainerdashboard`

## Documentation

Documentation can be found at [explainerdashboard.readthedocs.io](https://explainerdashboard.readthedocs.io/en/latest/).

(NOTE: at the moment some dependency issue is preventing sphinx from correctly rendering all the autodoc features)

## A simple demonstration

### Constructing an ExplainerBunch

The package works by first constructing an ExplainerBunch object. You can then use this ExplainerBunch to manually call different plots, or to start the dashboard. You construct the ExplainerBunch instancefrom your fitted `model`, a feature matrix `X`, and optionally the corresponding target values `y`. 

In addition you can pass:
- `metric`: permutation importances get calculated against a particular metric (for regression defaults to `r2_score` and for classification to `roc_auc_score`)
- `cats`: a list of onehot encoded variables (e.g. if encoded as 'Gender_Female', 'Gender_Male' you would pass `cats=['Gender']`). This allows you to group the onehotencoded columns together in various plots with the argument `cats=True`. 
- `idxs`: a list of indentifiers for each row in your dataset. This makes it easier to look up predictions for specific id's.
- `labels`: for classifier models a list of labels for the classes of your model.
- `na_fill`: Value used to fill missing values (default to -999)

E.g.:

```
X_train, y_train, X_test, y_test = titanic_survive()
train_names, test_names = titanic_names()

model = RandomForestClassifier(n_estimators=50, max_depth=5)
model.fit(X_train, y_train)


explainer = RandomForestClassifierBunch(model, X_test, y_test, roc_auc_score, 
                               cats=['Sex', 'Deck', 'Embarked'],
                               idxs=test_names, #names of passengers 
                               labels=['Not survived', 'Survived'])
```

You can then easily inspect the model using various plot function, such as e.g.:
- `explainer.plot_confusion_matrix(cutoff=0.6, normalized=True)`
- `explainer.plot_importances(cats=True)`
- `explainer.plot_pdp('PassengerClass', index=0)`
- `explainer.plot_shap_dependence('Age')`, etc.

See the [explainer_examples.ipynb](explainer_examples.ipynb) and [documentation](https://explainerdashboard.readthedocs.io/en/latest/) for more details.

### Starting an ExplainerDashboard
Once you have constructed an ExplainerBunch object, you can then pass this along to an
ExplainerDashboard that builds an interactive Plotly Dash analytical dashboard for 
easily exploring the various plots and analysis mentioned earlier. 

You can use a series of booleans to switch on or off certain tabs of the dashboard.
(Calculating shap interaction values can take quite a but of time if you have a large dataset with a lot of feature, 
so if you are not really interested in them, it may make sense to switch that tab off.)

Any additional `**kwargs` get passed down to the individual tabs. (mostly `n_features` and `round` for now)

```
db = ExplainerDashboard(explainer, 'Titanic Explainer`,
                        model_summary=True,
                        contributions=True,
                        shap_dependence=True,
                        shap_interaction=False,
                        shadow_trees=True)
```

You then start the dashboard on a particular port with `db.run(port=8050)`. 

If you wish to use e.g. unicorn to deploy the dashboard you should add `server = db.app` to your code to expose the Flask server. It may take some time to calculate all the properties of the ExplainerBunch (especiialy shap interaction values) as these get calculated lazily. However you can save the ExplainerBunch to disk with e.g. joblib and then load the ExplainerBunch with pre-calculated properties whenever you wish to start the dashboard. 

See [dashboard_examples.ipynb](dashboard_examples.ipynb)


## Deployed example:

You can find an example dashboard at [titanicexplainer.herokuapp.com](http://titanicexplainer.herokuapp.com)

(source code at [https://github.com/oegedijk/explainingtitanic](https://github.com/oegedijk/explainingtitanic))
