
# TODO:

## Layout:
- Find a proper frontender to help :)

## dfs:
- wrap shap values in pd.DataFrames?
- wrap predictions in pd.Series?

## Plots:


### Classifier plots:
- confusion matrix: display both count and percentage
- pdp: add multiclass option
- add classification model summary
- include cumulative lift curve to standard dashboard
- add cost calculator: cost of FPs and FNs
- classification plot:
    - round percentage to 1 digit
    - add `<br>` between count and percentage

### Regression plots:

## Explainers:
- add target name 
- add plain language explanations

## notebooks:
- add binder/colab links on github

## Dashboard:
- Add pandas profiling type col histograms, bar charts, correlation graphs, etc

### Components
- add pos_label_name property to PosLabelConnector search
- add "number of indexes" indicator to RandomIndexComponents for current restrictions

## Methods:
- Add LIME values
    - but tricky how to set kernel, model, etc
    - Lime values take a lot more DS judgement than SHAP values
- Add this method? : https://arxiv.org/abs/2006.04750?

## Tests:
- test model_output='probability' and 'raw' or 'logodds' seperately
- write tests for explainer_methods
- write tests for explainer_plots
- add test coverage 

## Docs:
- add docstrings to explainer_plots
- add screenshots of components to docs
- move screenshots to separate folder
- document PosLabelSelector and PosLabelConnector, e.g.:
        self.connector = PosLabelConnector(self.roc_auc, self)
        self.register_components(self.connector)
- add documentation how to deploy to heroku:
    - mock xgboost
    - uninstall xgboost with shell buildpack
    - graphviz buildpack


## Library level:
- Add Altair (vega) plots for easy inclusion in websites or fastpages blogs
- Long term: add option to load from directory with pickled model, data csv and config file
- add more screenshots to README with https://postimages.org/
- add test coverage: https://github.com/marketplace/actions/coveralls-python
- submit pull request to shap with broken test for https://github.com/slundberg/shap/issues/723