
# TODO:
- find out why dtreeviz for regression no longer working
- find a way to plot individual xgboost trees (now in dtreeviz!)
- replace custom permutation importances by sklearn permutation importances?
    - or submit PR to sklearn to support multi col permuations for cats?
    - or spend some time optimizing own permutation importance code?


## Layout:
- Find a proper frontender to help :)

## dfs:
- wrap shap values in pd.DataFrames?
- wrap predictions in pd.Series?

## Plots:
- Contributions: order by global mean_abs_shap or by index specific shap

### Classifier plots:
- confusion matrix: display both count and percentage
- pdp: add multiclass option
- add classification model summary
- include cumulative lift curve to standard dashboard
- add cost calculator: cost of FPs and FNs

### Regression plots:

## Explainers:


## notebooks:
- add binder/colab link on github

## Dashboard:
- Add pandas profiling type col histograms, bar charts, correlation graphs, etc

### Components
- add pos_label_name property to PosLabelConnector search
- add "number of indexes" indicator to RandomIndexComponents for current restrictions

## Methods:
- Add LIME values

## Tests:
- add multiclass classifier tests
- add dashboard integration tests using dash tests
- add ExplainerDashboard intergration test
- Add tests for decisiontrees, extratrees
- test model_output='probability' and 'raw' or 'logodds' seperately
- add tests for explainer_methods
- add test coverage (add a badge)

## Docs:
- add docstrings to explainer_methods
- add docstrings to explainer_plots
- add screenshots of components to docs
- move screenshots to separate folder
- document PosLabelSelector and PosLabelConnector, e.g.:
        self.connector = PosLabelConnector(self.roc_auc, self)
        self.register_components(self.connector)


## Library level:
- Add Altair (vega) plots for easy inclusion in websites
- Long term: add option to load from directory with pickled model, data csv and config file
- add more screenshots to README with https://postimages.org/
- https://github.com/marketplace/actions/coveralls-python
- submit pull request to dtreeviz to accept shadowtree as parameter
- submit pull request to shap with broken test for https://github.com/slundberg/shap/issues/723