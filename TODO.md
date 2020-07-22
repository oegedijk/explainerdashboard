
# TODO:
- find out why dtreeviz for regression no longer working
- find a way to plot individual xgboost trees (now in dtreeviz!)
- replace custom permutation importances by sklearn permuation importances?
    - or submit PR to sklearn to support multi col permuations?

## Layout:
- Find a proper frontender to help :)
- add RegressionRandomIndex to DecisionTrees tab

## dfs:
- wrap shap values in pd.DataFrames?
- wrap predictions in pd.Series?

## Plots:
- Add options sorting contributions plot from most negative to most positive
- Contributions: order by global mean_abs_shap or by index specific shap
- fix name of figure MEAN_ABS_SHAP
- highlight id in violin plots
- Add feature names to waterfall plot
- replace -999 in contributions table by "MISSING"
- add str indexes to shap detailed summary plots

### Classifier plots:
- confusion matrix: display both count and percentage

### Regression plots:
- make base contribution yellow
- call base contribution "starting average"
- add units to contributions table
- add units to pdp y axis
- add units to contributions graph y axis
- add units to decision trees plot

## Explainers:


## Dashboard:
- add option for vertical contributions?
 

### Components
- add pos_label_name property to PosLabelConnector search
- group cats in interaction does not update col options anymore.
- fix pdp component bug
- add "number of indexes" indicator to RandomIndexComponents for current restrictions


## Methods:
- Add LIME values

## Tests:
- add multiclass classifier tests
- add dashboard integration tests using dash tests
- add ExplainerDashboard intergration test
- Add tests for decisiontrees, extratrees
- test model_output='probability' and 'raw' or 'logodds' seperately
- add test coverage (add a badge)

## Docs:
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