
# TODO:
- find out why dtreeviz for regression no longer working
- find a way to plot individual xgboost trees
- replace custom permutation importances by sklearn permuation importances

## Layout:
- Find a proper frontender to help :)

## dfs:
- wrap shap values in pd.DataFrames?
- wrap predictions in pd.Series?

## Plots:
- Add options sorting contributions plot from most negative to most positive
- Contributions: order by global mean_abs_shap or by specific shap
- fix name of figure MEAN_ABS_SHAP
- Add Altair (vega) plots for easy inclusion in websites
- highlight id in violin plots
- Add feature names to waterfall plot
- replace -999 in contributions table by "MISSING"

### Classifier plots:
- confusion matrix: display both count and percentage

### Regression plots:
- regression plots: only take log of x-axis or y-axis
- add cats option (violin plots?) to plot_residuals_vs_feature
- add log ratio

## Explainers:
- check all shap values >-1 and <1 for model_output=probability

## Dashboard:
- make alternative tight layout for mode='inline' 
- add option for vertical contributions?
 

### Components
- add hide_title to all components

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
- Add SKETCHY theme example

## Library level:
- add @delegates_kwargs_and_doc_to() 
- add more screenshots to README with https://postimages.org/
- add badges to README: https://github.com/badges/shields
-> https://github.com/oegedijk/explainerdashboard/workflows/explainerdashboard/badge.svg
- https://github.com/marketplace/actions/coveralls-python
- submit pull request to dtreeviz to accept shadowtree as parameter