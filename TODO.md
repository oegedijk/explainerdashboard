
# TODO:
- fix hit enter on input field reloads page on decision tree tab
- Interaction feature stuck on one feature on heroku?
- find out why dtreeviz for regression no longer working
- find a way to plot individual xgboost trees
- replace custom permutation importances by sklearn permuation importances
- add non-shap dependent len(cols) property

## Layout:
- Find a proper frontender to help :)

## dfs:
- wrap shap values in pd.DataFrames?
- wrap predictions in pd.Series?

## Plots:
- Add options sorting contributions plot from most negative to most positive
- fix name of figure MEAN_ABS_SHAP
- Add Altair (vega) plots for easy inclusion in websites
- rename plotly_ from all plotting functions? or not if we're going to add altair?
- highlight id in violin plots

### Classifier plots:
- Add feature names to waterfall plot
- remove group cats option when cats is empty (not just None)
- confusion matrix: display both count and percentage


### Regression plots:
- regression plots: only take log of x-axis or y-axis
- add cats option (violin plots?) to plot_residuals_vs_feature
- add log ratio

## Explainers:
- add get_random_index with min/max residual for regression

## Dashboard:
- Make all DashboardTabs derive from ABC BaseExplainerTab
- Try to make a class decorator for title_and_label_selector?
- make alternative tight layout for mode='inline' 
- add option for vertical contributions?
- reformat contributions table
- add final prediction to contributions table
- font size on index input larger

### Components
- add hide_title to all components

## Methods:
- Multiprocessing shap summary graph 
- Move pdp function to explainer_methods.py
- Add LIME, Shaabas values for completeness?
- refactor for loop in contrib_df_summary

## Tests:
- add multiclass classifier tests
- add dashboard integration tests using dash tests
- add ExplainerDashboard intergration test
- Add tests for decisiontrees, extratrees
- test model_output='probability' and 'raw' or 'logodds' seperately
- add test coverage (add a badge)

# Docs:
- document X_background
- document properties with prop(pos_label)
- document model_output
- document JupyterExplainerDashboard, InlineExplainer
- add deploying with flask/gunicorn section
- convert to MyST for markdown friendly documentation?

## Library level:
- add badges to README: https://github.com/badges/shields
-> https://github.com/oegedijk/explainerdashboard/workflows/explainerdashboard/badge.svg
- https://github.com/marketplace/actions/coveralls-python
- submit pull request to dtreeviz to accept shadowtree as parameter
- turn all docstrings into sphinx-napolean google style (pyment)

