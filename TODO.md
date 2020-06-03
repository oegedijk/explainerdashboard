
# TODO:
- fix shap summary not displaying on page load Shap dependence
    - plus:  fix name of figure MEAN_ABS_SHAP
- fix hit enter on input field reloads page on decision tree tab
- autodetect (guess) shap type based on model
- let plots autodetect if col in cats
- find out why dtreeviz for regression no longer working
- find a way to plot individual xgboost trees


## Layout:
- Find a proper frontender to help :)

## dfs:
- wrap shap values in DataFrames
- wrap predictions in Series

## Plots:
- Add options sorting contributions plot from most negative to most positive
- Classification plot: add percentage in totals plot, add totals in percentage plot
- plot_precision: default no cutoff
- plot_precision: round percentages
- add cats option (violin plots?) to plot_residuals_vs_feature
- regression plots: only take log of x-axis or y-axis
- Add feature names to waterfall plot
- fix percentages difference bug lift plot vs classification plot
- Add Altair (vega) plots for easy inclusion in websites
- rename plotly_ from all plotting functions? or not if we're going to add altair?

### Regression plots:

## Explainers:
- add get_random_index with min/max residual for regression

## Dashboard:
- add option for vertical contributions?
- reformat contributions table
- add final prediction to contributions table

## Methods:
- Multiprocessing shap summary graph 
- Move pdp function to explainer_methods.py
- Add LIME, Shaabas values for completeness?
- refactor for loop in contrib_df_summary

## Tests:
- add multiclass classifier tests
- add individual dashboard tests
- add ExplainerDashboard intergration test
- Add tests for decisiontrees, extratrees
- test model_output='probability' and 'raw' or 'logodds' seperately

# Docs:
- document X_background
- document properties with prop(pos_label)
- document model_output

## Library level:
- submit pull request to dtreeviz to accept shadowtree as parameter
- turn all docstrings into sphynx compatible

