
# TODO:
- model summary markdown
- metrics (classifier + regression)
- cutoff_fraction -> rename cutoff_from_percentile
- shadowtree_df_summary only return df
- plot_shap_importances()
- plot_shap_summary() ?
- rename columns_ranked
- rename shadow trees -> individual trees
- rename plotly_ from all plotting functions


## Layout:

## Plots:
- add multiclass confusion matrix
- individual trees: highlight selected tree

### Regression plots:

## Explainers:

## Dashboard:
- add dependence plot to importances list
- Contributions: add div margin to index selector
- move number of features to display
- add option for vertical contributions?
- reformat contributions table

## Methods:

- Multiprocessing shap summary graph 
- Move pdp function to explainer_methods.py
- add feature explanations


## Library level:
- fix forever updating bug (seems shadow tree related?)
- fix jupyter reload pdp bug
- submit pull request to dtreeviz to accept shadowtree as parameter
- just add shap='tree', 'linear', 'deep', etc instead of separate classes
- add long description to pypi: https://packaging.python.org/guides/making-a-pypi-friendly-readme/
- Add tests
- Test with lightgbm, catboost, extratrees
- turn all docstrings into sphynx compatible
- Fix numpy mock import in readthedocs compile

