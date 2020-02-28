
# TODO:
- fix hit enter on input field reloads page

## Layout:

## dfs:


## Plots:
- Classification plot: add percentage in totals plot, add totals in percentage plot
- Add feature names to waterfall plot
- fix percentages difference bug lift plot vs classification plot
- figure out why roc_auc, pr_auc, confusion plots don't scale (probably to do with being forced square?)
- rename plotly_ from all plotting functions?
- Add Altair (vega) plots for easy inclusion in websites

### Regression plots:

## Explainers:


## Dashboard:
- add option for vertical contributions?
- reformat contributions table
- add final prediction to contributions table

## Methods:
- Multiprocessing shap summary graph 
- Move pdp function to explainer_methods.py
- add feature explanations (i.e. explanation of what feature is doing)
- Add LIME, Shaabas values for completeness?

## Library level:
- submit pull request to dtreeviz to accept shadowtree as parameter
- just add shap='tree', 'linear', 'deep', etc instead of separate classes
- add long description to pypi: https://packaging.python.org/guides/making-a-pypi-friendly-readme/
- Add more tests!
- Test with lightgbm, catboost, extratrees
- turn all docstrings into sphynx compatible
- Fix numpy mock import in readthedocs compile

