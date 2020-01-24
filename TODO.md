
# TODO:

## Layout:
- set all tabs default to False

## Plots:
- add multiclass confusion matrix
- individual trees: highlight selected tree
- fix shap dependence summary going full width

### Regression plots:

## Explainers:

## Dashboard:
- add dependence plot to importances list

## Methods:

- Multiprocessing shap summary graph 
- Move pdp function to explainer_methods.py
- add feature explanations

## Library level:
- fix forever updating bug (seems shadow tree related?)
- fix jupyter reload pdp bug
- just add kind='tree', 'linear', 'deep', etc
- add long description to pypi: https://packaging.python.org/guides/making-a-pypi-friendly-readme/
- Add tests
- Test with lightgbm, catboost, extratrees
- turn all docstrings into sphynx compatible
- Fix numpy mock import in readthedocs compile

