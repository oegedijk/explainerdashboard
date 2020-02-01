
# TODO:

## Layout:

## Plots:
- add multiclass confusion matrix
- individual trees: highlight selected tree
- Add feature names to waterfall plot

### Regression plots:

## Explainers:

## Dashboard:
- add option for vertical contributions?
- reformat contributions table
- add final prediction to contributions table

## Methods:

- Multiprocessing shap summary graph 
- Move pdp function to explainer_methods.py
- add feature explanations


## Library level:
- fix jupyter reload pdp bug
- submit pull request to dtreeviz to accept shadowtree as parameter
- just add shap='tree', 'linear', 'deep', etc instead of separate classes
- add long description to pypi: https://packaging.python.org/guides/making-a-pypi-friendly-readme/
- Add more tests!
- Test with lightgbm, catboost, extratrees
- turn all docstrings into sphynx compatible
- Fix numpy mock import in readthedocs compile

