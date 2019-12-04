
# TODO:

## Layout:
- Add percentile of scores to contributions tab
- add violin plot to categorical plots
- jitter categorical plots
- automatic load graphs upon page load  (dash > 1.4 bug?)
- Change pos label selector to regular dropdown
    - also give default value

## Plots:

### Classifier:
- Add lift curve

### Regression plots:
    - https://medium.com/microsoftazure/how-to-better-evaluate-the-goodness-of-fit-of-regressions-990dbf1c0091
    - predicted vs actual (should be on 45 deg line)
    - residuals vs prediction plot (heteroskedastic?)
    - residuals vs feature
    - residuals histogram
    - QQ plot
    - MAE, RMSE, RAE, RSE, R2

## Explainers:
- Add LinearExplainerBunch, DeepExplainerBunch

## Dashboard:
- Make flexible dashboard class with (layout, callback) tuples -TBD

## Methods:

- Multiprocessing shap summary graph -TBD
- Move pdp function to explainer_methods.py


## Library level:
- Add tests
- Test with lightgbm, catboost, extratrees
- turn all docstrings into sphynx compatible
- Fix numpy mock import in readthedocs compile

