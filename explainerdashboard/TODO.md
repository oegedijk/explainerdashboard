
# TO BE DONE:

Layout:
- Add percentile of scores to contributions tab
- add violin plot to categorical plots
- jitter categorical plots
- add random index to tree plot
- automatic load model_summary (shap 1.4 bug?)
- highlight idx in dependendence plot

- Regression tab:
    - https://medium.com/microsoftazure/how-to-better-evaluate-the-goodness-of-fit-of-regressions-990dbf1c0091
    - predicted vs actual (should be on 45 deg line)
    - residuals vs prediction plot (heteroskedastic?)
    - residuals vs feature
    - residuals histogram
    - QQ plot
    - MAE, RMSE, RAE, RSE, R2

Methods:
- Make flexible dashboard class with (layout, callback) tuples -TBD
- Multiprocessing shap summary graph -TBD
- Move pdp function to explainer_methods.py

Library level:
- Add tests
- turn docstrings into sphynx compatible
- Turn into proper package: 
    - register on pypi

