
# TO BE DONE:

Layout:
- add option to drop na's from interaction plot
- add option to drop na's from color coding dependence plot (or make them grey?)
- Add percentile of scores to contributions tab
- make background graphs white
- add violin plot to categorical plots

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
- to_shap_sql method


Library level:
- Add tests
- turn docstrings into sphynx compatible
- Turn into proper package: 
    - register on pypi

