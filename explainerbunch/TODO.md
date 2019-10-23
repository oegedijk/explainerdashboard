
# TO BE DONE:
Bugs:
- contribution plot =?
Layout:
- add actual roc auc score to plot
- add actual pr auc score to plot
- add option to drop na's from interaction plot
- add option to drop na's from color coding dependence plot (or make them grey?)
- Add percentile of scores to contributions tab
- Add quantiles to precision plot - TBD
- remove -999 from pdpplot

Methods:
- store layout settings and use for callbacks - TBD
    - add assert to make sure layout defined - TBD
- Make flexible dashboard class with (layout, callback) tuples -TBD
- Multiprocessing shap summary graph -TBD
- Move pdp function to explainer_methods.py
- to_shap_sql method
- ShapPlots method with only plot_mean_abs_shap


Library level:
- Add tests
- Turn into proper module: 
    - setuptools
    - register on pypi

# DONE:

- Undo monkeypatching. Change to simple functions... check

Reorganize classes - check 
- Incorporate layout into class - check 
- Incorporate callbacks into class?? - check 
- Make init independent of raw data/transformer - check

- Make tabs independent - check
    - Link between contributions and tree tab - check
- Use either int or str indices (parameter of layout?) check
- refactor dashboard into own class - check
- Get shap ordering without calculating shap interaction values - check