# Release Notes

## Version 0.2.11:
### Breaking Changes
- 
- 

### New Features
-   Added tooltips everywhere throughout the dashboard to explainer the
    components, plots, dropdowns and toggles of the dashboard itself.
-

### Bug Fixes
-
-

### Improvements
- changed colors on contributions graph up=green, down=red
    - added `higher_is_better` parameter to toggle the colors.
- Clarified wording on index selector components
- hiding `group cats` toggle everywhere when no cats are passed
- passing `**kwargs` of ExplainerDashbaord down to all all tabs and (sub) components
    so that you can configure components from an ExplainerDashboard param. 
    e.g. `ExplainerDashboard(explainer, higher_is_better=False).run()` will
    pass the higher_is_better param down to all components. In the case of the
    ShapContributionsGraphComponent and the XGBoostDecisionTrees component
    this will cause the red and green colors to flip (normally green is up
    and red is down.)

### Other Changes
-
-

## Version 0.2.11:
### Breaking Changes
- 
- 

### New Features
- added (very limited) sklearn.Pipeline support. You can pass a Pipeline as
    `model` parameter as long as the pipeline either:
    1. Does not add, remove or reorders any input columns
    2. has a .get_feature_names() method that returns the new column names
        (this is currently beings debated in sklearn SLEP007)
- added cutoff slider to CumulativePrecisionComponent
- For RegressionExplainer added ActualVsColComponent and PredsVsColComponent
    in order to investigate partial correlations between y/preds and 
    various features. 
- added `index_name` parameter: name of the index column (defaults to `X.index.name`
    or `idxs.name`). So when you pass `index_name="Passenger"`, you get
    a "Random Passenger" button on the index selector instead of "Random Index",
    etc.

### Bug Fixes
- Fixed a number of bugs for when no labels are passed (`y=None`):
    - fixing explainer.random_index() for when y is missing
    - Hiding label/y/residuals selector in RandomIndexSelectors
    - Hiding y/residuals in prediction summary
    - Hiding model_summary tab
    - Removing permutation importances from dashboard


### Improvements
- Seperated labels for "observed" and "average prediction" better in tree plot
- Renamed "actual" to "observed" in prediction summary
- added unique column check for whatif-component with clearer error message
- model metrics now formatted in a nice table
- removed most of the loading spinners as most graphs are not long loads anyway.

### Other Changes
-
-

## Version 0.2.10:

### New Features
- Explainer parameter `cats` now takes dicts as well where you can specify
    your own groups of onehotencoded columns.
        - e.g. instead of passing `cats=['Sex']` to group `['Sex_female', 'Sex_male', 'Sex_nan']`
        you can now do this explicitly: `cats={'Gender'=['Sex_female', 'Sex_male', 'Sex_nan']}`
        - Or combine the two methods: 
            `cats=[{'Gender'=['Sex_female', 'Sex_male', 'Sex_nan']}, 'Deck', 'Embarked']`


## Version 0.2.9:
### Breaking Changes


### New Features
- You don't have to pass the list of subcomponents in `self.register_components()`
    anymore: it will infer them automatically from `self.__dict__`.

### Improvements
-   ExplainerComponents now automatically stores all parameters to attributes
-   ExplainerComponents now automatically stores all parameters to a ._stored_params dict
-   ExplainerDashboard.to_yaml() now support instantiated tabs and stores parameters to yaml
-   ExplainerDashboard.to_yaml() now stores the import requirments of subcomponents
-   ExplainerDashboard.from_config() now instantiates tabs with stored parameters
-   ExplainerDashboard.from_config() now imports classes of subcomponents

### Other Changes
-   added docstrings to explainer_plots
-   added screenshots of ExplainerComponents to docs
-   added more gifs to the documentation

## Version 0.2.8:
### Breaking Changes
- split explainerdashboard.yaml into a explainer.yaml and dashboard.yaml
- Changed UI of the explainerdashboard CLI to reflect this
- This will make it easier in the future to have automatic rebuilds and redeploys
    when an modelfile, datafile or configuration file changes.

### New Features
-   Load an ExplainerDashboard from a configuration file with the classmethod, 
    e.g. : `ExplainerDashboard.from_config("dashboard.yaml")`
-

### Bug Fixes
-
-

### Improvements
-
-

### Other Changes
-
-

## Version 0.2.7:
### Breaking Changes
- 
- 

### New Features
-   explainer.dump() to store explainer, explainer.from_file() to load 
    explainer from file
-   Explainer.to_yaml() and ExplainerDashboard.to_yaml() can store the 
    configuration of your explainer/dashboard to file.
-   explainerdashboard CLI:
    - Start an explainerdashboard from the command-line!
    - start default dashboard from stored explainer : `explainerdashboard run explainer.joblib`
    - start full configured dashboard from config: `explainerdashboard run explainerdashboard.yaml`
    - build explainer based on input files defined in .yaml 
        (model.pkl, data.csv, etc): `explainerdashboard build explainerdashboard.yaml`
    - includes new ascii logo :)

### Bug Fixes
-
-

### Improvements
-   If idxs is not passed use X.index instead
-   explainer.idxs performance enhancements
-   added whatif component and tab to InlineExplainer
-   added cumulative precision component to InlineExplainer

### Other Changes
-
-


Version 0.2.6:

### Improvements
-   more straightforward imports: `from explainerdashboard import ClassifierExplainer, RegressionExplainer, ExplainerDashboard, InlineExplainer`
-   all custom imports (such as ExplainerComponents, Composites, Tabs, etc) 
    combined under `explainerdashboard.custom`:
    `from explainerdashboard.custom import *`

## version 0.2.5:
### Breaking Changes
- 
- 

### New Features
-   New dashboard tab: WhatIfComponent/WhatIfComposite/WhatIfTab: allows you
        to explore whatif scenario's by editing multiple featues and observing
        shap contributions and pdp plots. Switch off with ExplainerDashboard
        parameter whatif=False.
-   New login functionality: you can restrict access to your dashboard by passing
        a list of `[login, password]` pairs:
        `ExplainerDashboard(explainer, logins=[['login1', 'password1'], ['login2', 'password2']]).run()`
-   Added 'target' parameter to explainer, to make more descriptive plots.
        e.g. by setting target='Fare', will show 'Predicted Fare' instead of 
        simply 'Prediction' in various plots.
-   in detailed shap/interaction summary plots, can now click on single 
    shap value for a particular feature, and have that index highlighted
    for all features.
-   autodetecting Google colab environment and setting mode='external' 
    (and suggesting so for jupyter notebook environments)     
-   confusion matrix now showing both percentage and counts
-   Added classifier model performance summary component
-   Added cumulative precision component


### Bug Fixes
-
-

### Improvements
-   added documentation on how to deploy to heroku
-   Cleaned up modebars for figures
-   ClassifierExplainer asserts predict_proba attribute of model
-   with model_output='logodds' still display probability in prediction summary
-   for ClassifierExplainer: check if has predict_proba methods at init

### Other Changes
-   removed monkeypatching shap_explainer note
-

## version 0.2.4

### New Features
- added ExplainerDashboard parameter "responsive" (defaults to True) to make 
    the dashboard layout reponsive on mobile devices. Set it to False when e.g.
    running tests on headless browsers.

### Bug Fixes
-   Fixes bug that made RandomForest and xgboost explainers unpicklable

### Improvements
-   Added tests for picklability of explainers


## Version 0.2.3

### Breaking Changes
- RandomForestClassifierExplainer and RandomForestRegressionExplainer will be 
    deprecated: can now simply use ClassifierExplainer or RegressionExplainer and the
    mixin class will automatically be loaded.
- 

### New Features
- Now also support for visualizing individual trees for XGBoost models!
    (XGBClassifier and XGBRegressor). The XGBExplainer mixin class will be 
    automatically loaded and make decisiontree_df(), decision_path() and plot_trees()
    methods available, Decision Trees tab and components now also work for
    XGBoost models. 
- new parameter n_jobs for calculations that can be parallelized (e.g. permutation importances)
- contrib_df, plot_shap_contributions: can order by global shap feature 
    importance with sort='importance' (as well as 'abs', 'high-to-low' 
     'low-to-high')
- added actual outcome to plot_trees (for both RandomForest and XGB)

### Bug Fixes
-
-

### Improvements
- optimized code for calculating permutation importance, adding possibility to calculate in parallel
- shap dependence component: if no color col selected, output standard blue dots instead of ignoring update

### Other Changes
- added selenium integration tests for dashboards (also working with github actions)
- added tests for multiclass classsification, DecisionTree and ExtraTrees models
- added tests for XGBExplainers
- added proper docstrings to explainer_methods.py

## Version 0.2.2

### Bug Fixes
-   kernel shap bug fixed
-   contrib_df bug with topx fixed
-   fix for shap v0.36: import approximate_interactions from shap.utils instead of shap.common


## Version 0.2.1:
### Breaking Changes
- Removed ExplainerHeader from ExplainerComponents
    - so also removed parameter `header_mode` from ExplainerComponent parameters
    - You can now instead syncronize pos labels across components with a PosLabelSelector
        and PosLabelConnector.
- In regression plots instead of boolean ratio=True/False, 
        you now pass residuals={'difference', 'ratio', 'log-ratio'}
- decisiontree_df_summary renamed to decisiontree_summary_df (in line with contrib_summary_df)

### New Features
- added check all shap values >-1 and <1 for model_output=probability
- added parameter pos_label to all components and ExplainerDashboard to set
        the initial pos label
- added parameter block_selector_callbacks to ExplainerDashboard to block
    the global pos label selector's callbacks. If you already have PosLabelSelectors
    in your layout, this prevents clashes. 
- plot actual vs predicted now supported only logging x axis or only y axis
- residuals plots now support option residuals='log-ratio'
- residuals-vs-col plot now shows violin plot for categorical features
- added sorting option to contributions plot/graph: sort={'abs', 'high-to-low', 'low-to-high'}
- added final prediction to contributions plot

### Bug Fixes
- Interaction connector bug fixed in detailed summary: click didn't work
- pos label was ignored in explainer.plot_pdp()
- Fixed some UX issues with interations components

### Improvements
- All `State['tabs', 'value']` condition have been taken out of callbacks. This
    used to fix some bugs with dash tabs, but seems it works even without, so
    also no need to insert dummy_tabs in `ExplainerHeader`.
- All `ExplainerComponents` now have their own pos label selector, meaning
    that they are now fully self-containted and independent. No global dash
    elements in component callbacks. 
- You can define the layout of ExplainerComponents in a layout() method instead
    of _layout(). Should still define _register_callbacks() to define callbacks
    so that all subcomponents that have been registered will automatically
    get their callbacks registered as well. 
- Added regression `self.units` to prediction summary, shap plots, 
        contributions plots/table, pdp plot and trees plot.
- Clearer title for MEAN_ABS_SHAP importance and summary plots
- replace na_fill value in contributions table by "MISSING"
- add string idxs to shap and interactions summary and dependence plots, 
        including the violing plots
- pdp plot for classification now showing percentages instead of fractions



### Other Changes
-   added hide_title parameter to all components with a title
-   DecisionPathGraphComponent not available for RandomForestRegression models for now.
-   In contributions graph base value now called 'population average' and colored yellow.


## version 0.2:
### Breaking Changes
- InlineExplainer api has been completely redefined
- JupyterExplainerDashboard, ExplainerTab and JupyterExplainerTab have been deprecated



### New Features
- Major rewrite and refactor of the dashboard code, now modularized into ExplainerComponents
    and ExplainerComposites.
- ExplainerComponents can now be individually accessed through InlineExplainer
- All elements of components can now be switched on or off or be given an
    initial value.
- Makes it much, much easier to design own custom dashboards.
- ExplainerDashboard can be passed an arbitrary list of components to 
    display as tabs.

### Better docs:
- Added sections InlineExplainer, ExplainerTabs, ExplainerComponents, 
    CustomDashboards and Deployment
- Added screenshots to documentation.

### Bug Fixes
- fixes residuals y-pred instead of pred-y
-

### Improvements
-   Random Index Selector redesigned
-   Prediction summary redesigned
-   Tables now follow dbc.Table formatting
-   All connections between components now happen through explicit connectors
-   Layout of most components redesigned, with all elements made hideable

### Other Changes
-
-

## Version 0.1.13

### Bug Fixes
- Fixed bug with GradientBoostingClassifier where output format of shap.expected_value
    was not not properly accounted for. 
- 

### Improvements
- Cleaned up standalone label selector code
- Added check for shap base values to be between between 0 and 1 for model_output=='probability' 


## Version 0.1.12

### Breaking Changes
- ExplainerDashboardStandaloneTab is now called ExplainerTab
- 

### New Features

added support for the `jupyter-dash` package for inline dashboard in 
Jupyter notebooks, adding the following dashboard classes:

- `JupyterExplainerDashboard`
- `JupyterExplainerTab`
- `InlineExplainer`

## Template:
### Breaking Changes
- 
- 

### New Features
-
-

### Bug Fixes
-
-

### Improvements
-
-

### Other Changes
-
-