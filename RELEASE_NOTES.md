# Release Notes

## Version 2.2.1

### Breaking Changes
- 
- 

### New Features
-
-

### Bug Fixes
-   kernel shap bug fixed
-   contrib_df bug with topx fixed

### Improvements
-   
-

### Other Changes
-
-


## Version 2.1:
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