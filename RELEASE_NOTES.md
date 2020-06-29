# Release Notes

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