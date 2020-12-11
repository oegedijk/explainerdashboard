
# TODO:

## Bugs:
- ContributionTable with logodds multiplies by 100!
- dash contributions reload bug: Exception: Additivity check failed in TreeExplainer!

## Layout:
- Find a proper frontender to help :)

## dfs:
- wrap shap values in pd.DataFrames?
- wrap predictions in pd.Series?

## Plots:
- make plot background transparent?
- Only use ScatterGl above a certain cutoff
- seperate standard shap plots for shap_interaction plots 
    - using some kind of inheritance?
- change lines and annotation to this:
    - https://community.plotly.com/t/announcing-plotly-py-4-12-horizontal-and-vertical-lines-and-rectangles/46783
- add some of these:
    https://towardsdatascience.com/introducing-shap-decision-plots-52ed3b4a1cba


### Classifier plots:
- add wizard to lift curve
- pdp: add multiclass option
    - no icelines to keep it from getting too busy?

### Regression plots:


## Explainers:
- add plain language explanations
    - could add an parameter to the` explainer.plot_*` function  `in_words=True` in which 
        case instead of a plot the function returns a verbal description of the 
        relationship in the plot.
    - Then add an "in words" button to the components, that show a popup with
        the verbal explanation.
- rename RandomForestExplainer and XGBExplainer methods into something more logical
    - Breaking change!

## notebooks:


## Dashboard:

- add waitress to run options
- use bootstrap tabs for better theming (especially with dark themes)?
- organize explainer components according to tab
- Add EDA style feature histograms, bar charts, correlation graphs, etc
- add cost calculator/optimizer for classifier models based on confusion matrix weights
    - add Youden J's calculation
- add group fairness metrics
    - https://arxiv.org/pdf/1910.05591.pdf
    - https://cran.r-project.org/web/packages/fairmodels/vignettes/Basic_tutorial.html
    - http://manifold.mlvis.io/
        - generate groups programmatically!

### Components
- Make prediction summary work with FeatureInputComponent
- add querystring method to ExplainerComponents
- add pos_label_name property to PosLabelConnector search
- add "number of indexes" indicator to RandomIndexComponents for current restrictions
- set equivalent_col when toggling cats in dependence/interactions
- Add side-by-side option to cutoff selector component
- add width/height to components
- whatif:
    - add n_columns option to FeatureInputComponent
    - Add a constraints function to whatif component:
        - tests if current feature input is allowed
        - gives specific feedback when constraint broken
        - could build WhatIfComponentException for this?
    - Add sliders option to what if component


## Methods:
- add support for SamplingExplainer, PartitionExplainer, PermutationExplainer, AdditiveExplainer
- add support for LimeTabularExplainer:
    - http://gael-varoquaux.info/interpreting_ml_tuto/content/02_why/04_black_box_interpretation.html
    - https://shap.readthedocs.io/en/latest/generated/shap.explainers.other.LimeTabular.html
- Add this method? : https://arxiv.org/abs/2006.04750?

## Tests:
- test model_output='probability' and 'raw' or 'logodds' seperately
- write tests for explainer_methods
- write tests for explainer_plots

## Docs:
- add section to README on storing and loading explainer/dashboard from file/config
- document hiding components in composites
- retake screenshots of components as cards
- rerecord gifs
- Add type hints:
    - to explainers
    - to explainer class methods
    - to explainer_methods
    - to explainer_plots
- Add pydata video when it comes online (january 4th)


## Library level:
- add waitress to CLI

- hide (prefix '_') to non-public API class methods
- submit pull request to shap with broken test for 
    https://github.com/slundberg/shap/issues/723
- build explainerhub for hosting multiple explainerdashboard models
    - use waitress as wsgi
    - use watchdog to rebuild and reload. 
