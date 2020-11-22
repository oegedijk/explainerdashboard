
# TODO:

## Layout:
- Find a proper frontender to help :)

## dfs:
- wrap shap values in pd.DataFrames?
- wrap predictions in pd.Series?

## Plots:
- add some of these:
    https://towardsdatascience.com/introducing-shap-decision-plots-52ed3b4a1cba
- seperate standard shap plots for shap_interaction plots 
    - using some kind of inheritance?

### Classifier plots:
- pdp: add multiclass option
- add label percentage at cutoff to cumulative precision plot
- add wizard to lift curve

### Regression plots:
- add linear trendlines to y/preds vs col plots:
    - https://stackoverflow.com/questions/58708230/plotly-how-to-plot-a-regression-line-using-plotly


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
- organize explainer components according to tab
- Add EDA style feature histograms, bar charts, correlation graphs, etc
- add cost calculator/optimizer for classifier models based on confusion matrix weights
- add group fairness metrics
    - https://arxiv.org/pdf/1910.05591.pdf
    - https://cran.r-project.org/web/packages/fairmodels/vignettes/Basic_tutorial.html
    - http://manifold.mlvis.io/
        - generate groups programmatically!
- make terminate a class method
- add description param to all components
- add when warning when kwargs is not empty that will be passed down to all components


### Components
- rename to component_callbacks()
- add Tooltips to whatif component
- confusion matrix component: only show cutoff if binary==True

- add querystring method to ExplainerComponents
- add "experiment tracker" for what if...
- add pos_label_name property to PosLabelConnector search
- add "number of indexes" indicator to RandomIndexComponents for current restrictions
- set equivalent_col when toggling cats in dependence/interactions
- Add a constraints function to whatif component:
    - tests if current feature input is allowed
    - gives specific feedback when constraint broken
    - could build WhatIfComponentException for this?
- Add side-by-side option to cutoff selector component
- Add sliders option to what if component
- unify ClassifierMetricsSummaryComponent and RegressionMetricsSummaryComponent 
    to a single component
- add a barchart/piechart to classifier prediction summary component

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
- add regressionVsCol screenshot
- Add type hints:
    - to explainers
    - to explainer class methods
    - to explainer_methods
    - to explainer_plots
- Add pydata video: https://www.youtube.com/watch?v=1nMlfrDvwc8
- document PosLabelSelector and PosLabelConnector, e.g.:
        self.connector = PosLabelConnector(self.roc_auc, self)
        self.register_components(self.connector)
- add actual_vs_col and pred_vs_col screenshots to components docs

## Library level:
- hide (prefix '_') to non-public API class methods
- build release for conda-forge
    - get dash-auth on plotly anaconda channel
    - get dtreeviz on anaconda
- launch gunicorn server from python:
    https://damianzaremba.co.uk/2012/08/running-a-wsgi-app-via-gunicorn-from-python/
- Add Altair (vega) plots for easy inclusion in websites or fastpages blogs
- submit pull request to shap with broken test for https://github.com/slundberg/shap/issues/723
- build explainerhub for hosting multiple explainerdashboard models
    - using django?