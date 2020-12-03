
# TODO:

## Bugs:

## Layout:
- Find a proper frontender to help :)

## dfs:
- wrap shap values in pd.DataFrames?
- wrap predictions in pd.Series?

## Plots:
- seperate standard shap plots for shap_interaction plots 
    - using some kind of inheritance?
- change lines and annotation to this:
    - https://community.plotly.com/t/announcing-plotly-py-4-12-horizontal-and-vertical-lines-and-rectangles/46783
- add some of these:
    https://towardsdatascience.com/introducing-shap-decision-plots-52ed3b4a1cba


### Classifier plots:
- add label percentage at cutoff to cumulative precision plot
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
- give warning to use smaller dataset or less trees
- give warning to dump explainer

- add waitress to run options
- organize explainer components according to tab
- add kwargs to dashboard.to_yaml()
- Add EDA style feature histograms, bar charts, correlation graphs, etc
- add cost calculator/optimizer for classifier models based on confusion matrix weights
    - add Youden J's calculation
- add group fairness metrics
    - https://arxiv.org/pdf/1910.05591.pdf
    - https://cran.r-project.org/web/packages/fairmodels/vignettes/Basic_tutorial.html
    - http://manifold.mlvis.io/
        - generate groups programmatically!
- add description param to all components



### Components
- add hide_subtitle parameters to all components
- add description parameter to all components
- hide show points when feature is not in cats
- change single radioitems to dbc.Checklist switch=True
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
- add test for get_row_from_inputs test
- add prediction_summary_df test
- test model_output='probability' and 'raw' or 'logodds' seperately
- write tests for explainer_methods
- write tests for explainer_plots

## Docs:
- Remove ExplainerTabs
- Remove WhatIfComponent
- Document FeatureInputComponent
- add waitress to deployment examples
- Add type hints:
    - to explainers
    - to explainer class methods
    - to explainer_methods
    - to explainer_plots
- Add pydata video when it comes online (january 4th)
- document PosLabelSelector and PosLabelConnector, e.g.:
        self.connector = PosLabelConnector(self.roc_auc, self)
        self.register_components(self.connector)


## Library level:
- add waitress to CLI
- hide (prefix '_') to non-public API class methods
- build release for conda-forge
    - get dash-auth on plotly anaconda channel
    - get dtreeviz on anaconda
- launch gunicorn server from python:
    https://damianzaremba.co.uk/2012/08/running-a-wsgi-app-via-gunicorn-from-python/
- Add Altair (vega) plots for easy inclusion in websites or fastpages blogs
- submit pull request to shap with broken test for 
    https://github.com/slundberg/shap/issues/723
- build explainerhub for hosting multiple explainerdashboard models
    - using django?