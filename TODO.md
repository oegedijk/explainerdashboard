
# TODO:

## Version 0.3:
- check all register_dependencies()
- check InlineExplainer 

## Bugs:


## Layout:
- Find a proper frontender to help :)

## dfs:

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
- move predicted and actual to outer layer of ConfusionMatrixComponent
    - move predicted below graph?
- pdp: add multiclass option
    - no icelines just mean and index with different thickness
    - new method?

### Regression plots:


## Explainers:
- pass n_jobs to pdp_isolate
- add ExtraTrees and GradientBoostingClassifier to tree visualizers
- add plain language explanations
    - could add an parameter to the` explainer.plot_*` function  `in_words=True` in which 
        case instead of a plot the function returns a verbal description of the 
        relationship in the plot.
    - Then add an "in words" button to the components, that show a popup with
        the verbal explanation.

## notebooks:


## Dashboard:
- add SimpleClassifierDashboard
- add SimpleRegressionDashboard
- Add EDA style feature histograms, bar charts, correlation graphs, etc
- add cost calculator/optimizer for classifier models based on confusion matrix weights
    - add Youden J's calculation
- add group fairness metrics
    - https://arxiv.org/pdf/1910.05591.pdf
    - https://cran.r-project.org/web/packages/fairmodels/vignettes/Basic_tutorial.html
    - http://manifold.mlvis.io/
        - generate groups programmatically!

## Hub:
- automatic reloads with watchdog
- add reloader=None, debug=None, options
- make example deployment on heroku


### Components
- autodetect when uuid name get rendered and issue warning

- Add side-by-side option to cutoff selector component
- add filter to index selector using pattern matching callbacks:
    - https://dash.plotly.com/pattern-matching-callbacks
- add querystring method to ExplainerComponents
- add pos_label_name property to PosLabelConnector search
- add "number of indexes" indicator to RandomIndexComponents for current restrictions
- set equivalent_col when toggling cats in dependence/interactions

- add width/height to components
- whatif:
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
- add tests for InterpretML EBM (shap 0.37)
- write tests for explainerhub CLI add user
- test model_output='probability' and 'raw' or 'logodds' seperately
- write tests for explainer_methods
- write tests for explainer_plots

## Docs:
- add memory savings to docs/README:
    - memory_usage()
    - keep_shap_pos_label_only()
    - set_X_row_func, etc
- add cats_topx cats_sort to docs
- add hide_wizard and wizard to docs
- add hide_poweredby to docs
- add Docker deploy example (from issue)
- document register_components no longer necessary
- add new whatif parameters to README and docs
- add section to docs and README on storing and loading explainer/dashboard from file/config

- retake screenshots of components as cards
- Add type hints:
    - to explainers
    - to explainer class methods
    - to explainer_methods
    - to explainer_plots
- Add pydata video when it comes online (january 4th)


## Library level:
- Make example heroku deployment repo
- Make example heroku ExplainerHub repo
- submit pull request to shap with broken test for 
    https://github.com/slundberg/shap/issues/723

