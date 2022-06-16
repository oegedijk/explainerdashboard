
# TODO

- add get_descriptions_df tests
- do multiclass pdp

- experiment with dash_draggable: https://github.com/MehdiChelh/dash-draggable
- add set_shap_values tests
- add hub.to_yaml() dashboard dump option, e.g. 'joblib', 'dill' or 'pkl'
- add how to create `ExplainerComponent` to docs (see closed issue)
## Bugs:

## Plots:
- add hide_legend parameter
- add SHAP decision plots:
    https://towardsdatascience.com/introducing-shap-decision-plots-52ed3b4a1cba
- make plot background transparent?
- Only use ScatterGl above a certain cutoff
- seperate standard shap plots for shap_interaction plots 
    - using some kind of inheritance?
- change lines and annotation to this:
    - https://community.plotly.com/t/announcing-plotly-py-4-12-horizontal-and-vertical-lines-and-rectangles/46783


### Classifier plots:
- pdp: add multiclass option
    - no icelines just mean and index with different thickness
    - new method?

### Regression plots:

## Explainers:
- Turn print statements into logging
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
- Turn print statements into logging
- make poweredby right align
- more flexible instantiate_component:
    - no explainer needed (if explainer component detected, pass otherwise ignore)
- add TablePopout
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
- add to_html option


### Components
- add predictions list to whatif composite:
    - https://github.com/oegedijk/explainerdashboard/issues/85
- add circular callbacks to cutoff - cutoff percentile
- Add side-by-side option to cutoff selector component
- add filter to index selector using pattern matching callbacks:
    - https://dash.plotly.com/pattern-matching-callbacks
- add pos_label_name property to PosLabelConnector search
- add "number of indexes" indicator to RandomIndexComponents for current restrictions
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
- add pipeline with X_background test
- test explainer.dump and explainer.from_file with .pkl or .dill
- add get_descriptions_df tests -> sort='shap'
- set_shap_values test
- set_shap_interaction_values test
- add cv metrics tests
- random_index tests
- get_idx_sample
- y_binary with self.y_missing
- percentile_from_cutoff
- decisiontree
- add tests for InterpretML EBM (shap 0.37)
- write tests for explainerhub CLI add user
- test model_output='probability' and 'raw' or 'logodds' seperately
- write tests for explainer_methods
- write tests for explainer_plots

## Docs:
- retake screenshots of components as cards
- Add type hints:
    - to explainer class methods
    - to explainer_methods
    - to explainer_plots


## Library level:
- Make example heroku deployment repo
- Make example heroku ExplainerHub repo
- submit pull request to shap with broken test for 
    https://github.com/slundberg/shap/issues/723

