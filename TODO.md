
# TODO:
- try to get dtreeviz to work for regression with only test data
    - seems tricky as DecisionTree does not store individual samples...
- find a way to plot individual xgboost trees (now in dtreeviz!)

## Layout:
- Find a proper frontender to help :)

## dfs:
- wrap shap values in pd.DataFrames?
- wrap predictions in pd.Series?

## Plots:


### Classifier plots:
- confusion matrix: display both count and percentage
- pdp: add multiclass option
- add classification model summary
- include cumulative lift curve to standard dashboard
- add cost calculator: cost of FPs and FNs
- classification plot:
    - round percentage to 1 digit
    - add `<br>` between count and percentage

### Regression plots:

## Explainers:
- add target name 
- add plain language explanations


## notebooks:
- add binder/colab links on github

## Dashboard:
- open dtreeviz in seperate tab
- Add pandas profiling type col histograms, bar charts, correlation graphs, etc


### Components
- add pos_label_name property to PosLabelConnector search
- add "number of indexes" indicator to RandomIndexComponents for current restrictions

## Methods:
- Add LIME values
    - but tricky how to set kernel, model, etc
    - Lime values take a lot more DS judgement than SHAP values
- Add this method? : https://arxiv.org/abs/2006.04750?

## Tests:
- test model_output='probability' and 'raw' or 'logodds' seperately
- add test coverage (add a badge)

## Docs:
- add docstrings to explainer_plots
- add screenshots of components to docs
- move screenshots to separate folder
- document PosLabelSelector and PosLabelConnector, e.g.:
        self.connector = PosLabelConnector(self.roc_auc, self)
        self.register_components(self.connector)


## Library level:
- Add launch from colab option:
    - https://amitness.com/2020/06/google-colaboratory-tips/?s=03
- Add Altair (vega) plots for easy inclusion in websites or fastpages blogs
- Long term: add option to load from directory with pickled model, data csv and config file
- add more screenshots to README with https://postimages.org/
- add test coverage: https://github.com/marketplace/actions/coveralls-python
- submit pull request to shap with broken test for https://github.com/slundberg/shap/issues/723