
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
    - using some kind of inheritance
- DecisionTree plots: 
    - add some seperatation when pred ~= actual
    - rename y "observed" 
    - add bars for average and observed?

### Classifier plots:
- pdp: add multiclass option
- add cutoff to cumulative precision plot

### Regression plots:
- add actual vs feature
- add predicted vs feature

## Explainers:
- add plain language explanations
- rename RandomForestExplainer and XGBExplainer methods into something more logical
    - Breaking change!
- Allow sklearn pipelines as model input
- clearer error message for shap guess fail
- add `cats = dict(Sex=['Male', 'Female'])` option.


## notebooks:

## Dashboard:
- Add EDA style feature histograms, bar charts, correlation graphs, etc
- add cost calculator/optimizer for classifier models based on confusion matrix weights
- add group fairness metrics

### Components
- add "experiment tracker" for what if...
- add pos_label_name property to PosLabelConnector search
- add "number of indexes" indicator to RandomIndexComponents for current restrictions
- whatif component: check non duplicate feature names

## Methods:
- Add LIME values
    - but tricky how to set kernel, model, etc
    - Lime values take a lot more DS judgement than SHAP values
- Add this method? : https://arxiv.org/abs/2006.04750?

## Tests:
- add yaml->cli integration tests
- add tests for explainer.dump() and explainer.from_file()
- test model_output='probability' and 'raw' or 'logodds' seperately
- write tests for explainer_methods
- write tests for explainer_plots

## Docs:
- Add pydate video: https://www.youtube.com/watch?v=1nMlfrDvwc8
- document PosLabelSelector and PosLabelConnector, e.g.:
        self.connector = PosLabelConnector(self.roc_auc, self)
        self.register_components(self.connector)

## Library level:
- build release on conda-forge
- launch gunicorn server from python:
    https://damianzaremba.co.uk/2012/08/running-a-wsgi-app-via-gunicorn-from-python/
- Add Altair (vega) plots for easy inclusion in websites or fastpages blogs
- submit pull request to shap with broken test for https://github.com/slundberg/shap/issues/723