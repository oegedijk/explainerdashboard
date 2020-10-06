
# TODO:

## Layout:
- Find a proper frontender to help :)

## dfs:
- wrap shap values in pd.DataFrames?
- wrap predictions in pd.Series?

## Plots:
- add X_row parameter to plot_shap_contribution
- add X_row parameter to plot_pdp

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

## notebooks:

## Dashboard:
- Add EDA style feature histograms, bar charts, correlation graphs, etc
- add cost calculator/optimizer for classifier models based on confusion matrix weights
- add group fairness metrics? 

### Components
- add pos_label_name property to PosLabelConnector search
- add "number of indexes" indicator to RandomIndexComponents for current restrictions
- classifier prediction summary: no logodds.

## Methods:
- Add LIME values
    - but tricky how to set kernel, model, etc
    - Lime values take a lot more DS judgement than SHAP values
- Add this method? : https://arxiv.org/abs/2006.04750?

## Tests:
- test model_output='probability' and 'raw' or 'logodds' seperately
- write tests for explainer_methods
- write tests for explainer_plots
- add test coverage 

## Docs:
- add whatif docs
- add docstrings to explainer_plots
- add screenshots of components to docs
- move screenshots to separate folder
- document PosLabelSelector and PosLabelConnector, e.g.:
        self.connector = PosLabelConnector(self.roc_auc, self)
        self.register_components(self.connector)

## Library level:
- Add Altair (vega) plots for easy inclusion in websites or fastpages blogs
- Long term: add option to load from directory with pickled model, data csv and config file
- add more screenshots to README with https://postimages.org/
- add test coverage: https://github.com/marketplace/actions/coveralls-python
- submit pull request to shap with broken test for https://github.com/slundberg/shap/issues/723
- install vscode github pull requests and issues extension