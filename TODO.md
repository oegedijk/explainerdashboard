
# TODO:
- put idxs in df.index

## Layout:
- Find a proper frontender to help :)

## dfs:
- wrap shap values in pd.DataFrames?
- wrap predictions in pd.Series?

## Plots:
- add some of these:
    https://towardsdatascience.com/introducing-shap-decision-plots-52ed3b4a1cba

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
- add "experiment tracker" for what if...
- add pos_label_name property to PosLabelConnector search
- add "number of indexes" indicator to RandomIndexComponents for current restrictions

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
- add CLI documentation
- deployment: show how to to make automatically rebootable dashboard 
    with watchdog package
    - watchmedo shell-command  -p "*model.pkl;*data.csv;*explainerdashboard.yaml" -c "explainerdashboard build"
    
- add docstrings to explainer_plots
- add screenshots of all ExplainerComponents to docs
- move screenshots to separate folder
- document PosLabelSelector and PosLabelConnector, e.g.:
        self.connector = PosLabelConnector(self.roc_auc, self)
        self.register_components(self.connector)

## Library level:
- Add Altair (vega) plots for easy inclusion in websites or fastpages blogs
- add more screenshots to README with https://postimages.org/
- submit pull request to shap with broken test for https://github.com/slundberg/shap/issues/723