# TODO

**Meta**
- Last triage: 2026-02-08
- Owner: @oegedijk
- Rules: link an issue when possible; include size S/M/L; mark blockers.

**Done**
- [S][Methods][#220] get_contrib_df accepts list/array input (and hardened related X_row APIs).
- [M][Components][#176] FeatureInputComponent hide parameters.
- [M][Explainers][#198/#340] LightGBM string categorical handling across SHAP/plots.
- [S][Hub][#146/#342] hub.to_yaml integrate_dashboard_yamls honors pickle_type and dumps integrated explainer artifacts.
- [M][Explainers][#294] align/explain multiclass logodds between Contributions Plot and Prediction Box (+ PDP highlight and XGBoost decision path wording alignment).

**Now**
- [M][Explainers][#118] add LightGBM tree visualization support (dtreeviz).

**Next**
- [M][Dashboard][#263/#161] more flexible instantiate_component (no explainer needed for non-ExplainerComponents).

**Backlog: Explainers**
- [M] add plain language explanations for plots (in_words + UI toggle).
- [S] pass n_jobs to pdp_isolate.
- [M] add ExtraTrees and GradientBoostingClassifier to tree visualizers.
- [M][#118] add LightGBM tree visualization support (dtreeviz).

**Backlog: Dashboard**
- [S] make poweredby right-aligned.
- [M][#263/#161] more flexible instantiate_component (no explainer needed for non-ExplainerComponents).
- [M] add TablePopout.
- [M][#247] add EDA-style feature histograms/bar charts/correlation graphs.
- [M/L] add cost calculator/optimizer for classifier models (confusion matrix weights, Youden J).
- [M/L] add group fairness metrics (see refs in old TODO).

**Backlog: Hub**
- [M] automatic reloads with watchdog.
- [S] expose reloader/debug options in run().
- [M][#306] support selecting between multiple explainers from one hub/dashboard.

**Backlog: Components**
- [M][#165/#233] add predictions list to whatif composite.
- [S] add circular callbacks between cutoff and cutoff percentile.
- [S] add side-by-side option to cutoff selector component.
- [M][#249] add filter to index selector using pattern-matching callbacks.
- [S] add pos_label_name property to PosLabelConnector search.
- [S] add "number of indexes" indicator to RandomIndexComponents for current restrictions.
- [M] whatif constraints function with validation feedback.
- [M] add sliders option to whatif component.
- [S][#170] allow limiting What-If input editor to selected feature subset.
- [M][#171] allow optional contextual/image display in Individual Predictions.

**Backlog: Methods**
- [M] support SamplingExplainer, PartitionExplainer, PermutationExplainer, AdditiveExplainer.
- [M] support LimeTabularExplainer.
- [M] investigate method from https://arxiv.org/abs/2006.04750.
- [M][#256] improve multiclass LinearSVC support/docs (class-count mismatch with SHAP output).
- [M][#229] clarify/add support path for Poisson and Gamma regression explainers.

**Backlog: Plots**
- [S] add hide_legend parameter.
- [M] add SHAP decision plots (ref: https://towardsdatascience.com/introducing-shap-decision-plots-52ed3b4a1cba).
- [S] make plot background transparent (configurable).
- [S] use ScatterGL only above a point-count cutoff.
- [M] separate standard shap plots vs shap_interaction plots (inheritance or new class).
- [S] update lines/annotations to new Plotly helpers (ref: https://community.plotly.com/t/announcing-plotly-py-4-12-horizontal-and-vertical-lines-and-rectangles/46783).
- [M] PDP multiclass support end-to-end (data + plots + UI).

**Backlog: Tests**
- [S] add pipeline with X_background test.
- [S] test explainer.dump/explainer.from_file with .pkl or .dill.
- [S] add get_descriptions_df tests (including sort='shap').
- [S] add set_shap_values test.
- [S] add set_shap_interaction_values test.
- [S] add get_idx_sample tests.
- [S] add y_binary with self.y_missing tests.
- [S] add percentile_from_cutoff tests.
- [S] add tests for InterpretML EBM (shap 0.37).
- [S] add tests for explainerhub CLI add user.
- [S] test model_output='probability' vs 'raw' vs 'logodds' explicitly.
- [M] expand explainer_methods tests.
- [M] add explainer_plots tests.

**Backlog: Docs**
- [S] retake screenshots of components as cards.
- [M] add type hints to explainer class methods, explainer_methods, explainer_plots.
- [S][#280] add guidance on `X` and `X_background` sizing/performance trade-offs.

**Backlog: Library/Infra**
- [S] example deployment repo (Heroku or alternative).
- [S] example ExplainerHub deployment repo.
- [S][#284] publish a public showcase deployment (e.g. Hugging Face Spaces).

**External / Upstream**
- [M] submit SHAP PR with broken test for https://github.com/slundberg/shap/issues/723.
