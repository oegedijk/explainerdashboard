__all__ = [
    "ImportancesComposite",
    "ClassifierModelStatsComposite",
    "RegressionModelStatsComposite",
    "IndividualPredictionsComposite",
    "ShapDependenceComposite",
    "ShapInteractionsComposite",
    "DecisionTreesComposite",
    "WhatIfComposite",
    "SimplifiedClassifierComposite",
    "SimplifiedRegressionComposite",
]

import dash_bootstrap_components as dbc
from dash import html

from ..explainers import RandomForestExplainer, XGBExplainer
from ..dashboard_methods import *
from .classifier_components import *
from .regression_components import *
from .overview_components import *
from .connectors import *
from .shap_components import *
from .decisiontree_components import *
from .. import to_html


class ImportancesComposite(ExplainerComponent):
    def __init__(
        self,
        explainer,
        title="Feature Importances",
        name=None,
        hide_title=True,
        hide_importances=False,
        hide_descriptions=False,
        hide_selector=True,
        **kwargs,
    ):
        """Overview tab of feature importances

        Can show both permutation importances and mean absolute shap values.

        Args:
            explainer (Explainer): explainer object constructed with either
                        ClassifierExplainer() or RegressionExplainer()
            title (str, optional): Title of tab or page. Defaults to
                        "Feature Importances".
            name (str, optional): unique name to add to Component elements.
                        If None then random uuid is generated to make sure
                        it's unique. Defaults to None.
            hide_title (bool, optional): hide the title
            hide_importances (bool, optional): hide the ImportancesComponent
            hide_descriptions (bool, optional): hide the FeatureDescriptionsComponent
            hide_selector (bool, optional): hide the post label selector.
                Defaults to True.
        """
        super().__init__(explainer, title, name)

        self.importances = ImportancesComponent(
            explainer, name=self.name + "0", hide_selector=hide_selector, **kwargs
        )
        self.feature_descriptions = FeatureDescriptionsComponent(explainer, **kwargs)

        if not self.explainer.descriptions:
            self.hide_descriptions = True

    def layout(self):
        return dbc.Container(
            [
                make_hideable(
                    dbc.Row(html.H2(self.title), class_name="mt-4"),
                    hide=self.hide_title,
                ),
                dbc.Row(
                    [
                        make_hideable(
                            dbc.Col(
                                [
                                    self.importances.layout(),
                                ]
                            ),
                            hide=self.hide_importances,
                        ),
                    ],
                    class_name="mt-4",
                ),
                dbc.Row(
                    [
                        make_hideable(
                            dbc.Col(
                                [
                                    self.feature_descriptions.layout(),
                                ]
                            ),
                            hide=self.hide_descriptions,
                        ),
                    ],
                    class_name="mt-4",
                ),
            ],
            fluid=True,
        )

    def to_html(self, state_dict=None, add_header=True):
        html = to_html.hide(to_html.title(self.title), hide=self.hide_title)
        html += to_html.card_rows(
            [
                to_html.hide(
                    self.importances.to_html(state_dict, add_header=False),
                    self.hide_importances,
                )
            ],
            [
                to_html.hide(
                    self.feature_descriptions.to_html(state_dict, add_header=False),
                    self.hide_descriptions,
                )
            ],
        )
        if add_header:
            return to_html.add_header(html)
        return html


class ClassifierModelStatsComposite(ExplainerComponent):
    def __init__(
        self,
        explainer,
        title="Classification Stats",
        name=None,
        hide_title=True,
        hide_selector=True,
        hide_globalcutoff=False,
        hide_modelsummary=False,
        hide_confusionmatrix=False,
        hide_precision=False,
        hide_classification=False,
        hide_rocauc=False,
        hide_prauc=False,
        hide_liftcurve=False,
        hide_cumprecision=False,
        pos_label=None,
        bin_size=0.1,
        quantiles=10,
        cutoff=0.5,
        **kwargs,
    ):
        """Composite of multiple classifier related components:
            - precision graph
            - confusion matrix
            - lift curve
            - classification graph
            - roc auc graph
            - pr auc graph

        Args:
            explainer (Explainer): explainer object constructed with either
                        ClassifierExplainer() or RegressionExplainer()
            title (str, optional): Title of tab or page. Defaults to
                        "Decision Trees".
            name (str, optional): unique name to add to Component elements.
                        If None then random uuid is generated to make sure
                        it's unique. Defaults to None.
            hide_title (bool, optional): hide title. Defaults to True.
            hide_selector (bool, optional): hide all pos label selectors. Defaults to True.
            hide_globalcutoff (bool, optional): hide CutoffPercentileComponent
            hide_modelsummary (bool, optional): hide ClassifierModelSummaryComponent
            hide_confusionmatrix (bool, optional): hide ConfusionMatrixComponent
            hide_precision (bool, optional): hide PrecisionComponent
            hide_classification (bool, optional): hide ClassificationComponent
            hide_rocauc (bool, optional): hide RocAucComponent
            hide_prauc (bool, optional): hide PrAucComponent
            hide_liftcurve (bool, optional): hide LiftCurveComponent
            hide_cumprecision (bool, optional): hide CumulativePrecisionComponent
            pos_label ({int, str}, optional): initial pos label. Defaults to explainer.pos_label
            bin_size (float, optional): bin_size for precision plot. Defaults to 0.1.
            quantiles (int, optional): number of quantiles for precision plot. Defaults to 10.
            cutoff (float, optional): initial cutoff. Defaults to 0.5.
        """
        super().__init__(explainer, title, name)

        self.summary = ClassifierModelSummaryComponent(
            explainer, hide_selector=hide_selector, pos_label=pos_label, **kwargs
        )
        self.precision = PrecisionComponent(
            explainer, hide_selector=hide_selector, pos_label=pos_label, **kwargs
        )
        self.confusionmatrix = ConfusionMatrixComponent(
            explainer, hide_selector=hide_selector, pos_label=pos_label, **kwargs
        )
        self.cumulative_precision = CumulativePrecisionComponent(
            explainer, hide_selector=hide_selector, pos_label=pos_label, **kwargs
        )
        self.liftcurve = LiftCurveComponent(
            explainer, hide_selector=hide_selector, pos_label=pos_label, **kwargs
        )
        self.classification = ClassificationComponent(
            explainer, hide_selector=hide_selector, pos_label=pos_label, **kwargs
        )
        self.rocauc = RocAucComponent(
            explainer, hide_selector=hide_selector, pos_label=pos_label, **kwargs
        )
        self.prauc = PrAucComponent(
            explainer, hide_selector=hide_selector, pos_label=pos_label, **kwargs
        )

        self.cutoffpercentile = CutoffPercentileComponent(
            explainer,
            name=self.name + "8",
            hide_selector=hide_selector,
            pos_label=pos_label,
            cutoff=cutoff,
            **kwargs,
        )
        self.cutoffconnector = CutoffConnector(
            self.cutoffpercentile,
            [
                self.summary,
                self.precision,
                self.confusionmatrix,
                self.liftcurve,
                self.cumulative_precision,
                self.classification,
                self.rocauc,
                self.prauc,
            ],
        )

    def layout(self):
        return dbc.Container(
            [
                make_hideable(
                    dbc.Row(html.H2("Model Performance:"), class_name="mt-4 gx-4"),
                    hide=self.hide_title,
                ),
                dbc.Row(
                    [
                        make_hideable(
                            dbc.Col(
                                [
                                    self.cutoffpercentile.layout(),
                                ]
                            ),
                            hide=self.hide_globalcutoff,
                        ),
                    ],
                    class_name="mt-4 gx-4",
                ),
                dbc.Row(
                    [
                        make_hideable(
                            dbc.Col(self.summary.layout()), hide=self.hide_modelsummary
                        ),
                        make_hideable(
                            dbc.Col(self.confusionmatrix.layout()),
                            hide=self.hide_confusionmatrix,
                        ),
                    ],
                    class_name="mt-4 gx-4",
                ),
                dbc.Row(
                    [
                        make_hideable(
                            dbc.Col(self.precision.layout()), hide=self.hide_precision
                        ),
                        make_hideable(
                            dbc.Col(self.classification.layout()),
                            hide=self.hide_classification,
                        ),
                    ],
                    class_name="mt-4 gx-4",
                ),
                dbc.Row(
                    [
                        make_hideable(
                            dbc.Col(self.rocauc.layout()), hide=self.hide_rocauc
                        ),
                        make_hideable(
                            dbc.Col(self.prauc.layout()), hide=self.hide_prauc
                        ),
                    ],
                    class_name="mt-4 gx-4",
                ),
                dbc.Row(
                    [
                        make_hideable(
                            dbc.Col(self.liftcurve.layout()), self.hide_liftcurve
                        ),
                        make_hideable(
                            dbc.Col(self.cumulative_precision.layout()),
                            self.hide_cumprecision,
                        ),
                    ],
                    class_name="mt-4 gx-4",
                ),
            ],
            fluid=True,
        )

    def to_html(self, state_dict=None, add_header=True):
        html = to_html.hide(to_html.title(self.title), hide=self.hide_title)
        html += to_html.card_rows(
            [
                to_html.hide(
                    self.summary.to_html(state_dict, add_header=False),
                    hide=self.hide_modelsummary,
                ),
                to_html.hide(
                    self.confusionmatrix.to_html(state_dict, add_header=False),
                    hide=self.hide_confusionmatrix,
                ),
            ],
            [
                to_html.hide(
                    self.precision.to_html(state_dict, add_header=False),
                    hide=self.hide_precision,
                ),
                to_html.hide(
                    self.classification.to_html(state_dict, add_header=False),
                    hide=self.hide_classification,
                ),
            ],
            [
                to_html.hide(
                    self.rocauc.to_html(state_dict, add_header=False),
                    hide=self.hide_rocauc,
                ),
                to_html.hide(
                    self.prauc.to_html(state_dict, add_header=False),
                    hide=self.hide_prauc,
                ),
            ],
            [
                to_html.hide(
                    self.liftcurve.to_html(state_dict, add_header=False),
                    hide=self.hide_liftcurve,
                ),
                to_html.hide(
                    self.cumulative_precision.to_html(state_dict, add_header=False),
                    hide=self.hide_cumprecision,
                ),
            ],
        )
        if add_header:
            return to_html.add_header(html)
        return html


class RegressionModelStatsComposite(ExplainerComponent):
    def __init__(
        self,
        explainer,
        title="Regression Stats",
        name=None,
        hide_title=True,
        hide_modelsummary=False,
        hide_predsvsactual=False,
        hide_residuals=False,
        hide_regvscol=False,
        logs=False,
        pred_or_actual="vs_pred",
        residuals="difference",
        col=None,
        **kwargs,
    ):
        """Composite for displaying multiple regression related graphs:

        - predictions vs actual plot
        - residual plot
        - residuals vs feature

        Args:
            explainer (Explainer): explainer object constructed with either
                        ClassifierExplainer() or RegressionExplainer()
            title (str, optional): Title of tab or page. Defaults to
                        "Regression Stats".
            name (str, optional): unique name to add to Component elements.
                        If None then random uuid is generated to make sure
                        it's unique. Defaults to None.
            hide_title (bool, optional): hide title. Defaults to True.
            hide_modelsummary (bool, optional): hide RegressionModelSummaryComponent
            hide_predsvsactual (bool, optional): hide PredictedVsActualComponent
            hide_residuals (bool, optional): hide ResidualsComponent
            hide_regvscol (bool, optional): hide RegressionVsColComponent
            logs (bool, optional): Use log axis. Defaults to False.
            pred_or_actual (str, optional): plot residuals vs predictions
                        or vs y (actual). Defaults to "vs_pred".
            residuals (str, {'difference', 'ratio', 'log-ratio'} optional):
                    How to calcualte residuals. Defaults to 'difference'.
            col ({str, int}, optional): Feature to use for residuals plot. Defaults to None.
        """
        super().__init__(explainer, title, name)

        assert pred_or_actual in [
            "vs_actual",
            "vs_pred",
        ], "pred_or_actual should be 'vs_actual' or 'vs_pred'!"

        self.modelsummary = RegressionModelSummaryComponent(
            explainer, name=self.name + "0", **kwargs
        )
        self.preds_vs_actual = PredictedVsActualComponent(
            explainer, name=self.name + "0", logs=logs, **kwargs
        )
        self.residuals = ResidualsComponent(
            explainer,
            name=self.name + "1",
            pred_or_actual=pred_or_actual,
            residuals=residuals,
            **kwargs,
        )
        self.reg_vs_col = RegressionVsColComponent(
            explainer, name=self.name + "2", logs=logs, **kwargs
        )

    def layout(self):
        return dbc.Container(
            [
                make_hideable(
                    dbc.Row(html.H2("Model Performance:"), class_name="mt-4 gx-4"),
                    hide=self.hide_title,
                ),
                dbc.Row(
                    [
                        make_hideable(
                            dbc.Col(self.modelsummary.layout()),
                            hide=self.hide_modelsummary,
                        ),
                        make_hideable(
                            dbc.Col(self.preds_vs_actual.layout()),
                            hide=self.hide_predsvsactual,
                        ),
                    ],
                    class_name="mt-4 gx-4",
                ),
                dbc.Row(
                    [
                        make_hideable(
                            dbc.Col(self.residuals.layout()), hide=self.hide_residuals
                        ),
                        make_hideable(
                            dbc.Col(self.reg_vs_col.layout()), hide=self.hide_regvscol
                        ),
                    ],
                    class_name="mt-4 gx-4",
                ),
            ],
            fluid=True,
        )

    def to_html(self, state_dict=None, add_header=True):
        html = to_html.hide(to_html.title(self.title), hide=self.hide_title)
        html += to_html.card_rows(
            [
                to_html.hide(
                    self.modelsummary.to_html(state_dict, add_header=False),
                    hide=self.hide_modelsummary,
                ),
                to_html.hide(
                    self.preds_vs_actual.to_html(state_dict, add_header=False),
                    hide=self.hide_predsvsactual,
                ),
            ],
            [
                to_html.hide(
                    self.residuals.to_html(state_dict, add_header=False),
                    hide=self.hide_residuals,
                ),
                to_html.hide(
                    self.reg_vs_col.to_html(state_dict, add_header=False),
                    hide=self.hide_regvscol,
                ),
            ],
        )
        if add_header:
            return to_html.add_header(html)
        return html


class IndividualPredictionsComposite(ExplainerComponent):
    def __init__(
        self,
        explainer,
        title="Individual Predictions",
        name=None,
        hide_predindexselector=False,
        hide_predictionsummary=False,
        hide_contributiongraph=False,
        hide_pdp=False,
        hide_contributiontable=False,
        hide_title=False,
        hide_selector=True,
        index_check=True,
        **kwargs,
    ):
        """Composite for a number of component that deal with individual predictions:

        - random index selector
        - prediction summary
        - shap contributions graph
        - shap contribution table
        - pdp graph

        Args:
            explainer (Explainer): explainer object constructed with either
                        ClassifierExplainer() or RegressionExplainer()
            title (str, optional): Title of tab or page. Defaults to
                        "Individual Predictions".
            name (str, optional): unique name to add to Component elements.
                        If None then random uuid is generated to make sure
                        it's unique. Defaults to None.
            hide_predindexselector (bool, optional): hide ClassifierRandomIndexComponent
                or RegressionRandomIndexComponent
            hide_predictionsummary (bool, optional): hide ClassifierPredictionSummaryComponent
                or RegressionPredictionSummaryComponent
            hide_contributiongraph (bool, optional): hide ShapContributionsGraphComponent
            hide_pdp (bool, optional): hide PdpComponent
            hide_contributiontable (bool, optional): hide ShapContributionsTableComponent
            hide_title (bool, optional): hide title. Defaults to False.
            index_check (bool, optional): only pass valid indexes from random index
                selector to feature input. Defaults to True.
            hide_selector(bool, optional): hide all pos label selectors. Defaults to True.
        """
        super().__init__(explainer, title, name)

        if self.explainer.is_classifier:
            self.index = ClassifierRandomIndexComponent(
                explainer, hide_selector=hide_selector, **kwargs
            )
            self.summary = ClassifierPredictionSummaryComponent(
                explainer, hide_selector=hide_selector, **kwargs
            )
        elif self.explainer.is_regression:
            self.index = RegressionRandomIndexComponent(
                explainer, hide_selector=hide_selector, **kwargs
            )
            self.summary = RegressionPredictionSummaryComponent(
                explainer, hide_selector=hide_selector, **kwargs
            )

        self.contributions = ShapContributionsGraphComponent(
            explainer, hide_selector=hide_selector, **kwargs
        )
        self.pdp = PdpComponent(
            explainer, name=self.name + "3", hide_selector=hide_selector, **kwargs
        )
        self.contributions_list = ShapContributionsTableComponent(
            explainer, hide_selector=hide_selector, **kwargs
        )

        self.index_connector = IndexConnector(
            self.index,
            [self.summary, self.contributions, self.pdp, self.contributions_list],
            explainer=explainer if index_check else None,
        )

    def layout(self):
        return dbc.Container(
            [
                dbc.Row(
                    [
                        make_hideable(
                            dbc.Col(self.index.layout()),
                            hide=self.hide_predindexselector,
                        ),
                        make_hideable(
                            dbc.Col(self.summary.layout()),
                            hide=self.hide_predictionsummary,
                        ),
                    ],
                    class_name="mt-4 g-x4",
                ),
                dbc.Row(
                    [
                        make_hideable(
                            dbc.Col(self.contributions.layout()),
                            hide=self.hide_contributiongraph,
                        ),
                        make_hideable(dbc.Col(self.pdp.layout()), hide=self.hide_pdp),
                    ],
                    class_name="mt-4 gx-4",
                ),
                dbc.Row(
                    [
                        make_hideable(
                            dbc.Col(self.contributions_list.layout(), md=6),
                            hide=self.hide_contributiontable,
                        ),
                        dbc.Col(
                            [
                                html.Div([]),
                            ],
                            md=6,
                        ),
                    ],
                    class_name="mt-4 gx-4",
                ),
            ],
            fluid=True,
        )

    def to_html(self, state_dict=None, add_header=True):
        html = to_html.title(self.title)
        html += to_html.card_rows(
            [
                to_html.hide(
                    self.index.to_html(state_dict, add_header=False),
                    self.hide_predindexselector,
                ),
                to_html.hide(
                    self.summary.to_html(state_dict, add_header=False),
                    self.hide_predictionsummary,
                ),
            ],
            [
                to_html.hide(
                    self.contributions.to_html(state_dict, add_header=False),
                    self.hide_contributiongraph,
                ),
                to_html.hide(
                    self.pdp.to_html(state_dict, add_header=False), self.hide_pdp
                ),
            ],
            [
                to_html.hide(
                    self.contributions_list.to_html(state_dict, add_header=False),
                    self.hide_contributiontable,
                )
            ],
        )
        if add_header:
            return to_html.add_header(html)
        return html


class WhatIfComposite(ExplainerComponent):
    def __init__(
        self,
        explainer,
        title="What if...",
        name=None,
        hide_whatifindexselector=False,
        hide_inputeditor=False,
        hide_whatifprediction=False,
        hide_whatifcontributiongraph=False,
        hide_whatifpdp=False,
        hide_whatifcontributiontable=False,
        hide_title=True,
        hide_selector=True,
        index_check=True,
        n_input_cols=4,
        sort="importance",
        **kwargs,
    ):
        """Composite for the whatif component:

        Args:
            explainer (Explainer): explainer object constructed with either
                        ClassifierExplainer() or RegressionExplainer()
            title (str, optional): Title of tab or page. Defaults to
                        "Individual Predictions".
            name (str, optional): unique name to add to Component elements.
                        If None then random uuid is generated to make sure
                        it's unique. Defaults to None.
            hide_title (bool, optional): hide title. Defaults to True.
            hide_selector(bool, optional): hide all pos label selectors. Defaults to True.
            hide_whatifindexselector (bool, optional): hide ClassifierRandomIndexComponent
                or RegressionRandomIndexComponent
            hide_inputeditor (bool, optional): hide FeatureInputComponent
            hide_whatifprediction (bool, optional): hide PredictionSummaryComponent
            hide_whatifcontributiongraph (bool, optional): hide ShapContributionsGraphComponent
            hide_whatifcontributiontable (bool, optional): hide ShapContributionsTableComponent
            hide_whatifpdp (bool, optional): hide PdpComponent
            index_check (bool, optional): only pass valid indexes from random index
                selector to feature input. Defaults to True.
            n_input_cols (int, optional): number of columns to divide the feature inputs into.
                Defaults to 4.
            sort ({'abs', 'high-to-low', 'low-to-high', 'importance'}, optional): sorting of shap values.
                        Defaults to 'importance'.
        """
        super().__init__(explainer, title, name)

        if "hide_whatifcontribution" in kwargs:
            print(
                "Warning: hide_whatifcontribution will be deprecated, use hide_whatifcontributiongraph instead!"
            )
            self.hide_whatifcontributiongraph = kwargs["hide_whatifcontribution"]

        self.input = FeatureInputComponent(
            explainer,
            hide_selector=hide_selector,
            n_input_cols=self.n_input_cols,
            **update_params(kwargs, hide_index=False),
        )

        if self.explainer.is_classifier:
            self.index = ClassifierRandomIndexComponent(
                explainer, hide_selector=hide_selector, **kwargs
            )
            self.prediction = ClassifierPredictionSummaryComponent(
                explainer,
                feature_input_component=self.input,
                hide_star_explanation=True,
                hide_selector=hide_selector,
                **kwargs,
            )
        elif self.explainer.is_regression:
            self.index = RegressionRandomIndexComponent(explainer, **kwargs)
            self.prediction = RegressionPredictionSummaryComponent(
                explainer, feature_input_component=self.input, **kwargs
            )

        self.contribgraph = ShapContributionsGraphComponent(
            explainer,
            feature_input_component=self.input,
            hide_selector=hide_selector,
            sort=sort,
            **kwargs,
        )
        self.contribtable = ShapContributionsTableComponent(
            explainer,
            feature_input_component=self.input,
            hide_selector=hide_selector,
            sort=sort,
            **kwargs,
        )

        self.pdp = PdpComponent(
            explainer,
            feature_input_component=self.input,
            hide_selector=hide_selector,
            **kwargs,
        )

        self.index_connector = IndexConnector(
            self.index, self.input, explainer=explainer if index_check else None
        )

    def layout(self):
        return dbc.Container(
            [
                make_hideable(
                    dbc.Row(html.H1(self.title), class_name="mt-4 gx-4"),
                    hide=self.hide_title,
                ),
                dbc.Row(
                    [
                        make_hideable(
                            dbc.Col(
                                [
                                    self.index.layout(),
                                ],
                                md=6,
                            ),
                            hide=self.hide_whatifindexselector,
                        ),
                        make_hideable(
                            dbc.Col(
                                [
                                    self.prediction.layout(),
                                ],
                                md=6,
                            ),
                            hide=self.hide_whatifprediction,
                        ),
                    ],
                    class_name="mt-4 gx-4",
                ),
                dbc.Row(
                    [
                        make_hideable(
                            dbc.Col(self.input.layout()), hide=self.hide_inputeditor
                        ),
                    ],
                    class_name="mt-4 gx-4",
                ),
                dbc.Row(
                    [
                        make_hideable(
                            dbc.Col(self.contribgraph.layout()),
                            hide=self.hide_whatifcontributiongraph,
                        ),
                        make_hideable(
                            dbc.Col(self.pdp.layout()), hide=self.hide_whatifpdp
                        ),
                    ],
                    class_name="mt-4 gx-4",
                ),
                dbc.Row(
                    [
                        make_hideable(
                            dbc.Col([self.contribtable.layout()], md=6),
                            hide=self.hide_whatifcontributiontable,
                        ),
                        dbc.Col([], md=6),
                    ],
                    class_name="mt-4 gx-4",
                ),
            ],
            fluid=True,
        )

    def to_html(self, state_dict=None, add_header=True):
        html = to_html.title(self.title)
        html += to_html.card_rows(
            [
                to_html.hide(
                    self.index.to_html(state_dict, add_header=False),
                    self.hide_whatifindexselector,
                ),
                to_html.hide(
                    self.prediction.to_html(state_dict, add_header=False),
                    self.hide_whatifprediction,
                ),
            ],
            [
                to_html.hide(
                    self.input.to_html(state_dict, add_header=False),
                    self.hide_inputeditor,
                )
            ],
            [
                to_html.hide(
                    self.contribgraph.to_html(state_dict, add_header=False),
                    self.hide_whatifcontributiongraph,
                ),
                to_html.hide(
                    self.pdp.to_html(state_dict, add_header=False), self.hide_whatifpdp
                ),
            ],
            [
                to_html.hide(
                    self.contribtable.to_html(state_dict, add_header=False),
                    self.hide_whatifcontributiontable,
                )
            ],
        )
        html = to_html.div(html)
        if add_header:
            return to_html.add_header(html)
        return html


class ShapDependenceComposite(ExplainerComponent):
    def __init__(
        self,
        explainer,
        title="Feature Dependence",
        name=None,
        hide_selector=True,
        hide_shapsummary=False,
        hide_shapdependence=False,
        depth=None,
        **kwargs,
    ):
        """Composite of ShapSummary and ShapDependence component

        Args:
            explainer (Explainer): explainer object constructed with either
                        ClassifierExplainer() or RegressionExplainer()
            title (str, optional): Title of tab or page. Defaults to
                        "Feature Dependence".
            name (str, optional): unique name to add to Component elements.
                        If None then random uuid is generated to make sure
                        it's unique. Defaults to None.
            hide_selector (bool, optional): hide all pos label selectors. Defaults to True.
            hide_shapsummary (bool, optional): hide ShapSummaryComponent
            hide_shapdependence (bool, optional): ShapDependenceComponent
            depth (int, optional): Number of features to display. Defaults to None.
        """
        super().__init__(explainer, title, name)

        self.shap_summary = ShapSummaryComponent(
            self.explainer,  # name=self.name+"0",
            **update_params(kwargs, hide_selector=hide_selector, depth=depth),
        )
        self.shap_dependence = ShapDependenceComponent(
            self.explainer, hide_selector=hide_selector, **kwargs  # name=self.name+"1",
        )
        self.connector = ShapSummaryDependenceConnector(
            self.shap_summary, self.shap_dependence
        )

    def layout(self):
        return dbc.Container(
            [
                dbc.Row(
                    [
                        make_hideable(
                            dbc.Col(self.shap_summary.layout()),
                            hide=self.hide_shapsummary,
                        ),
                        make_hideable(
                            dbc.Col(self.shap_dependence.layout()),
                            hide=self.hide_shapdependence,
                        ),
                    ]
                ),
            ],
            class_name="my-4 g-4",
            fluid=True,
        )

    def to_html(self, state_dict=None, add_header=True):
        html = to_html.title(self.title)
        html += to_html.card_rows(
            [
                to_html.hide(
                    self.shap_summary.to_html(state_dict, add_header=False),
                    self.hide_shapsummary,
                ),
                to_html.hide(
                    self.shap_dependence.to_html(state_dict, add_header=False),
                    self.hide_shapdependence,
                ),
            ],
        )
        if add_header:
            return to_html.add_header(html)
        return html


class ShapInteractionsComposite(ExplainerComponent):
    def __init__(
        self,
        explainer,
        title="Feature Interactions",
        name=None,
        hide_selector=True,
        hide_interactionsummary=False,
        hide_interactiondependence=False,
        depth=None,
        **kwargs,
    ):
        """Composite of InteractionSummaryComponent and InteractionDependenceComponent

        Args:
            explainer (Explainer): explainer object constructed with either
                        ClassifierExplainer() or RegressionExplainer()
            title (str, optional): Title of tab or page. Defaults to
                        "Feature Interactions".
            name (str, optional): unique name to add to Component elements.
                        If None then random uuid is generated to make sure
                        it's unique. Defaults to None.
            hide_selector (bool, optional): hide all pos label selectors. Defaults to True.
            hide_interactionsummary (bool, optional): hide InteractionSummaryComponent
            hide_interactiondependence (bool, optional): hide InteractionDependenceComponent
            depth (int, optional): Initial number of features to display. Defaults to None.
        """
        super().__init__(explainer, title, name)
        self.interaction_summary = InteractionSummaryComponent(
            explainer, hide_selector=hide_selector, depth=depth, **kwargs
        )
        self.interaction_dependence = InteractionDependenceComponent(
            explainer, hide_selector=hide_selector, **kwargs
        )
        self.connector = InteractionSummaryDependenceConnector(
            self.interaction_summary, self.interaction_dependence
        )

    def layout(self):
        return dbc.Container(
            [
                dbc.Row(
                    [
                        make_hideable(
                            dbc.Col(self.interaction_summary.layout()),
                            hide=self.hide_interactionsummary,
                        ),
                        make_hideable(
                            dbc.Col(self.interaction_dependence.layout()),
                            hide=self.hide_interactiondependence,
                        ),
                    ],
                    class_name="mt-4 gx-4",
                )
            ],
            fluid=True,
        )

    def to_html(self, state_dict=None, add_header=True):
        html = to_html.title(self.title)
        html += to_html.card_rows(
            [
                to_html.hide(
                    self.interaction_summary.to_html(state_dict, add_header=False),
                    self.hide_interactionsummary,
                ),
                to_html.hide(
                    self.interaction_dependence.to_html(state_dict, add_header=False),
                    self.hide_interactiondependence,
                ),
            ],
        )
        if add_header:
            return to_html.add_header(html)
        return html


class DecisionTreesComposite(ExplainerComponent):
    def __init__(
        self,
        explainer,
        title="Decision Trees",
        name=None,
        hide_treeindexselector=False,
        hide_treesgraph=False,
        hide_treepathtable=False,
        hide_treepathgraph=False,
        hide_selector=True,
        index_check=True,
        **kwargs,
    ):
        """Composite of decision tree related components:

        - index selector
        - individual decision trees barchart
        - decision path table
        - deciion path graph

        Args:
            explainer (Explainer): explainer object constructed with either
                        RandomForestClassifierExplainer() or RandomForestRegressionExplainer()
            title (str, optional): Title of tab or page. Defaults to
                        "Decision Trees".
            name (str, optional): unique name to add to Component elements.
                        If None then random uuid is generated to make sure
                        it's unique. Defaults to None.
            hide_treeindexselector (bool, optional): hide ClassifierRandomIndexComponent
                or RegressionRandomIndexComponent
            hide_treesgraph (bool, optional): hide DecisionTreesComponent
            hide_treepathtable (bool, optional): hide DecisionPathTableComponent
            hide_treepathgraph (bool, optional): DecisionPathGraphComponent
            hide_selector (bool, optional): hide all pos label selectors. Defaults to True.
            index_check (bool, optional): only pass valid indexes from random index
                selector to feature input. Defaults to True.
        """
        super().__init__(explainer, title, name)

        self.trees = DecisionTreesComponent(
            explainer, hide_selector=hide_selector, **kwargs
        )
        self.decisionpath_table = DecisionPathTableComponent(
            explainer, hide_selector=hide_selector, **kwargs
        )

        if explainer.is_classifier:
            self.index = ClassifierRandomIndexComponent(
                explainer, hide_selector=hide_selector, **kwargs
            )
        elif explainer.is_regression:
            self.index = RegressionRandomIndexComponent(explainer, **kwargs)

        self.decisionpath_graph = DecisionPathGraphComponent(
            explainer, hide_selector=hide_selector, **kwargs
        )

        self.index_connector = IndexConnector(
            self.index,
            [self.trees, self.decisionpath_table, self.decisionpath_graph],
            explainer=explainer if index_check else None,
        )
        self.highlight_connector = HighlightConnector(
            self.trees, [self.decisionpath_table, self.decisionpath_graph]
        )

    def layout(self):
        return dbc.Container(
            [
                dbc.Row(
                    [
                        make_hideable(
                            dbc.Col(
                                [
                                    self.index.layout(),
                                ]
                            ),
                            hide=self.hide_treeindexselector,
                        ),
                    ],
                    class_name="mt-4 gx-4",
                ),
                dbc.Row(
                    [
                        make_hideable(
                            dbc.Col(
                                [
                                    self.trees.layout(),
                                ]
                            ),
                            hide=self.hide_treesgraph,
                        ),
                    ],
                    class_name="mt-4 g-x4",
                ),
                dbc.Row(
                    [
                        make_hideable(
                            dbc.Col(
                                [
                                    self.decisionpath_table.layout(),
                                ]
                            ),
                            hide=self.hide_treepathtable,
                        ),
                    ],
                    class_name="mt-4 gx-4",
                ),
                dbc.Row(
                    [
                        make_hideable(
                            dbc.Col([self.decisionpath_graph.layout()]),
                            hide=self.hide_treepathgraph,
                        ),
                    ],
                    class_name="mt-4 g-x4",
                ),
            ],
            fluid=True,
        )

    def to_html(self, state_dict=None, add_header=True):
        html = to_html.title(self.title)
        html += to_html.card_rows(
            [
                to_html.hide(
                    self.index.to_html(state_dict, add_header=False),
                    self.hide_treeindexselector,
                )
            ],
            [
                to_html.hide(
                    self.trees.to_html(state_dict, add_header=False),
                    self.hide_treesgraph,
                )
            ],
            [
                to_html.hide(
                    self.decisionpath_table.to_html(state_dict, add_header=False),
                    self.hide_treepathtable,
                )
            ],
        )
        if add_header:
            return to_html.add_header(html)
        return html


class SimplifiedClassifierComposite(ExplainerComponent):
    def __init__(
        self,
        explainer,
        title="Simple Classifier Explainer",
        name=None,
        hide_title=False,
        classifier_custom_component="roc_auc",
        hide_confusionmatrix=False,
        hide_classifier_custom_component=False,
        hide_shapsummary=False,
        hide_shapdependence=False,
        hide_predindexselector=False,
        hide_predictionsummary=False,
        hide_contributiongraph=False,
        **kwargs,
    ):
        """Composite of multiple classifier related components, on a single tab:
            - confusion matrix
            - one other model quality indicator: choose from pr auc graph, precision graph,
                    lift curve, classification graph, or roc auc graph
            - shap importance
            - shap dependence
            - index selector
            - index prediction summary
            - index shap contribution graph
        Args:
            explainer (Explainer): explainer object constructed with either
                        ClassifierExplainer() or RegressionExplainer()
            title (str, optional): Title of tab or page. Defaults to
                        "Simple Classification Stats".
            name (str, optional): unique name to add to Component elements.
                        If None then random uuid is generated to make sure
                        it's unique. Defaults to None.
            hide_title (bool, optional): hide the title. Defaults to False.
            classifier_custom_component (str, optional): custom classifier quality indicator
                    supported by the ClassifierExplainer object. Valid values are:
                    'roc_auc', 'metrics', pr_auc', 'precision_graph', 'lift_curve',
                    'classification'. Defaults to 'roc_auc'.
            hide_confusionmatrix (bool, optional): hide ConfusionMatrixComponent
            hide_classifier_custom_component (bool, optional): hide the chosen classifier_custom_component
            hide_shapsummary (bool, optional): hide ShapSummaryComponent
            hide_shapdependence (bool, optional): hide ShapDependenceComponent
            hide_predindexselector (bool, optional): hide ClassifierRandomIndexComponent
                or RegressionRandomIndexComponent
            hide_predictionsummary (bool, optional): hide ClassifierPredictionSummaryComponent
                or RegressionPredictionSummaryComponent
            hide_contributiongraph (bool, optional): hide ShapContributionsGraphComponent
        """
        super().__init__(explainer, title=title, name=name)

        self.confusionmatrix = ConfusionMatrixComponent(
            explainer,
            **update_params(
                kwargs, hide_percentage=True, hide_selector=True, hide_normalize=True
            ),
        )

        # select custom classifier report metric
        if classifier_custom_component == "metrics":
            self.classifier_custom_component = ClassifierModelSummaryComponent(
                explainer, **update_params(kwargs, hide_selector=True)
            )
        elif classifier_custom_component == "pr_auc":
            self.classifier_custom_component = PrAucComponent(
                explainer, **update_params(kwargs, hide_selector=True)
            )
        elif classifier_custom_component == "precision_graph":
            self.classifier_custom_component = PrecisionComponent(
                explainer, **update_params(kwargs, hide_selector=True)
            )
        elif classifier_custom_component == "lift_curve":
            self.classifier_custom_component = LiftCurveComponent(
                explainer, **update_params(kwargs, hide_selector=True)
            )
        elif classifier_custom_component == "classifiction":
            self.classifier_custom_component = ClassificationComponent(
                explainer, **update_params(kwargs, hide_selector=True)
            )
        elif classifier_custom_component == "roc_auc":
            self.classifier_custom_component = RocAucComponent(
                explainer, **update_params(kwargs, hide_selector=True)
            )
        else:
            raise ValueError(
                "ERROR: SimplifiedClassifierDashboard parameter classifier_custom_component "
                "should be in {'metrics', 'roc_auc', pr_auc', 'precision_graph', 'lift_curve', 'class_graph'} "
                f"but you passed {classifier_custom_component}!"
            )

        # SHAP summary & dependence
        self.shap_summary = ShapSummaryComponent(
            explainer,
            **update_params(
                kwargs,
                title="Shap Feature Importances",
                hide_index=True,
                hide_selector=True,
                depth=None,
                hide_depth=True,
            ),
        )
        self.shap_dependence = ShapDependenceComponent(
            explainer,
            **update_params(
                kwargs, hide_selector=True, hide_index=True, color_col="no_color_col"
            ),
        )

        # SHAP contribution, along with prediction summary
        self.index = ClassifierRandomIndexComponent(
            explainer, hide_selector=True, **kwargs
        )
        self.summary = ClassifierPredictionSummaryComponent(
            explainer, hide_index=True, hide_selector=True, **kwargs
        )
        self.contributions = ShapContributionsGraphComponent(
            explainer, hide_index=True, hide_depth=True, hide_selector=True, **kwargs
        )

        self.cutoffconnector = CutoffConnector(
            self.confusionmatrix, self.classifier_custom_component
        )
        self.connector = ShapSummaryDependenceConnector(
            self.shap_summary, self.shap_dependence
        )
        self.index_connector = IndexConnector(
            self.index, [self.summary, self.contributions]
        )

    def layout(self):
        return dbc.Container(
            [
                make_hideable(
                    dbc.Row(
                        html.H1(self.title, id="simple-classifier-composite-title"),
                        class_name="mt-4 gx-4",
                    ),
                    hide=self.hide_title,
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.H2("Model performance"),
                                dbc.Row(
                                    [
                                        make_hideable(
                                            dbc.Col(self.confusionmatrix.layout()),
                                            hide=self.hide_confusionmatrix,
                                        ),
                                        make_hideable(
                                            dbc.Col(
                                                self.classifier_custom_component.layout()
                                            ),
                                            hide=self.hide_classifier_custom_component,
                                        ),
                                    ]
                                ),
                            ]
                        )
                    ],
                    class_name="mt-4 gx-4",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.H3("SHAP values"),
                                dbc.Row(
                                    [
                                        make_hideable(
                                            dbc.Col(self.shap_summary.layout()),
                                            hide=self.hide_shapsummary,
                                        ),
                                        make_hideable(
                                            dbc.Col(self.shap_dependence.layout()),
                                            hide=self.hide_shapdependence,
                                        ),
                                    ]
                                ),
                            ]
                        )
                    ],
                    class_name="mt-4 gx-4",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.H2("Individual predictions"),
                                dbc.Row(
                                    [
                                        make_hideable(
                                            dbc.Col(self.index.layout()),
                                            hide=self.hide_predindexselector,
                                        ),
                                        make_hideable(
                                            dbc.Col(self.summary.layout()),
                                            hide=self.hide_predictionsummary,
                                        ),
                                    ],
                                ),
                                dbc.Row(
                                    [
                                        make_hideable(
                                            dbc.Col(self.contributions.layout()),
                                            hide=self.hide_contributiongraph,
                                        ),
                                    ]
                                ),
                            ]
                        )
                    ],
                    class_name="mt-4 gx-4",
                ),
            ],
            fluid=True,
        )

    def to_html(self, state_dict=None, add_header=True):
        html = to_html.hide(to_html.title(self.title), hide=self.hide_title)
        html += to_html.card_rows(
            [
                to_html.hide(
                    self.confusionmatrix.to_html(state_dict, add_header=False),
                    hide=self.hide_confusionmatrix,
                ),
                to_html.hide(
                    self.classifier_custom_component.to_html(
                        state_dict, add_header=False
                    ),
                    hide=self.hide_classifier_custom_component,
                ),
            ],
            [
                to_html.hide(
                    self.shap_summary.to_html(state_dict, add_header=False),
                    hide=self.hide_shapsummary,
                ),
                to_html.hide(
                    self.shap_dependence.to_html(state_dict, add_header=False),
                    hide=self.hide_shapdependence,
                ),
            ],
            [
                to_html.hide(
                    self.index.to_html(state_dict, add_header=False),
                    hide=self.hide_predindexselector,
                ),
                to_html.hide(
                    self.summary.to_html(state_dict, add_header=False),
                    hide=self.hide_predictionsummary,
                ),
            ],
            [
                to_html.hide(
                    self.contributions.to_html(state_dict, add_header=False),
                    hide=self.hide_contributiongraph,
                )
            ],
        )
        if add_header:
            return to_html.add_header(html)
        return html


class SimplifiedRegressionComposite(ExplainerComponent):
    def __init__(
        self,
        explainer,
        title="Simple Regression Explainer",
        name=None,
        hide_title=False,
        regression_custom_component="vs_col",
        hide_goodness_of_fit=False,
        hide_regression_custom_component=False,
        hide_shapsummary=False,
        hide_shapdependence=False,
        hide_predindexselector=False,
        hide_predictionsummary=False,
        hide_contributiongraph=False,
        **kwargs,
    ):
        """Composite of multiple classifier related components, on a single tab:
            - goodness of fit component
            - one other model quality indicator: 'metrics', 'residuals' or'residuals_vs_col'
            - shap importance
            - shap dependence
            - index selector
            - index prediction summary
            - index shap contribution graph
        Args:
            explainer (Explainer): explainer object constructed with either
                        ClassifierExplainer() or RegressionExplainer()
            title (str, optional): Title of tab or page. Defaults to
                        "Simple Classification Stats".
            name (str, optional): unique name to add to Component elements.
                        If None then random uuid is generated to make sure
                        it's unique. Defaults to None.
            hide_title (bool, optional): hide the title. Defaults to False.
            regression_custom_component (str, optional): custom classifier quality
                indicator supported by the ClassifierExplainer object. Valid values are:
                'metrics', 'residuals' or'vs_col'
            hide_goodness_of_fit (bool, optional): hide goodness of fit component
            hide_regression_custom_component (bool, optional): hide the chosen
                regression_custom_component
            hide_shapsummary (bool, optional): hide ShapSummaryComponent
            hide_shapdependence (bool, optional): hide ShapDependenceComponent
            hide_predindexselector (bool, optional): hide RegressionRandomIndexComponent
                or RegressionRandomIndexComponent
            hide_predictionsummary (bool, optional): hide RegressionPredictionSummaryComponent
                or RegressionPredictionSummaryComponent
            hide_contributiongraph (bool, optional): hide ShapContributionsGraphComponent
        """
        super().__init__(explainer, title, name)

        self.goodness_of_fit = PredictedVsActualComponent(explainer, **kwargs)

        # select custom classifier report metric
        if regression_custom_component == "metrics":
            self.regression_custom_component = RegressionModelSummaryComponent(
                explainer, **kwargs
            )
        elif regression_custom_component == "residuals":
            self.regression_custom_component = ResidualsComponent(explainer, **kwargs)
        elif regression_custom_component == "vs_col":
            self.regression_custom_component = RegressionVsColComponent(
                explainer, **update_params(kwargs, display="predicted")
            )
        else:
            raise ValueError(
                "ERROR: SimplifiedRegressionDashboard parameter "
                "regression_custom_component should be in {'metrics', 'residuals', 'vs_col'}"
                f" but you passed {regression_custom_component}!"
            )

        # SHAP summary & dependence
        self.shap_summary = ShapSummaryComponent(
            explainer,
            **update_params(
                kwargs,
                title="Shap Feature Importances",
                hide_index=True,
                depth=None,
                hide_depth=True,
            ),
        )
        self.shap_dependence = ShapDependenceComponent(
            explainer, **update_params(kwargs, hide_index=True)
        )

        # SHAP contribution, along with prediction summary
        self.index = RegressionRandomIndexComponent(explainer, **kwargs)
        self.summary = RegressionPredictionSummaryComponent(
            explainer, hide_index=True, **kwargs
        )
        self.contributions = ShapContributionsGraphComponent(
            explainer, **update_params(kwargs, hide_index=True, hide_depth=True)
        )

        self.connector = ShapSummaryDependenceConnector(
            self.shap_summary, self.shap_dependence
        )
        self.index_connector = IndexConnector(
            self.index, [self.summary, self.contributions]
        )

    def layout(self):
        return dbc.Container(
            [
                make_hideable(
                    dbc.Row(
                        html.H1(self.title, id="simple-regression-composite-title"),
                        class_name="mt-4 gx-4",
                    ),
                    hide=self.hide_title,
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.H2("Model performance"),
                                dbc.Row(
                                    [
                                        make_hideable(
                                            dbc.Col(self.goodness_of_fit.layout()),
                                            hide=self.hide_goodness_of_fit,
                                        ),
                                        make_hideable(
                                            dbc.Col(
                                                self.regression_custom_component.layout()
                                            ),
                                            hide=self.hide_regression_custom_component,
                                        ),
                                    ]
                                ),
                            ]
                        )
                    ],
                    class_name="mt-4 gx-4",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.H3("SHAP values"),
                                dbc.Row(
                                    [
                                        make_hideable(
                                            dbc.Col(self.shap_summary.layout()),
                                            hide=self.hide_shapsummary,
                                        ),
                                        make_hideable(
                                            dbc.Col(self.shap_dependence.layout()),
                                            hide=self.hide_shapdependence,
                                        ),
                                    ]
                                ),
                            ]
                        )
                    ],
                    class_name="mt-4 gx-4",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.H2("Individual predictions"),
                                dbc.Row(
                                    [
                                        make_hideable(
                                            dbc.Col(self.index.layout()),
                                            hide=self.hide_predindexselector,
                                        ),
                                        make_hideable(
                                            dbc.Col(self.summary.layout()),
                                            hide=self.hide_predictionsummary,
                                        ),
                                    ]
                                ),
                                dbc.Row(
                                    [
                                        make_hideable(
                                            dbc.Col(self.contributions.layout()),
                                            hide=self.hide_contributiongraph,
                                        ),
                                    ]
                                ),
                            ]
                        )
                    ],
                    class_name="mt-4 gx-4",
                ),
            ],
            fluid=True,
        )

    def to_html(self, state_dict=None, add_header=True):
        html = to_html.hide(to_html.title(self.title), hide=self.hide_title)
        html += to_html.card_rows(
            [
                to_html.hide(
                    self.goodness_of_fit.to_html(state_dict, add_header=False),
                    hide=self.hide_goodness_of_fit,
                ),
                to_html.hide(
                    self.regression_custom_component.to_html(
                        state_dict, add_header=False
                    ),
                    hide=self.hide_regression_custom_component,
                ),
            ],
            [
                to_html.hide(
                    self.shap_summary.to_html(state_dict, add_header=False),
                    hide=self.hide_shapsummary,
                ),
                to_html.hide(
                    self.shap_dependence.to_html(state_dict, add_header=False),
                    hide=self.hide_shapdependence,
                ),
            ],
            [
                to_html.hide(
                    self.index.to_html(state_dict, add_header=False),
                    hide=self.hide_predindexselector,
                ),
                to_html.hide(
                    self.summary.to_html(state_dict, add_header=False),
                    hide=self.hide_predictionsummary,
                ),
            ],
            [
                to_html.hide(
                    self.contributions.to_html(state_dict, add_header=False),
                    hide=self.hide_contributiongraph,
                )
            ],
        )
        if add_header:
            return to_html.add_header(html)
        return html
