import pandas as pd
import numpy as np
import types
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from explainerdashboard import ClassifierExplainer


def test_linreg_explainer_len(precalculated_linear_regression_explainer, testlen):
    assert len(precalculated_linear_regression_explainer) == testlen


def test_linreg_int_idx(precalculated_linear_regression_explainer, test_names):
    assert precalculated_linear_regression_explainer.get_idx(test_names[0]) == 0


def test_linreg_random_index(precalculated_linear_regression_explainer):
    assert isinstance(precalculated_linear_regression_explainer.random_index(), int)
    assert isinstance(
        precalculated_linear_regression_explainer.random_index(return_str=True), str
    )


def test_linreg_preds(precalculated_linear_regression_explainer):
    assert isinstance(precalculated_linear_regression_explainer.preds, np.ndarray)


def test_linreg_pred_percentiles(precalculated_linear_regression_explainer):
    assert isinstance(
        precalculated_linear_regression_explainer.pred_percentiles(), np.ndarray
    )


def test_linreg_permutation_importances(precalculated_linear_regression_explainer):
    assert isinstance(
        precalculated_linear_regression_explainer.get_permutation_importances_df(),
        pd.DataFrame,
    )


def test_linreg_metrics(precalculated_linear_regression_explainer):
    assert isinstance(precalculated_linear_regression_explainer.metrics(), dict)
    assert isinstance(
        precalculated_linear_regression_explainer.metrics_descriptions(), dict
    )


def test_linreg_mean_abs_shap_df(precalculated_linear_regression_explainer):
    assert isinstance(
        precalculated_linear_regression_explainer.get_mean_abs_shap_df(), pd.DataFrame
    )


def test_linreg_top_interactions(precalculated_linear_regression_explainer):
    assert isinstance(
        precalculated_linear_regression_explainer.top_shap_interactions("Age"), list
    )
    assert isinstance(
        precalculated_linear_regression_explainer.top_shap_interactions("Age", topx=4),
        list,
    )


def test_linreg_contrib_df(precalculated_linear_regression_explainer):
    assert isinstance(
        precalculated_linear_regression_explainer.get_contrib_df(0), pd.DataFrame
    )
    assert isinstance(
        precalculated_linear_regression_explainer.get_contrib_df(0, topx=3),
        pd.DataFrame,
    )


def test_linreg_shap_base_value(precalculated_linear_regression_explainer):
    assert isinstance(
        precalculated_linear_regression_explainer.shap_base_value(),
        (np.floating, float),
    )


def test_linreg_shap_values_shape(precalculated_linear_regression_explainer):
    assert precalculated_linear_regression_explainer.get_shap_values_df().shape == (
        len(precalculated_linear_regression_explainer),
        len(precalculated_linear_regression_explainer.merged_cols),
    )


def test_linreg_shap_values(precalculated_linear_regression_explainer):
    assert isinstance(
        precalculated_linear_regression_explainer.get_shap_values_df(), pd.DataFrame
    )


def test_linreg_mean_abs_shap(precalculated_linear_regression_explainer):
    assert isinstance(
        precalculated_linear_regression_explainer.get_mean_abs_shap_df(), pd.DataFrame
    )


def test_linreg_calculate_properties(precalculated_linear_regression_explainer):
    precalculated_linear_regression_explainer.calculate_properties(
        include_interactions=False
    )


def test_linreg_pdp_df(precalculated_linear_regression_explainer):
    assert isinstance(
        precalculated_linear_regression_explainer.pdp_df("Age"), pd.DataFrame
    )
    assert isinstance(
        precalculated_linear_regression_explainer.pdp_df("Gender"), pd.DataFrame
    )
    assert isinstance(
        precalculated_linear_regression_explainer.pdp_df("Deck"), pd.DataFrame
    )
    assert isinstance(
        precalculated_linear_regression_explainer.pdp_df("Age", index=0), pd.DataFrame
    )
    assert isinstance(
        precalculated_linear_regression_explainer.pdp_df("Gender", index=0),
        pd.DataFrame,
    )


def test_logreg_preds(precalculated_logistic_regression_explainer):
    assert isinstance(precalculated_logistic_regression_explainer.preds, np.ndarray)


def test_logreg_pred_percentiles(precalculated_logistic_regression_explainer):
    assert isinstance(
        precalculated_logistic_regression_explainer.pred_percentiles(), np.ndarray
    )


def test_logreg_columns_ranked_by_shap(precalculated_logistic_regression_explainer):
    assert isinstance(
        precalculated_logistic_regression_explainer.columns_ranked_by_shap(), list
    )


def test_logreg_permutation_importances(precalculated_logistic_regression_explainer):
    assert isinstance(
        precalculated_logistic_regression_explainer.get_permutation_importances_df(),
        pd.DataFrame,
    )


def test_logreg_mean_abs_shap_df(precalculated_logistic_regression_explainer):
    assert isinstance(
        precalculated_logistic_regression_explainer.get_mean_abs_shap_df(), pd.DataFrame
    )


def test_logreg_contrib_df(precalculated_logistic_regression_explainer):
    assert isinstance(
        precalculated_logistic_regression_explainer.get_contrib_df(0), pd.DataFrame
    )
    assert isinstance(
        precalculated_logistic_regression_explainer.get_contrib_df(0, topx=3),
        pd.DataFrame,
    )


def test_logreg_shap_base_value(precalculated_logistic_regression_explainer):
    assert isinstance(
        precalculated_logistic_regression_explainer.shap_base_value(),
        (np.floating, float),
    )


def test_logreg_shap_values_shape(precalculated_logistic_regression_explainer):
    assert precalculated_logistic_regression_explainer.get_shap_values_df().shape == (
        len(precalculated_logistic_regression_explainer),
        len(precalculated_logistic_regression_explainer.merged_cols),
    )


def test_logreg_shap_values(precalculated_logistic_regression_explainer):
    assert isinstance(
        precalculated_logistic_regression_explainer.get_shap_values_df(), pd.DataFrame
    )


def test_logreg_mean_abs_shap(precalculated_logistic_regression_explainer):
    assert isinstance(
        precalculated_logistic_regression_explainer.get_mean_abs_shap_df(), pd.DataFrame
    )


def test_logreg_calculate_properties(precalculated_logistic_regression_explainer):
    precalculated_logistic_regression_explainer.calculate_properties(
        include_interactions=False
    )


def test_logreg_pdp_df(precalculated_logistic_regression_explainer):
    assert isinstance(
        precalculated_logistic_regression_explainer.pdp_df("Age"), pd.DataFrame
    )
    assert isinstance(
        precalculated_logistic_regression_explainer.pdp_df("Gender"), pd.DataFrame
    )
    assert isinstance(
        precalculated_logistic_regression_explainer.pdp_df("Deck"), pd.DataFrame
    )
    assert isinstance(
        precalculated_logistic_regression_explainer.pdp_df("Age", index=0), pd.DataFrame
    )
    assert isinstance(
        precalculated_logistic_regression_explainer.pdp_df("Gender", index=0),
        pd.DataFrame,
    )


def test_logreg_pos_label(precalculated_logistic_regression_explainer):
    precalculated_logistic_regression_explainer.pos_label = 1
    precalculated_logistic_regression_explainer.pos_label = "Not survived"
    assert isinstance(precalculated_logistic_regression_explainer.pos_label, int)
    assert isinstance(precalculated_logistic_regression_explainer.pos_label_str, str)
    assert precalculated_logistic_regression_explainer.pos_label == 0
    assert precalculated_logistic_regression_explainer.pos_label_str == "Not survived"


def test_logreg_pred_probas(precalculated_logistic_regression_explainer):
    assert isinstance(
        precalculated_logistic_regression_explainer.pred_probas(), np.ndarray
    )


def test_logreg_metrics(precalculated_logistic_regression_explainer):
    assert isinstance(precalculated_logistic_regression_explainer.metrics(), dict)
    assert isinstance(
        precalculated_logistic_regression_explainer.metrics(cutoff=0.9), dict
    )


def test_logreg_precision_df(precalculated_logistic_regression_explainer):
    assert isinstance(
        precalculated_logistic_regression_explainer.get_precision_df(), pd.DataFrame
    )
    assert isinstance(
        precalculated_logistic_regression_explainer.get_precision_df(multiclass=True),
        pd.DataFrame,
    )
    assert isinstance(
        precalculated_logistic_regression_explainer.get_precision_df(quantiles=4),
        pd.DataFrame,
    )


def test_logreg_lift_curve_df(precalculated_logistic_regression_explainer):
    assert isinstance(
        precalculated_logistic_regression_explainer.get_liftcurve_df(), pd.DataFrame
    )


##### KERNEL TESTS


def test_logistic_regression_kernel_shap_values(logistic_regression_kernel_explainer):
    assert isinstance(
        logistic_regression_kernel_explainer.shap_base_value(), (np.floating, float)
    )
    assert logistic_regression_kernel_explainer.get_shap_values_df().shape == (
        len(logistic_regression_kernel_explainer),
        len(logistic_regression_kernel_explainer.merged_cols),
    )
    assert isinstance(
        logistic_regression_kernel_explainer.get_shap_values_df(), pd.DataFrame
    )


def test_linear_regression_kernel_shap_values(linear_regression_kernel_explainer):
    assert isinstance(
        linear_regression_kernel_explainer.shap_base_value(), (np.floating, float)
    )
    assert linear_regression_kernel_explainer.get_shap_values_df().shape == (
        len(linear_regression_kernel_explainer),
        len(linear_regression_kernel_explainer.merged_cols),
    )
    assert isinstance(
        linear_regression_kernel_explainer.get_shap_values_df(), pd.DataFrame
    )


def test_multiclass_linearsvc_monkey_patch_guidance_supports_kernel_shap():
    """Regression test for GH #256.

    The current ClassifierExplainer error message suggests a monkey patch for
    classifiers without predict_proba(). For multiclass LinearSVC, that guidance
    should not lead to a class-count mismatch in SHAP outputs.
    """

    X, y = load_iris(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=0, stratify=y
    )

    model = LinearSVC(C=1.0, dual=False, max_iter=5000, multi_class="ovr")
    model.fit(X_train, y_train)

    # Monkey patch currently suggested by the ClassifierExplainer assertion text.
    def predict_proba(self, X):
        pred = self.predict(X)
        return np.array([1 - pred, pred]).T

    model.predict_proba = types.MethodType(predict_proba, model)

    explainer = ClassifierExplainer(
        model,
        X_test.iloc[:6].reset_index(drop=True),
        y_test.iloc[:6].reset_index(drop=True),
        shap="kernel",
        X_background=X_train.iloc[:10].reset_index(drop=True),
    )

    shap_df = explainer.get_shap_values_df()
    assert isinstance(shap_df, pd.DataFrame)


def test_multiclass_linearsvc_decision_function_fallback_supports_kernel_shap():
    X, y = load_iris(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=1, stratify=y
    )

    model = LinearSVC(C=1.0, dual=False, max_iter=5000, multi_class="ovr")
    model.fit(X_train, y_train)

    explainer = ClassifierExplainer(
        model,
        X_test.iloc[:6].reset_index(drop=True),
        y_test.iloc[:6].reset_index(drop=True),
        shap="kernel",
        X_background=X_train.iloc[:10].reset_index(drop=True),
    )

    shap_df = explainer.get_shap_values_df()
    assert isinstance(shap_df, pd.DataFrame)


def test_multiclass_linearsvc_pdp_df_supports_decision_function_fallback():
    """Regression test for PDP fallback via decision_function."""
    X, y = load_iris(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=11, stratify=y
    )

    model = LinearSVC(C=1.0, dual=False, max_iter=5000, multi_class="ovr")
    model.fit(X_train, y_train)

    explainer = ClassifierExplainer(
        model,
        X_test.iloc[:20].reset_index(drop=True),
        y_test.iloc[:20].reset_index(drop=True),
        shap="kernel",
        X_background=X_train.iloc[:15].reset_index(drop=True),
    )

    pdp = explainer.pdp_df(X.columns[0])
    assert isinstance(pdp, pd.DataFrame)
    assert len(pdp) > 0


def test_multiclass_linearsvc_permutation_importances_supports_decision_function_fallback():
    """Regression test for permutation scorer fallback via decision_function."""
    X, y = load_iris(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=12, stratify=y
    )

    model = LinearSVC(C=1.0, dual=False, max_iter=5000, multi_class="ovr")
    model.fit(X_train, y_train)

    explainer = ClassifierExplainer(
        model,
        X_test.iloc[:20].reset_index(drop=True),
        y_test.iloc[:20].reset_index(drop=True),
        shap="kernel",
        X_background=X_train.iloc[:15].reset_index(drop=True),
    )

    perm_df = explainer.get_permutation_importances_df()
    assert isinstance(perm_df, pd.DataFrame)
    assert not perm_df.empty


def test_multiclass_logreg_pdp_df_still_works_with_predict_proba():
    """Guard test to ensure existing predict_proba path remains stable."""
    X, y = load_iris(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=21, stratify=y
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    explainer = ClassifierExplainer(
        model,
        X_test.iloc[:20].reset_index(drop=True),
        y_test.iloc[:20].reset_index(drop=True),
        shap="kernel",
        X_background=X_train.iloc[:15].reset_index(drop=True),
    )

    pdp = explainer.pdp_df(X.columns[0])
    assert isinstance(pdp, pd.DataFrame)
    assert len(pdp) > 0


def test_multiclass_logreg_permutation_importances_still_works_with_predict_proba():
    """Guard test to ensure existing predict_proba permutation flow stays intact."""
    X, y = load_iris(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=22, stratify=y
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    explainer = ClassifierExplainer(
        model,
        X_test.iloc[:20].reset_index(drop=True),
        y_test.iloc[:20].reset_index(drop=True),
        shap="kernel",
        X_background=X_train.iloc[:15].reset_index(drop=True),
    )

    perm_df = explainer.get_permutation_importances_df()
    assert isinstance(perm_df, pd.DataFrame)
    assert not perm_df.empty
