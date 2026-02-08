__all__ = [
    "IndexNotFoundError",
    "append_dict_to_df",
    "safe_isinstance",
    "unwrap_calibrated_classifier",
    "align_categorical_dtypes",
    "sanitize_categorical_predict_input",
    "sorted_categorical_values",
    "guess_shap",
    "mape_score",
    "parse_cats",
    "is_binary_like_onehot_column",
    "infer_cats_from_transformed_X",
    "get_encoded_and_regular_cols",
    "split_pipeline",
    "rename_pipeline_columns",
    "get_transformed_X",
    "build_pipeline_extraction_warning",
    "retrieve_onehot_value",
    "merge_categorical_columns",
    "matching_cols",
    "remove_cat_names",
    "X_cats_to_X",
    "merge_categorical_shap_values",
    "merge_categorical_shap_interaction_values",
    "make_one_vs_all_scorer",
    "permutation_importances",
    "cv_permutation_importances",
    "get_mean_absolute_shap_df",
    "get_grid_points",
    "get_pdp_df",
    "get_precision_df",
    "get_liftcurve_df",
    "get_contrib_df",
    "get_contrib_summary_df",
    "normalize_shap_interaction_values",
    "get_decisionpath_df",
    "get_decisiontree_summary_df",
    "get_xgboost_node_dict",
    "get_xgboost_path_df",
    "get_xgboost_path_summary_df",
    "get_xgboost_preds_df",
    "get_multiclass_logodds_scores",
    "get_xgboost_output_label",
    "_ensure_numeric_predictions",  # Internal helper for XGBoost 3.0+ compatibility
    "_safe_make_scorer",  # Internal helper for CatBoost compatibility
]

from functools import partial
import re
from collections import Counter
from typing import Callable, List, Optional, Union
import warnings
import logging

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype


from sklearn.metrics import make_scorer
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, StratifiedKFold

from joblib import Parallel, delayed

logger = logging.getLogger(__name__)


def _ensure_numeric_predictions(pred):
    """Convert predictions to numeric format, handling XGBoost 3.0+ string format.

    Args:
        pred: Prediction output from model (may be string, array, scalar, list)

    Returns:
        Numeric prediction (numpy array or scalar float)
    """
    # Handle None
    if pred is None:
        return None

    # Handle string predictions (XGBoost 3.0+ may return strings like '[3.2967056E1]' or '[8.563135E-2,7.169811E-1,1.9738752E-1]')
    if isinstance(pred, str):
        try:
            # Remove brackets and whitespace
            cleaned = pred.strip().strip("[]").strip()
            # Check if it contains comma-separated values
            if "," in cleaned:
                # Multiple values - convert to array
                values = [float(v.strip()) for v in cleaned.split(",")]
                return np.asarray(values)
            else:
                # Single value
                return float(cleaned)
        except (ValueError, AttributeError, TypeError):
            # If conversion fails, try regex extraction
            import re

            # Use non-capturing group to get full numeric matches, not just exponent part
            pattern = r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?"
            matches = re.findall(pattern, pred)
            if matches:
                if len(matches) == 1:
                    return float(matches[0])
                else:
                    return np.asarray([float(m) for m in matches])
            # If all else fails, return as-is (will raise error later)
            return pred

    # Handle list/tuple of strings or mixed types
    if isinstance(pred, (list, tuple)):
        try:
            converted = []
            for item in pred:
                if isinstance(item, str):
                    cleaned = item.strip().strip("[]").strip()
                    converted.append(float(cleaned))
                else:
                    item_conv = _ensure_numeric_predictions(item)
                    converted.append(
                        float(item_conv)
                        if not isinstance(item_conv, np.ndarray)
                        else item_conv
                    )
            return np.asarray(converted)
        except (ValueError, AttributeError, TypeError):
            pass  # Fall through to array conversion

    # Convert to numpy array for processing
    try:
        pred_array = np.asarray(pred)
    except (ValueError, TypeError):
        # If we can't convert to array, try direct conversion
        if isinstance(pred, (int, float)):
            return float(pred)
        return pred

    # Handle string arrays (XGBoost 3.0+ may return arrays of strings)
    if pred_array.dtype.kind == "U":  # Unicode string array
        try:
            # Convert each string element to float
            def _convert_elem(elem):
                if isinstance(elem, str):
                    cleaned = elem.strip().strip("[]").strip()
                    # Handle comma-separated values in string
                    if "," in cleaned:
                        # Multiple values - should not happen in scalar context, but handle it
                        values = [float(v.strip()) for v in cleaned.split(",")]
                        return values[0] if len(values) == 1 else np.asarray(values)
                    # Handle scientific notation
                    try:
                        return float(cleaned)
                    except ValueError:
                        # Try regex extraction as fallback
                        import re

                        match = re.search(
                            r"[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?", cleaned
                        )
                        if match:
                            return float(match.group())
                        raise
                elif isinstance(elem, (int, float, np.integer, np.floating)):
                    return float(elem)
                elif isinstance(elem, np.ndarray):
                    return float(elem.item()) if elem.ndim == 0 else elem
                return elem

            if pred_array.ndim == 0:
                # Scalar string array
                return _convert_elem(pred_array.item())
            else:
                # Multi-dimensional string array
                converted = []
                for p in pred_array.flatten():
                    try:
                        converted.append(_convert_elem(p))
                    except (ValueError, TypeError):
                        # Try regex extraction as fallback
                        import re

                        p_str = str(p)
                        match = re.search(
                            r"[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?", p_str
                        )
                        if match:
                            converted.append(float(match.group()))
                        else:
                            raise
                return np.array(converted).reshape(pred_array.shape)
        except (ValueError, AttributeError, TypeError):
            # If conversion fails, return original (will raise error later)
            return pred

    # Already numeric, return as numpy array or scalar
    if pred_array.ndim == 0:
        return pred_array.item()
    return pred_array


def _decision_scores_to_probas(decision_scores, n_labels=None):
    """Map decision_function outputs to probability-like class scores."""
    scores = np.asarray(decision_scores)
    if scores.ndim == 0:
        scores = scores.reshape(1)
    if scores.ndim == 1 and n_labels and n_labels > 2 and scores.shape[0] == n_labels:
        scores = scores.reshape(1, -1)

    if scores.ndim == 1:
        clipped = np.clip(scores.astype("float64"), -709, 709)
        pos_probs = 1.0 / (1.0 + np.exp(-clipped))
        return np.column_stack([1.0 - pos_probs, pos_probs])

    if scores.ndim == 2:
        if scores.shape[1] == 1:
            clipped = np.clip(scores[:, 0].astype("float64"), -709, 709)
            pos_probs = 1.0 / (1.0 + np.exp(-clipped))
            return np.column_stack([1.0 - pos_probs, pos_probs])

        shifted = scores - np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(shifted)
        denom = np.sum(exp_scores, axis=1, keepdims=True)
        return exp_scores / np.clip(denom, np.finfo("float64").tiny, None)

    raise ValueError(
        f"Unexpected decision_function output shape {scores.shape}. "
        "Expected 1D or 2D scores."
    )


def _predict_proba_with_fallback(model, model_input, n_labels=None):
    """Return per-class probabilities, with decision_function fallback."""
    pred_probas = None
    predict_error = None

    if hasattr(model, "predict_proba"):
        try:
            pred_raw = model.predict_proba(model_input)
            pred_raw = _ensure_numeric_predictions(pred_raw)
            pred_probas = np.asarray(pred_raw, dtype="float64")
        except Exception as e:
            predict_error = e

    if pred_probas is not None:
        if pred_probas.ndim == 1:
            if n_labels == 2:
                pred_probas = np.column_stack([1.0 - pred_probas, pred_probas])
            else:
                pred_probas = None
        elif pred_probas.ndim != 2:
            pred_probas = None

        if (
            pred_probas is not None
            and n_labels is not None
            and pred_probas.shape[1] != n_labels
        ):
            pred_probas = None

    if pred_probas is None and hasattr(model, "decision_function"):
        decision_scores_raw = model.decision_function(model_input)
        decision_scores_raw = _ensure_numeric_predictions(decision_scores_raw)
        pred_probas = _decision_scores_to_probas(decision_scores_raw, n_labels=n_labels)

    if pred_probas is None:
        if predict_error is not None:
            raise ValueError(
                "Could not compute class probabilities from model.predict_proba(...)."
            ) from predict_error
        raise ValueError(
            "Could not compute class probabilities: model has neither a working "
            "predict_proba(...) nor decision_function(...)."
        )

    if n_labels is not None and pred_probas.shape[1] != n_labels:
        raise ValueError(
            f"Expected {n_labels} class probabilities, got shape {pred_probas.shape}."
        )
    return pred_probas


def get_multiclass_logodds_scores(model, model_input, n_labels):
    """Return per-class raw scores used as multiclass logodds/margins.

    Tries common model APIs in order and returns None when unavailable.

    Args:
        model: Fitted classifier model.
        model_input: Single-row model input (already sanitized for model type).
        n_labels (int): Expected number of classes.

    Returns:
        np.ndarray or None: 1d array of raw scores of length n_labels.
    """
    raw_scores = None

    for kwargs in (
        {"output_margin": True},
        {"raw_score": True},
        {"prediction_type": "RawFormulaVal"},
    ):
        try:
            raw_scores_raw = model.predict(model_input, **kwargs)
            raw_scores_raw = _ensure_numeric_predictions(raw_scores_raw)
            raw_scores = np.asarray(raw_scores_raw).squeeze()
            break
        except TypeError:
            pass
        except Exception:
            logger.debug(
                "Could not get multiclass raw margins with predict kwargs=%s",
                kwargs,
                exc_info=True,
            )

    if raw_scores is None and hasattr(model, "decision_function"):
        try:
            raw_scores_raw = model.decision_function(model_input)
            raw_scores_raw = _ensure_numeric_predictions(raw_scores_raw)
            raw_scores = np.asarray(raw_scores_raw).squeeze()
        except Exception:
            logger.debug(
                "Could not get multiclass raw margins with decision_function",
                exc_info=True,
            )

    if raw_scores is not None and raw_scores.ndim > 1:
        raw_scores = raw_scores[0]
    if raw_scores is not None and raw_scores.ndim == 1 and len(raw_scores) == n_labels:
        return raw_scores
    return None


def get_xgboost_output_label(model_output=None):
    """Map explainer model_output to xgboost path summary output label."""
    if model_output == "logodds":
        return "logodds"
    return "margin"


def _safe_make_scorer(
    metric, greater_is_better=True, response_method="predict", **kwargs
):
    """Wrapper around make_scorer that handles models without __sklearn_tags__.

    This fixes compatibility issues with CatBoost and other models that don't
    implement the __sklearn_tags__ attribute required by newer scikit-learn versions.
    """
    # Try to create the scorer normally
    try:
        scorer = make_scorer(
            metric,
            greater_is_better=greater_is_better,
            response_method=response_method,
            **kwargs,
        )
    except Exception:
        # If creation fails, create a wrapper scorer
        scorer = None

    # Create a wrapper that handles __sklearn_tags__ errors when scorer is called
    def _wrapped_scorer(estimator, X, y_true):
        try:
            if scorer is not None:
                return scorer(estimator, X, y_true)
        except AttributeError as e:
            if "__sklearn_tags__" in str(e):
                # Model doesn't have __sklearn_tags__, call predict/predict_proba directly
                if response_method == "predict_proba":
                    n_labels = (
                        len(estimator.classes_)
                        if hasattr(estimator, "classes_")
                        else None
                    )
                    y_pred = _predict_proba_with_fallback(
                        estimator,
                        X,
                        n_labels=n_labels,
                    )
                else:
                    y_pred = estimator.predict(X)
                    y_pred = _ensure_numeric_predictions(y_pred)

                if hasattr(metric, "__call__"):
                    score = metric(y_true, y_pred)
                else:
                    from sklearn.metrics import get_scorer

                    scorer_obj = get_scorer(metric)
                    score = scorer_obj._score_func(y_true, y_pred)

                return score if greater_is_better else -score
            raise

        # If scorer creation failed, use direct prediction
        if scorer is None:
            if response_method == "predict_proba":
                n_labels = (
                    len(estimator.classes_) if hasattr(estimator, "classes_") else None
                )
                y_pred = _predict_proba_with_fallback(
                    estimator,
                    X,
                    n_labels=n_labels,
                )
            else:
                y_pred = estimator.predict(X)
                y_pred = _ensure_numeric_predictions(y_pred)

            if hasattr(metric, "__call__"):
                score = metric(y_true, y_pred)
            else:
                from sklearn.metrics import get_scorer

                scorer_obj = get_scorer(metric)
                score = scorer_obj._score_func(y_true, y_pred)

            return score if greater_is_better else -score

    return _wrapped_scorer


def append_dict_to_df(df: pd.DataFrame, row_dict: dict) -> pd.DataFrame:
    """Appends a row to the dataframe 'df' and returns the new
    dataframe.

    Args:
        df (pd.DataFrame) data frame
        row_dict (dict): row data

    Returns:
        pd.DataFrame
    """
    if not row_dict:
        return df
    # Create new DataFrame with same dtypes as input df
    new_row_df = pd.DataFrame([row_dict], columns=df.columns).astype(df.dtypes)
    return pd.concat([df, new_row_df], ignore_index=True)


class IndexNotFoundError(Exception):
    def __init__(self, message="Index not Found", index=None):
        if index is not None:
            message = f"Index {index} not found!"
        super().__init__(message)


def safe_isinstance(obj, *instance_str):
    """Checks instance by comparing str(type(obj)) to one or more
    instance_str."""
    obj_str = str(type(obj))
    for i in instance_str:
        if i.endswith("'>"):
            if obj_str.endswith(i):
                return True
        else:
            if obj_str[:-2].endswith(i):
                return True
    return False


def unwrap_calibrated_classifier(model):
    """Return the fitted base estimator for a CalibratedClassifierCV model."""
    if not safe_isinstance(model, "CalibratedClassifierCV"):
        return model

    calibrated_classifiers = getattr(model, "calibrated_classifiers_", None)
    if calibrated_classifiers:
        calibrated = calibrated_classifiers[0]
        for attr in ("estimator", "base_estimator"):
            estimator = getattr(calibrated, attr, None)
            if estimator is not None:
                return estimator

    for attr in ("estimator", "base_estimator"):
        estimator = getattr(model, attr, None)
        if estimator is not None:
            return estimator

    return model


def align_categorical_dtypes(
    df_target: pd.DataFrame,
    df_reference: pd.DataFrame,
    columns: List[str] | None = None,
    copy: bool = True,
) -> pd.DataFrame:
    """Align categorical/boolean dtypes in df_target to match df_reference."""
    if df_target is None:
        return df_target
    if columns is None:
        columns = df_target.columns
    aligned = df_target.copy() if copy else df_target
    for col in columns:
        if col not in aligned.columns or col not in df_reference.columns:
            continue
        ref_dtype = df_reference[col].dtype
        if isinstance(ref_dtype, pd.CategoricalDtype):
            aligned[col] = aligned[col].astype(ref_dtype)
        elif is_bool_dtype(ref_dtype) and not is_bool_dtype(aligned[col].dtype):
            aligned[col] = aligned[col].astype(ref_dtype)
    return aligned


def sanitize_categorical_predict_input(
    df: pd.DataFrame, model, missing_category="NaN"
) -> pd.DataFrame:
    """Sanitize categorical prediction input for models with explicit cat feature indices.

    Currently normalizes CatBoost categorical columns so missing values do not crash
    prediction callbacks.
    """
    if not isinstance(df, pd.DataFrame):
        return df

    if not safe_isinstance(
        model, "catboost.core.CatBoost", "CatBoostClassifier", "CatBoostRegressor"
    ):
        return df

    get_cat_feature_indices = getattr(model, "get_cat_feature_indices", None)
    if not callable(get_cat_feature_indices):
        return df

    cat_feature_indices = list(get_cat_feature_indices() or [])
    if not cat_feature_indices:
        return df

    cat_cols = [df.columns[i] for i in cat_feature_indices if 0 <= i < len(df.columns)]
    if not cat_cols:
        return df

    sanitized = df.copy()

    def _normalize_cat_value(value):
        if pd.isna(value):
            return missing_category
        if isinstance(value, (float, np.floating)):
            return str(value)
        return value

    for col in cat_cols:
        sanitized[col] = sanitized[col].astype("object").map(_normalize_cat_value)

    return sanitized


def sorted_categorical_values(values):
    """Sort categorical values safely when types are mixed.

    Keeps original values but orders deterministically:
    booleans, then numbers, then other values by type/value string.
    """

    def _is_na(value):
        try:
            return bool(pd.isna(value))
        except Exception:
            return False

    def _sort_key(value):
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, bool):
            return (0, int(value))
        if isinstance(value, (int, float)) and not _is_na(value):
            return (1, float(value))
        return (2, type(value).__name__, str(value))

    clean_values = [value for value in values if not _is_na(value)]
    return sorted(clean_values, key=_sort_key)


def guess_shap(model):
    """guesses which SHAP explainer to use for a particular model, based
    on str(type(model)). Returns 'tree' for tree based models such as
    RandomForest and XGBoost that need shap.TreeExplainer, and 'linear'
    for linear models such as LinearRegression or Elasticnet that can use
    shap.LinearExplainer.

    Args:
        model: a fitted (sklearn-compatible) model

    Returns:
        str: {'tree', 'linear', None}
    """
    model = unwrap_calibrated_classifier(model)

    tree_models = [
        "RandomForestClassifier",
        "RandomForestRegressor",
        "DecisionTreeClassifier",
        "DecisionTreeRegressor",
        "ExtraTreesClassifier",
        "ExtraTreesRegressor",
        "GradientBoostingClassifier",
        "GradientBoostingRegressor",
        "HistGradientBoostingClassifier",
        "HistGradientBoostingRegressor",
        "XGBClassifier",
        "XGBRegressor",
        "LGBMClassifier",
        "LGBMRegressor",
        "CatBoostClassifier",
        "CatBoostRegressor",
        "NGClassifier",
        "NGBRegressor",
        "GBTClassifier",
        " GBTRegressor",
        "IsolationForest",
    ]
    linear_models = [
        "LinearRegression",
        "LogisticRegression",
        "Ridge",
        "Lasso",
        "ElasticNet",
        "SGDClassifier",
    ]

    skorch_models = [
        "skorch.net.NeuralNet",
        "skorch.regressor.NeuralNetRegressor",
        "skorch.classifier.NeuralNetClassifier",
        "skorch.classifier.NeuralNetBinaryClassifier",
    ]

    for tree_model in tree_models:
        if str(type(model)).endswith(tree_model + "'>"):
            return "tree"

    for lin_model in linear_models:
        if str(type(model)).endswith(lin_model + "'>"):
            return "linear"

    for skorch_model in skorch_models:
        if str(type(model)).endswith(skorch_model + "'>"):
            return "skorch"

    return None


def mape_score(y_true, y_pred):
    """returns Mean Absolute Percentage Error"""
    epsilon = np.finfo(np.float64).eps
    absolute_percentage_errors = np.abs(y_pred - y_true) / np.maximum(
        np.abs(y_true), epsilon
    )
    mape = np.average(absolute_percentage_errors)
    return mape


def parse_cats(X, cats, sep: str = "_"):
    """parse onehot encoded columns to a onehot_dict.
    - cats can be a dict where you enumerate each individual onehot encoded column belonging to
        each categorical feature, e.g. cats={
                    'Sex':['Sex_female', 'Sex_male'],
                    'Deck':['Deck_A', 'Deck_B', 'Deck_C', 'Deck_nan']
                    }
    - if you encode your categorical features as Cat_Label, you can pass a list of the
        original feature names: cats=["Sex", "Deck"]
    - or a combination of the two: cats = ["Sex", {'Deck':['Deck_A', 'Deck_B', 'Deck_C', 'Deck_nan']}]

    Asserts that all columns can be found in X.columns.
    Asserts that all columns are only passed once.
    """
    all_cols = X.columns
    onehot_cols = []
    onehot_dict = {}

    col_counter = Counter()

    if isinstance(cats, dict):
        for k, v in cats.items():
            assert set(
                v
            ).issubset(
                set(all_cols)
            ), f"These cats columns for {k} could not be found in X.columns: {set(v) - set(all_cols)}!"
            col_counter.update(v)
        onehot_dict = cats
    elif isinstance(cats, list):
        for cat in cats:
            if isinstance(cat, str):
                onehot_dict[cat] = [c for c in all_cols if c.startswith(cat + sep)]
                col_counter.update(onehot_dict[cat])
            if isinstance(cat, dict):
                for k, v in cat.items():
                    assert set(
                        v
                    ).issubset(
                        set(all_cols)
                    ), f"These cats columns for {k} could not be found in X.columns: {set(v) - set(all_cols)}!"
                    col_counter.update(v)
                    onehot_dict[k] = v
    multi_cols = [v for v, c in col_counter.most_common() if c > 1]
    assert not multi_cols, (
        f"The following columns seem to have been passed to cats multiple times: {multi_cols}. "
        "Please make sure that each onehot encoded column is only assigned to one cat column!"
    )
    assert not set(onehot_dict.keys()) & set(all_cols), (
        f"These new cats columns are already in X.columns: {list(set(onehot_dict.keys()) & set(all_cols))}! "
        "Please select a different name for your new cats columns!"
    )
    for col, count in col_counter.most_common():
        assert is_binary_like_onehot_column(
            X[col]
        ), f"{col} is not a onehot encoded column (i.e. has values other than 0, 1)!"
    onehot_cols = list(onehot_dict.keys())
    for col in [col for col in all_cols if col not in col_counter.keys()]:
        onehot_dict[col] = [col]
    return onehot_cols, onehot_dict


def get_encoded_and_regular_cols(cols, onehot_dict):
    """return a list of onehot encoded cols and a list of remainder cols."""
    encoded_cols = []
    for enc_cols in onehot_dict.values():
        if len(enc_cols) > 1:
            encoded_cols.extend(enc_cols)
    regular_cols = [col for col in cols if col not in encoded_cols]
    return encoded_cols, regular_cols


def split_pipeline(pipeline: Pipeline, verbose: int = 1):
    """Splits a sklearn or imblearn pipeline into a transformer pipeline and the final estimator.

    Args:
        pipeline (sklearn.Pipeline): a fitted pipeline with an estimator
            with .predict method as the last step.
        verbose: verbose output

    Returns:
        transformer_pipeline, estimator

    """
    if not safe_isinstance(
        pipeline, "sklearn.pipeline.Pipeline", "imblearn.pipeline.Pipeline"
    ):
        raise ValueError(
            f"pipeline should either be an sklearn or an imblearn pipeline, but you passed {pipeline}!"
        )

    assert hasattr(pipeline.steps[-1][1], "predict"), (
        "When passing an sklearn.Pipeline, the last step of the pipeline should be a model, "
        f"but {pipeline.steps[-1][1]} does not have a .predict() function!"
    )

    if verbose:
        logger.info("Splitting pipeline...")
        skipped_transforms = [
            name
            for name, transform in pipeline.steps[:-1]
            if not hasattr(transform, "transform")
        ]
        if skipped_transforms:
            logger.info(
                "Skipping steps that lack a .transform() method: %s",
                ", ".join(skipped_transforms),
            )

    transform_steps = [
        (name, transform)
        for name, transform in pipeline.steps[:-1]
        if hasattr(transform, "transform")
    ]
    if transform_steps:
        transformer_pipeline = Pipeline(transform_steps)
    else:
        from sklearn.preprocessing import FunctionTransformer

        transformer_pipeline = Pipeline([("identity", FunctionTransformer())])
    estimator = pipeline.steps[-1][1]

    return transformer_pipeline, estimator


def is_binary_like_onehot_column(series: pd.Series) -> bool:
    """Return whether a series behaves like one-hot encoded values.

    Accepts strict {0, 1} columns and binary-like variants with two unique numeric
    values (e.g. after a scaler was applied to one-hot features in a pipeline).
    """
    numeric_unique = np.unique(pd.Series(series).dropna().astype(float))
    return set(numeric_unique).issubset({0.0, 1.0}) or len(numeric_unique) <= 2


def infer_cats_from_transformed_X(
    X_transformed: pd.DataFrame, original_columns: List[str], sep: str = "_"
) -> dict:
    """Infer one-hot groupings from transformed pipeline column names.

    Uses original pre-transform feature names to detect expanded one-hot columns.
    Only groups columns when:
    1) more than one transformed column matches an original feature, and
    2) all matched columns are binary-like.
    """
    inferred = {}
    transformed_columns = list(X_transformed.columns)

    for original_col in original_columns:
        matched_cols = []
        for transformed_col in transformed_columns:
            tail = transformed_col.split("__")[-1]
            if tail == original_col or tail.startswith(f"{original_col}{sep}"):
                matched_cols.append(transformed_col)

        if len(matched_cols) <= 1:
            continue
        if not all(
            is_binary_like_onehot_column(X_transformed[col]) for col in matched_cols
        ):
            continue

        first_col = matched_cols[0]
        marker = f"{original_col}{sep}"
        if marker in first_col:
            group_name = first_col[: first_col.index(marker) + len(original_col)]
        else:
            group_name = original_col
        inferred[group_name] = matched_cols

    return inferred


def rename_pipeline_columns(
    columns: List[str],
    strip_pipeline_prefix: bool = False,
    feature_name_fn: Optional[Callable[[str], str]] = None,
    verbose: int = 1,
) -> List[str]:
    """Optionally transform pipeline output feature names."""
    if feature_name_fn is not None:
        renamed = [feature_name_fn(col) for col in columns]
    elif strip_pipeline_prefix:
        renamed = [col.split("__", 1)[1] if "__" in col else col for col in columns]
    else:
        return columns

    if len(set(renamed)) == len(renamed):
        return renamed

    deduped = []
    counts = Counter()
    for col in renamed:
        counts[col] += 1
        if counts[col] == 1:
            deduped.append(col)
        else:
            deduped.append(f"{col}__{counts[col]}")

    if verbose:
        logger.warning(
            "Feature name transformation produced duplicate columns; appended numeric suffixes to keep names unique."
        )
    return deduped


def get_transformed_X(
    transformer_pipeline: Pipeline,
    X: pd.DataFrame,
    verbose: int = 1,
    strip_pipeline_prefix: bool = False,
    feature_name_fn: Optional[Callable[[str], str]] = None,
):
    """takes a transformer_pipeline (all the steps except the final Estimator of an sklearn or imblearn Pipeline)
    and uses it to transform input DataFrame X and returns a transformed DataFrame.

    If all steps have .get_feature_names_out() method implemented, then uses that to assign column names.
    If not and the number of columns stays equal, simply assign the old column names.
    Else assign column names 'col1', 'col2', etc...

    Args:
        transformer_pipeline: transformer part of a Pipeline, generated with split_pipeline()
        X: input DataFrame to be transformed

    Returns:
        X_transformed: transformed dataframe with column names assigned
    """
    X_transformed = transformer_pipeline.transform(X)

    if hasattr(transformer_pipeline, "get_feature_names_out"):
        try:
            columns = list(transformer_pipeline.get_feature_names_out())
            if len(columns) != X_transformed.shape[1]:
                raise ValueError(
                    f"len(pipeline[:-1].get_feature_names_out())={len(columns)} does"
                    f" not equal X_transformed.shape[1]={X_transformed.shape[1]}!"
                )
            columns = rename_pipeline_columns(
                columns,
                strip_pipeline_prefix=strip_pipeline_prefix,
                feature_name_fn=feature_name_fn,
                verbose=verbose,
            )
            return pd.DataFrame(X_transformed, columns=columns, index=X.index)
        except Exception as e:
            if verbose:
                logger.warning(
                    "Failed to retrieve new column names from transformer_pipeline.get_feature_names_out(): %s",
                    e,
                )

    if X_transformed.shape == X.values.shape:
        if verbose:
            logger.info(
                "Transformer pipeline outputs same number of columns; using X.columns (%s). "
                "Ensure your pipeline does not add/remove/reorder columns.",
                X.columns.tolist(),
            )
        try:
            for i, pipe in enumerate(transformer_pipeline):
                if hasattr(pipe, "n_features_in_"):
                    assert pipe.n_features_in_ == len(X.columns)
            return pd.DataFrame(X_transformed, columns=X.columns, index=X.index)
        except Exception as e:
            logger.warning(
                ".n_features_in_ did not match len(X.columns)=%s for pipeline step %s: %s. Error: %s",
                len(X.columns),
                i,
                pipe,
                e,
            )

    if verbose:
        logger.warning(
            "Pipeline does not have a functioning .get_feature_names_out() method, "
            "nor do all pipeline steps return the same number of columns as input, "
            "so assigning column names 'col1', 'col2', etc."
        )
    columns = [f"col{i + 1}" for i in range(X_transformed.shape[1])]

    return pd.DataFrame(X_transformed, columns=columns, index=X.index)


def build_pipeline_extraction_warning(error: Exception) -> str:
    """Build a user-facing warning when pipeline extraction fails."""
    return (
        "Warning: Failed to extract a data transformer with column names and final "
        "model from the Pipeline. So set shap='kernel' to use the (slower and "
        "approximate) model-agnostic shap.KernelExplainer instead. "
        "If possible, ensure pipeline transformers implement get_feature_names_out(), "
        "and verify pipeline transform() can run on the provided X/X_background. "
        f"Error: {error}"
    )


def retrieve_onehot_value(
    X, encoded_col, onehot_cols, not_encoded="NOT_ENCODED", sep="_"
):
    """
    Reverses a onehot encoding.

    i.e. Finds the column name starting with encoded_col_ that has a value of 1.
        if no such column exists (e.g. they are all 0), then return 'NOT_ENCODED'

    Args:
        X (pd.DataFrame): dataframe from which to retrieve onehot column
        encoded_col (str): Name of the encoded col (e.g. 'Sex')
        onehot_cols (list): list of onehot cols, e.g. ['Sex_female', 'Sex_male']
        sep (str): seperator between category and value, e.g. '_' for Sex_Male.

    Returns:
        pd.Series with categories. If no category is found, coded as "NOT_ENCODED".
    """
    feature_value = np.argmax(X[onehot_cols].values, axis=1)

    # if not a single 1 then encoded feature must have been dropped
    feature_value[np.max(X[onehot_cols].values, axis=1) == 0] = -1
    if all([col.startswith(col + "_") for col in onehot_cols]):
        mapping = {-1: encoded_col + not_encoded}
    else:
        mapping = {-1: not_encoded}

    mapping.update({i: col for i, col in enumerate(onehot_cols)})
    return pd.Series(feature_value).map(mapping)


def merge_categorical_columns(
    X, onehot_dict=None, cols=None, not_encoded_dict=None, sep="_", drop_regular=False
):
    cat_pieces = []

    for col_name, col_list in onehot_dict.items():
        if len(col_list) > 1:
            merged_col = retrieve_onehot_value(
                X,
                col_name,
                col_list,
                not_encoded_dict.get(col_name, "NOT_ENCODED"),
                sep,
            ).astype("category")
            cat_pieces.append(pd.DataFrame({col_name: merged_col}))
        else:
            if not drop_regular:
                if isinstance(X[col_name].dtype, pd.CategoricalDtype):
                    cat_pieces.append(
                        pd.DataFrame({col_name: pd.Categorical(X[col_name])})
                    )
                else:
                    cat_pieces.append(pd.DataFrame({col_name: X[col_name].values}))

    if cat_pieces:
        X_cats = pd.concat(cat_pieces, axis=1)
    else:
        X_cats = pd.DataFrame()

    if cols:
        X_cats = X_cats[cols]

    return X_cats


def matching_cols(cols1, cols2):
    """returns True if cols1 and cols2 match."""
    if isinstance(cols1, pd.DataFrame):
        cols1 = cols1.columns
    if isinstance(cols2, pd.DataFrame):
        cols2 = cols2.columns
    if len(cols1) != len(cols2):
        return False
    if (pd.Index(cols1) == pd.Index(cols2)).all():
        return True
    return False


def remove_cat_names(X_cats, onehot_dict, onehot_missing_dict=None):
    """removes the leading category names in the onehotencoded columns.
    Turning e.g 'Sex_male' into 'male', etc"""
    X_cats = X_cats.copy()
    for cat, cols in onehot_dict.items():
        if len(cols) > 1:
            mapping = {
                c: (c[len(cat) + 1 :] if c.startswith(cat + "_") else c) for c in cols
            }
            if onehot_missing_dict:
                mapping.update({onehot_missing_dict[cat]: onehot_missing_dict[cat]})
            X_cats[cat] = X_cats[cat].map(mapping, na_action="ignore")
    return X_cats


def X_cats_to_X(X_cats, onehot_dict, X_columns, sep="_"):
    """
    re-onehotencodes a dataframe where onehotencoded columns had previously
    been merged with merge_categorical_columns(...)

    Args:
        X_cats (pd.DataFrame): dataframe with merged categorical columns cats
        onehot_dict (dict): dict of features with lists for onehot-encoded variables,
             e.g. {'Fare': ['Fare'], 'Sex' : ['Sex_male', 'Sex_Female']}
        X_columns: list of columns of original dataframe

    Returns:
        pd.DataFrame: dataframe X with same encoding as original
    """
    non_cat_cols = [col for col in X_cats.columns if col in X_columns]
    X_new = X_cats[non_cat_cols].copy()
    for cat, cols in onehot_dict.items():
        if len(cols) > 1:
            for col in cols:
                X_new[col] = (X_cats[cat] == col).astype(np.int8)
    return X_new[X_columns]


def merge_categorical_shap_values(shap_df, onehot_dict=None, output_cols=None):
    """
    Returns a new feature new shap values np.array
    where the shap values of onehotencoded categorical features have been
    added up.

    Args:
        shap_df(pd.DataFrame): dataframe of shap values with appropriate column names
        onehot_dict (dict): dict of features with lists for onehot-encoded variables,
             e.g. {'Fare': ['Fare'], 'Sex' : ['Sex_male', 'Sex_Female']}

    Returns:
        pd.DataFrame
    """
    onehot_cols = []
    for col_name, col_list in onehot_dict.items():
        if len(col_list) > 1:
            shap_df[col_name] = shap_df[col_list].sum(axis=1)
            onehot_cols.append(col_name)
    if output_cols is not None:
        return shap_df[output_cols]
    return shap_df[onehot_cols]


def merge_categorical_shap_interaction_values(
    shap_interaction_values, old_columns, new_columns, onehot_dict
):
    """
    Returns a 3d numpy array shap_interaction_values where the onehot-encoded
    categorical columns have been added up together.

    Warning:
        Column names in new_columns that are not found in old_columns are
        assumed to be categorical feature names.

    Args:
        shap_interaction_values (np.ndarray): shap_interaction output from
            e.g. shap.TreeExplainer(X).shap_interaction_values().
        old_columns (list of str): list of column names with onehotencodings,
            e.g. ["Age", "Sex_Male", "Sex_Female"]
        new_columns (list of str): list of column names without onehotencodings,
            e.g. ["Age", "Sex"]
        onehot_dict (dict): dict of features with lists for onehot-encoded variables,
             e.g. {'Fare': ['Fare'], 'Sex' : ['Sex_male', 'Sex_Female']}

    Returns:
        np.ndarray: shap_interaction values with all the onehot-encoded features
            summed together.
    """

    if isinstance(old_columns, pd.DataFrame):
        old_columns = old_columns.columns
    if isinstance(new_columns, pd.DataFrame):
        new_columns = new_columns.columns
    old_columns = pd.Index(old_columns)
    new_columns = pd.Index(new_columns)

    siv = np.zeros(
        (shap_interaction_values.shape[0], len(new_columns), len(new_columns))
    )

    # note: given the for loops here, this code could probably be optimized.
    #       But only runs once anyway...
    for new_col1 in new_columns:
        for new_col2 in new_columns:
            newcol_idx1 = new_columns.get_loc(new_col1)
            newcol_idx2 = new_columns.get_loc(new_col2)
            oldcol_idxs1 = [old_columns.get_loc(col) for col in onehot_dict[new_col1]]
            oldcol_idxs2 = [old_columns.get_loc(col) for col in onehot_dict[new_col2]]
            siv[:, newcol_idx1, newcol_idx2] = shap_interaction_values[
                :, oldcol_idxs1, :
            ][:, :, oldcol_idxs2].sum(axis=(1, 2))
    return siv


def make_one_vs_all_scorer(metric, pos_label=1, greater_is_better=True):
    """
    Returns a binary one vs all scorer for a single class('pos_label') of a
    multiclass classifier metric.

    Args:
        metric (function): classification metric of the form metric(y_true, y_pred)
        pos_label (int): index of the positive label. Defaults to 1.
        greater_is_better (bool): does a higher metric correspond to a better model.
            Defaults to True.

    Returns:
        a binary sklearn-compatible scorer function.
    """

    def one_vs_all_metric(metric, pos_label, y_true, y_pred):
        return metric((y_true == pos_label).astype(int), y_pred[:, pos_label])

    partial_metric = partial(one_vs_all_metric, metric, pos_label)
    sign = 1 if greater_is_better else -1

    def _scorer(clf, X, y):
        warnings.filterwarnings("ignore", category=UserWarning)
        n_labels = len(clf.classes_) if hasattr(clf, "classes_") else None
        y_pred = _predict_proba_with_fallback(clf, X, n_labels=n_labels)
        warnings.filterwarnings("default", category=UserWarning)
        y_pred = _ensure_numeric_predictions(y_pred)
        y_pred = np.asarray(y_pred)
        score = sign * partial_metric(y, y_pred)
        return score

    return _scorer


def permutation_importances(
    model,
    X,
    y,
    metric,
    onehot_dict=None,
    greater_is_better=True,
    needs_proba=False,
    pos_label=1,
    n_repeats=1,
    n_jobs=None,
    sort=True,
    pass_nparray=False,
    verbose=0,
):
    """
    adapted from rfpimp package, returns permutation importances, optionally grouping
    onehot-encoded features together.

    Args:
        model: fitted model for which you'd like to calculate importances.
        X (pd.DataFrame): dataframe of features
        y (pd.Series): series of targets
        metric: metric to be evaluated (usually R2 for regression, roc_auc for
            classification)
        onehot_dict (dict): dict of features with lists for onehot-encoded variables,
             e.g. {'Fare': ['Fare'], 'Sex' : ['Sex_male', 'Sex_Female']}
        greater_is_better (bool): indicates whether the higher score on the metric
            indicates a better model.
        needs_proba (bool): does the metric need a classification probability
            or direct prediction?
        pos_label (int): for classification, the label to use a positive label.
            Defaults to 1.
        n_repeats (int): number of time to permute each column to take the average score.
            Defaults to 1.
        n_jobs (int): number of jobs for joblib parallel. Defaults to None.
        sort (bool): sort the output from highest importances to lowest.
        pass_nparray (bool, optional): instead of the X pass X.values to model.
            This is useful for skorch models that do not accepts dataframes.
        verbose (int): set to 1 to print output for debugging. Defaults to 0.
    """
    X = X.copy()

    if onehot_dict is None:
        onehot_dict = {col: [col] for col in X.columns}

    if isinstance(metric, str):
        scorer = _safe_make_scorer(
            metric,
            greater_is_better=greater_is_better,
            response_method="predict_proba" if needs_proba else "predict",
        )
    elif not needs_proba or pos_label is None:
        scorer = _safe_make_scorer(
            metric, greater_is_better=greater_is_better, response_method="predict"
        )
    else:
        scorer = make_one_vs_all_scorer(metric, pos_label, greater_is_better)
    if pass_nparray:
        baseline = scorer(model, X.values, y.values)
    else:
        baseline = scorer(model, X, y)

    def _permutation_importance(
        model,
        X,
        y,
        scorer,
        col_name,
        col_list,
        baseline,
        n_repeats=1,
        pass_nparray=False,
    ):
        X = X.copy()
        scores = []
        for i in range(n_repeats):
            old_cols = X[col_list].copy()
            permuted = X[col_list].sample(frac=1, replace=False)
            permuted.index = X.index
            X[col_list] = permuted
            if pass_nparray:
                scores.append(scorer(model, X.values, y.values))
            else:
                scores.append(scorer(model, X, y))

            X[col_list] = old_cols
        return col_name, np.mean(scores)

    scores = Parallel(n_jobs=n_jobs)(
        delayed(_permutation_importance)(
            model, X, y, scorer, col_name, col_list, baseline, n_repeats, pass_nparray
        )
        for col_name, col_list in onehot_dict.items()
    )

    importances_df = pd.DataFrame(scores, columns=["Feature", "Score"])
    importances_df["Importance"] = baseline - importances_df["Score"]
    importances_df = importances_df[["Feature", "Importance", "Score"]]
    if sort:
        return importances_df.sort_values("Importance", ascending=False)
    else:
        return importances_df


def cv_permutation_importances(
    model,
    X,
    y,
    metric,
    onehot_dict=None,
    greater_is_better=True,
    needs_proba=False,
    pos_label=None,
    cv=None,
    n_repeats=1,
    n_jobs=None,
    pass_nparray=False,
    verbose=0,
):
    """
    Returns the permutation importances averages over `cv` cross-validated folds.

    Args:
        model: fitted model for which you'd like to calculate importances.
        X (pd.DataFrame): dataframe of features
        y (pd.Series): series of targets
        metric: metric to be evaluated (usually R2 for regression, roc_auc for
            classification)
        onehot_dict (dict): dict of features with lists for onehot-encoded variables,
             e.g. {'Fare': ['Fare'], 'Sex' : ['Sex_male', 'Sex_Female']}
        greater_is_better (bool): indicates whether the higher score on the metric
            indicates a better model.
        needs_proba (bool): does the metric need a classification probability
            or direct prediction?
        pos_label (int): for classification, the label to use a positive label.
            Defaults to 1.
        cv (int): number of cross-validation folds to apply.
        sort (bool): sort the output from highest importances to lowest.
        pass_nparray (bool, optional): instead of the X pass X.values to model.
            This is useful for skorch models that do not accepts dataframes.
        verbose (int): set to 1 to print output for debugging. Defaults to 0.
    """
    if cv is None:
        return permutation_importances(
            model,
            X,
            y,
            metric,
            onehot_dict,
            greater_is_better=greater_is_better,
            needs_proba=needs_proba,
            pos_label=pos_label,
            n_repeats=n_repeats,
            n_jobs=n_jobs,
            sort=False,
            pass_nparray=pass_nparray,
            verbose=verbose,
        )

    if needs_proba:
        skf = StratifiedKFold(n_splits=cv, random_state=None, shuffle=False)
        splitter = skf.split(X, y)
    else:
        kf = KFold(n_splits=cv, random_state=None, shuffle=False)
        splitter = kf.split(X)

    model = clone(model)
    for i, (train_index, test_index) in enumerate(splitter):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model.fit(X_train, y_train)

        imp = permutation_importances(
            model,
            X_test,
            y_test,
            metric,
            onehot_dict,
            greater_is_better=greater_is_better,
            needs_proba=needs_proba,
            pos_label=pos_label,
            n_repeats=n_repeats,
            n_jobs=n_jobs,
            sort=False,
            pass_nparray=pass_nparray,
            verbose=verbose,
        )
        if i == 0:
            imps = imp[["Feature", "Importance"]]
        else:
            imps = imps.merge(
                imp[["Feature", "Importance"]],
                on="Feature",
                suffixes=("", "_" + str(i)),
            )

    return (
        imps.set_index("Feature")
        .mean(axis=1)
        .to_frame()
        .rename(columns={0: "Importance"})
        .sort_values("Importance", ascending=False)
        .reset_index()
    )


def get_mean_absolute_shap_df(columns, shap_values, onehot_dict=None):
    """
    Returns a dataframe with the mean absolute shap values for each feature.

    Args:
        columns (list of str): list of column names
        shap_values (np.ndarray): 2d array of SHAP values
        onehot_dict (dict): dict of features with lists for onehot-encoded variables,
             e.g. {'Fare': ['Fare'], 'Sex' : ['Sex_male', 'Sex_Female']}

    Returns:
        pd.DataFrame with columns 'Feature' and 'MEAN_ABS_SHAP'.
    """
    if onehot_dict is None:
        onehot_dict = {col: [col] for col in columns}
    columns = pd.Index(columns)
    shap_abs_mean_dict = {}
    for col_name, col_list in onehot_dict.items():
        shap_abs_mean_dict[col_name] = np.absolute(
            shap_values[:, [columns.get_loc(col) for col in col_list]].sum(axis=1)
        ).mean()

    shap_df = (
        pd.DataFrame(
            {
                "Feature": list(shap_abs_mean_dict.keys()),
                "MEAN_ABS_SHAP": list(shap_abs_mean_dict.values()),
            }
        )
        .sort_values("MEAN_ABS_SHAP", ascending=False)
        .reset_index(drop=True)
    )
    return shap_df


def get_grid_points(array, n_grid_points=10, min_percentage=0, max_percentage=100):
    """seperates a numerical array into a number of grid points. Helper function
    for get_pdp_df.

    Args:
        array (np.array): array
        n_grid_points (int, optional): number of points to divide array in.
            Defaults to 10.
        min_percentage (int, optional): Minimum percentage to start at,
            ignoring outliers. Defaults to 0.
        max_percentage (int, optional): Maximum percentage to reach, ignoring
            outliers. Defaults to 100.

    Raises:
        ValueError: [description]

    Returns:
        np.array
    """

    if isinstance(array, pd.Series):
        array = array.values
    else:
        array = np.array(array)
    if not is_numeric_dtype(array):
        raise ValueError("array should be a numeric dtype!")

    percentile_grids = np.linspace(
        start=min_percentage, stop=max_percentage, num=n_grid_points
    )
    value_grids = np.percentile(array, percentile_grids)
    return value_grids


def get_pdp_df(
    model,
    X_sample: pd.DataFrame,
    feature: Union[str, List],
    pos_label=1,
    n_grid_points: int = 10,
    min_percentage: int = 0,
    max_percentage: int = 100,
    multiclass: bool = False,
    grid_values: List = None,
    is_classifier: bool = False,
    cast_to_float32: bool = False,
):
    """Returns a dataframe with partial dependence for every row in X_sample for a number of feature values

    Args:
        model (): sklearn compatible model to generate pdp for
        X_sample (pd.DataFrame): X to generate pdp for
        feature (Union[str, List]): Feature to generate pdp for. Either the
            name of a column in X_sample, or a list of onehot-encoded columns.
        pos_label (int, optional): for classifier model, which class to use
            as the positive class. Defaults to 1.
        n_grid_points (int, optional): For numeric features: number of grid points
            to divide the x axis by. Defaults to 10.
        min_percentage (int, optional): For numeric features: minimum percentage of
            samples to start x axis by. If large than 0 a form of winsorizing the
            x axis. Defaults to 0.
        max_percentage (int, optional): For numeric features: maximum percentage of
            samples to end x axis by. If smaller than 100 a form of winsorizing the
            x axis. Defaults to 100.
        multiclass (bool, optional): for classifier models, return a list of dataframes,
            one for each predicted label.
        grid_values (list, optional): list of grid values. Default to None, in which
            case it will be inferred from X_sample.
        is_classifier (bool, optional): model is a classifier with a pred_probas method.
        cast_to_float32 (bool, optional): cast model input to np.float32 (necessary for
            skorch models)
    """

    def _model_input(data):
        if isinstance(data, pd.DataFrame):
            data = sanitize_categorical_predict_input(data, model)

        if cast_to_float32:
            if isinstance(data, pd.DataFrame):
                return data.values.astype("float32")
            return np.asarray(data, dtype="float32")
        if (
            isinstance(data, pd.DataFrame)
            and not safe_isinstance(
                model,
                "catboost.core.CatBoost",
                "CatBoostClassifier",
                "CatBoostRegressor",
            )
            and not safe_isinstance(
                model, "sklearn.pipeline.Pipeline", "imblearn.pipeline.Pipeline"
            )
            and not hasattr(model, "feature_names_in_")
        ):
            return data.values
        return data

    if grid_values is None:
        if isinstance(feature, str):
            if not is_numeric_dtype(X_sample[feature]):
                grid_values = sorted_categorical_values(
                    X_sample[feature].unique().tolist()
                )
            else:
                grid_values = get_grid_points(
                    X_sample[feature],
                    n_grid_points=n_grid_points,
                    min_percentage=min_percentage,
                    max_percentage=max_percentage,
                ).tolist()
        elif isinstance(feature, list):
            grid_values = feature
        else:
            raise ValueError(
                "feature should either be a column name (str), "
                "or a list of onehot-encoded columns!"
            )

    if is_classifier:
        first_row = _model_input(X_sample.iloc[[0]])
        warnings.filterwarnings("ignore", category=UserWarning)
        class_count = len(model.classes_) if hasattr(model, "classes_") else None
        n_labels = _predict_proba_with_fallback(
            model, first_row, n_labels=class_count
        ).shape[1]
        warnings.filterwarnings("default", category=UserWarning)
        if multiclass:
            pdp_dfs = [pd.DataFrame() for i in range(n_labels)]
        else:
            pdp_df = pd.DataFrame()
    else:
        pdp_df = pd.DataFrame()

    def _coerce_value(value, dtype):
        if isinstance(dtype, pd.CategoricalDtype):
            return value
        if is_bool_dtype(dtype):
            return bool(value)
        return value

    for grid_value in grid_values:
        dtemp = X_sample.copy()
        if isinstance(feature, list):
            if grid_value in X_sample.columns:
                assert set(X_sample[grid_value].unique()).issubset({0, 1}), (
                    f"{grid_values} When passing a list of features these have to be onehotencoded!"
                    f"But X_sample['{grid_value}'].unique()=={list(set(X_sample[grid_value].unique()))}"
                )
            for col in feature:
                dtemp[col] = _coerce_value(col == grid_value, X_sample[col].dtype)
        else:
            dtemp[[feature]] = _coerce_value(grid_value, X_sample[feature].dtype)
        align_cols = feature if isinstance(feature, list) else [feature]
        dtemp = align_categorical_dtypes(
            dtemp, X_sample, columns=align_cols, copy=False
        )
        if is_classifier:
            dtemp_model = _model_input(dtemp)
            pred_probas = _predict_proba_with_fallback(
                model,
                dtemp_model,
                n_labels=n_labels,
            ).squeeze()
            if multiclass:
                for i in range(n_labels):
                    pdp_dfs[i][grid_value] = pred_probas[:, i]
            else:
                pdp_df[grid_value] = pred_probas[:, pos_label]
        else:
            dtemp_model = _model_input(dtemp)
            preds_raw = model.predict(dtemp_model)
            preds_raw = _ensure_numeric_predictions(preds_raw)
            preds = np.asarray(preds_raw).squeeze()
            pdp_df[grid_value] = preds
    if multiclass:
        return pdp_dfs
    else:
        return pdp_df


def get_precision_df(
    pred_probas, y_true, bin_size=None, quantiles=None, round=3, pos_label=1
):
    """
    returns a pd.DataFrame with the predicted probabilities and
    the observed frequency per bin_size or quantile.

    If pred_probas has one dimension (i.e. only probabilities of positive class)
    only returns a single precision. If pred_probas containts probabilities for
    every class (typically a multiclass classifier), also returns precision
    for every class in every bin.

    Args:
        pred_probas (np.ndarray): result of model.predict_proba(X). Can either
            be probabilities of a single class or multiple classes.
        y_true (np.ndarray): array of true class labels.
        bin_size (float): bin sizes to bin by. E.g. 0.1 to bin all prediction
            between 0 and 0.1, between 0.1 and 0.2, etc. If setting bin_size
            you cannot set quantiles.
        quantiles (int): number of quantiles to divide pred_probas in.
            e.g. if quantiles=4, set bins such that the lowest 25% of pred_probas
            go into first bin, next 25% go in second bin, etc. Each bin will
            have (approximatly the same amount of observations). If setting
            quantiles you cannot set bin_size.
        round (int): the number of figures to round the output by. Defaults to 3.
        pos_label (int): the label of the positive class. Defaults to 1.

    Returns:
        pd.DataFrame with columns ['p_min', 'p_max', 'p_avg', 'bin_width',
        'precision', 'count']
    """
    if bin_size is None and quantiles is None:
        bin_size = 0.1

    assert (bin_size is not None and quantiles is None) or (
        bin_size is None and quantiles is not None
    ), "either only pass bin_size or only pass quantiles!"

    if len(pred_probas.shape) == 2:
        # in case the full binary classifier pred_proba is passed,
        # we only select the probability of the positive class
        predictions_df = pd.DataFrame(
            {"pred_proba": pred_probas[:, pos_label], "target": y_true}
        )
        n_classes = pred_probas.shape[1]
    else:
        predictions_df = pd.DataFrame({"pred_proba": pred_probas, "target": y_true})
        n_classes = 1

    predictions_df = predictions_df.sort_values("pred_proba")

    # define a placeholder df:
    columns = ["p_min", "p_max", "p_avg", "bin_width", "precision", "count"]
    if n_classes > 1:
        for i in range(n_classes):
            columns.append("precision_" + str(i))

    precision_df = pd.DataFrame(columns=columns)

    if bin_size:
        thresholds = np.arange(0.0, 1.0, bin_size).tolist()
        # loop through prediction intervals, and compute
        for bin_min, bin_max in zip(thresholds, thresholds[1:] + [1.0]):
            if bin_min != bin_max:
                new_row_dict = {
                    "p_min": [bin_min],
                    "p_max": [bin_max],
                    "p_avg": [bin_min + (bin_max - bin_min) / 2.0],
                    "bin_width": [bin_max - bin_min],
                }

                if bin_min == 0.0:
                    new_row_dict["p_avg"] = predictions_df[
                        (predictions_df.pred_proba >= bin_min)
                        & (predictions_df.pred_proba <= bin_max)
                    ]["pred_proba"].mean()
                    new_row_dict["precision"] = (
                        predictions_df[
                            (predictions_df.pred_proba >= bin_min)
                            & (predictions_df.pred_proba <= bin_max)
                        ].target
                        == pos_label
                    ).mean()
                    new_row_dict["count"] = predictions_df[
                        (predictions_df.pred_proba >= bin_min)
                        & (predictions_df.pred_proba <= bin_max)
                    ].target.count()
                    if n_classes > 1:
                        for i in range(n_classes):
                            new_row_dict["precision_" + str(i)] = (
                                predictions_df[
                                    (predictions_df.pred_proba >= bin_min)
                                    & (predictions_df.pred_proba <= bin_max)
                                ].target
                                == i
                            ).mean()
                else:
                    new_row_dict["p_avg"] = predictions_df[
                        (predictions_df.pred_proba > bin_min)
                        & (predictions_df.pred_proba <= bin_max)
                    ]["pred_proba"].mean()
                    new_row_dict["precision"] = (
                        predictions_df[
                            (predictions_df.pred_proba > bin_min)
                            & (predictions_df.pred_proba <= bin_max)
                        ].target
                        == pos_label
                    ).mean()
                    new_row_dict["count"] = (
                        predictions_df[
                            (predictions_df.pred_proba > bin_min)
                            & (predictions_df.pred_proba <= bin_max)
                        ].target
                        == pos_label
                    ).count()
                    if n_classes > 1:
                        for i in range(n_classes):
                            new_row_dict["precision_" + str(i)] = (
                                predictions_df[
                                    (predictions_df.pred_proba > bin_min)
                                    & (predictions_df.pred_proba <= bin_max)
                                ].target
                                == i
                            ).mean()
                new_row_df = pd.DataFrame(new_row_dict, columns=precision_df.columns)
                if not new_row_df.empty:
                    for column in new_row_df.columns:
                        new_row_df[column] = new_row_df[column].astype(
                            precision_df[column].dtype
                        )
                    precision_df = pd.concat(
                        [precision_df, new_row_df], ignore_index=True
                    )

    elif quantiles:
        preds_quantiles = np.array_split(predictions_df.pred_proba.values, quantiles)
        target_quantiles = np.array_split(predictions_df.target.values, quantiles)

        last_p_max = 0.0
        for preds, targets in zip(preds_quantiles, target_quantiles):
            new_row_dict = {
                "p_min": [last_p_max],
                "p_max": [preds.max()],
                "p_avg": [preds.mean()],
                "bin_width": [preds.max() - last_p_max],
                "precision": [np.mean(targets == pos_label)],
                "count": [len(preds)],
            }
            if n_classes > 1:
                for i in range(n_classes):
                    new_row_dict["precision_" + str(i)] = np.mean(targets == i)

            new_row_df = pd.DataFrame(
                new_row_dict, columns=precision_df.columns
            ).astype(precision_df.dtypes)
            precision_df = pd.concat([precision_df, new_row_df])
            last_p_max = preds.max()

    precision_df[["p_avg", "precision"]] = (
        precision_df[["p_avg", "precision"]]
        .astype(float)
        .apply(partial(np.round, decimals=round))
    )
    if n_classes > 1:
        precision_cols = ["precision_" + str(i) for i in range(n_classes)]
        precision_df[precision_cols] = (
            precision_df[precision_cols]
            .astype(float)
            .apply(partial(np.round, decimals=round))
        )
    return precision_df


def get_liftcurve_df(pred_probas, y, pos_label=1, n_rows=100):
    """returns a pd.DataFrame that can be used to generate a lift curve plot.

    Args:
        pred_probas (np.ndarray): predicted probabilities of the positive class
        y (np.ndarray): the actual labels (y_true), encoded 0, 1 [, 2, 3, etc]
        pos_label (int): label of the positive class. Defaults to 1.

    Returns:
        pd.DataFrame with columns=['pred_proba', 'y', 'index', 'index_percentage',
                'positives', 'precision', 'cumulative_percentage_pos',
                'random_pos', 'random_precision', 'random_cumulative_percentage_pos']
    """
    lift_df = (
        pd.DataFrame({"pred_proba": pred_probas, "y": y.astype("int32")})
        .sort_values("pred_proba", ascending=False)
        .reset_index(drop=True)
    )
    lift_df["index"] = (lift_df.index + 1).astype("int32")
    lift_df["index_percentage"] = (100 * lift_df["index"] / len(lift_df)).astype(
        "float32"
    )
    lift_df["positives"] = (lift_df.y == pos_label).astype(int).cumsum()
    lift_df["precision"] = (100 * (lift_df["positives"] / lift_df["index"])).astype(
        "float32"
    )
    lift_df["cumulative_percentage_pos"] = (
        100 * (lift_df["positives"] / (lift_df.y == pos_label).astype(int).sum())
    ).astype("float32")
    lift_df["random_pos"] = (
        (lift_df.y == pos_label).astype(int).mean() * lift_df["index"]
    ).astype("float32")
    lift_df["random_precision"] = (
        100 * (lift_df["random_pos"] / lift_df["index"])
    ).astype("float32")
    lift_df["random_cumulative_percentage_pos"] = (
        100 * (lift_df["random_pos"] / (lift_df.y == pos_label).astype(int).sum())
    ).astype("float32")
    for y_label in range(y.nunique()):
        lift_df["precision_" + str(y_label)] = (
            100 * (lift_df.y == y_label).astype(int).cumsum() / lift_df["index"]
        )
    if len(lift_df) > 100:
        lift_df = lift_df.iloc[
            np.linspace(0, len(lift_df), num=n_rows, dtype=int, endpoint=False)
        ]
    return lift_df


def get_contrib_df(
    shap_base_value, shap_values, X_row, topx=None, cutoff=None, sort="abs", cols=None
):
    """
    Return a contrib_df DataFrame that lists the SHAP contribution of each input
    variable for a single prediction, formatted in a way that makes it easy to
    plot a waterfall plot.

    Args:
        shap_base_value (float): the value of shap.Explainer.expected_value
        shap_values (np.ndarray): single array of shap values for a specific
            prediction, corresponding to X_row
        X_row (pd.DataFrame): a single row of data, generated with e.g. X.iloc[[index]]
        topx (int): only display the topx highest impact features.
        cutoff (float): only display features with a SHAP value of at least
            cutoff.
        sort ({'abs', 'high-to-low', 'low-to-high'}), sort the shap value
            contributions either from highest absolute shap to lowest absolute
            shap ('abs'), or from most positive to most negative ('high-to-low')
            or from most negative to most positive ('low-to-high'). Defaults
            to 'abs'.
        cols (list of str): particular list of columns to display, in that order. Will
            override topx, cutoff, sort, etc.

    Features below topx or cutoff are summed together under _REST. Final
    prediction is added as _PREDICTION.

    Returns:
        pd.DataFrame with columns=['col', 'contribution', 'value', 'cumulative', 'base']
    """
    assert isinstance(
        X_row, pd.DataFrame
    ), "X_row should be a pd.DataFrame! Use X.iloc[[index]]"
    assert (
        len(X_row.iloc[[0]].values[0].shape) == 1
    ), """X is not the right shape: len(X.values[0]) should be 1.
            Try passing X.iloc[[index]]"""
    assert sort in {"abs", "high-to-low", "low-to-high", "importance", None}

    # start with the shap_base_value
    base_df = pd.DataFrame(
        {"col": ["_BASE"], "contribution": [shap_base_value], "value": [""]}
    )

    contrib_df = pd.DataFrame(
        {"col": X_row.columns, "contribution": shap_values, "value": X_row.values[0]}
    )
    if cols is None:
        if cutoff is None and topx is not None:
            cutoff = contrib_df.contribution.abs().nlargest(topx).min()
        elif cutoff is None and topx is None:
            cutoff = 0

        display_df = contrib_df[contrib_df.contribution.abs() >= cutoff]
        if topx is not None and len(display_df) > topx:
            # in case of ties around cutoff
            display_df = display_df.reindex(
                display_df.contribution.abs().sort_values(ascending=False).index
            ).head(topx)

        display_df_neg = display_df[display_df.contribution < 0]
        display_df_pos = display_df[display_df.contribution >= 0]
        logger.debug(
            "Excluded contributions: %s",
            contrib_df[~contrib_df.col.isin(display_df.col.tolist())],
        )

        rest_df = pd.DataFrame(
            {
                "col": ["_REST"],
                "contribution": [
                    contrib_df[~contrib_df.col.isin(display_df.col.tolist())][
                        "contribution"
                    ].sum()
                ],
                "value": [""],
            }
        )

        # sort the df by absolute value from highest to lowest:
        if sort == "abs":
            display_df = display_df.reindex(
                display_df.contribution.abs().sort_values(ascending=False).index
            )
            contrib_df = pd.concat([base_df, display_df, rest_df], ignore_index=True)
        if sort == "high-to-low":
            display_df_pos = display_df_pos.reindex(
                display_df_pos.contribution.abs().sort_values(ascending=False).index
            )
            display_df_neg = display_df_neg.reindex(
                display_df_neg.contribution.abs().sort_values().index
            )
            contrib_df = pd.concat(
                [base_df, display_df_pos, rest_df, display_df_neg], ignore_index=True
            )
        if sort == "low-to-high":
            display_df_pos = display_df_pos.reindex(
                display_df_pos.contribution.abs().sort_values().index
            )
            display_df_neg = display_df_neg.reindex(
                display_df_neg.contribution.abs().sort_values(ascending=False).index
            )
            contrib_df = pd.concat(
                [base_df, display_df_neg, rest_df, display_df_pos], ignore_index=True
            )
    else:
        display_df = (
            contrib_df[contrib_df.col.isin(cols)]
            .set_index("col")
            .reindex(cols)
            .reset_index()
        )
        rest_df = pd.DataFrame(
            {
                "col": ["_REST"],
                "contribution": [
                    contrib_df[~contrib_df.col.isin(cols)]["contribution"].sum()
                ],
                "value": [""],
            }
        )
        contrib_df = pd.concat([base_df, display_df, rest_df], ignore_index=True)

    # add cumulative contribution from top to bottom (for making bar chart):
    contrib_df["cumulative"] = contrib_df.contribution.cumsum()
    contrib_df["base"] = contrib_df["cumulative"] - contrib_df["contribution"]

    pred_df = (
        contrib_df[["contribution"]]
        .sum()
        .to_frame()
        .T.assign(
            col="_PREDICTION", value="", cumulative=lambda df: df.contribution, base=0
        )
    )
    return pd.concat([contrib_df, pred_df], ignore_index=True)


def get_contrib_summary_df(
    contrib_df, model_output="raw", round=2, units="", na_fill=None
):
    """
    returns a DataFrame that summarizes a contrib_df as a pair of
    Reasons+Effect.

    Args:
        contrib_df (pd.DataFrame): output from get_contrib_df(...)
        model_output (str, {'raw', 'probability', 'logodds'}): the type of
            predictions that the model produces. 'probability' multiplies by 100
            and adds '%'.
        round (int): number of decimals to round the output to. Defaults to 1.
        units (str): units to add to output. Defaults to "".
        na_fill (int, str): if value equals na_fill replace with "MISSING".

    """
    assert model_output in {"raw", "probability", "logodds"}
    contrib_summary_df = pd.DataFrame(columns=["Reason", "Effect"])

    for _, row in contrib_df.iterrows():
        if row["col"] == "_BASE":
            reason = "Average of population"
            effect = ""
        elif row["col"] == "_REST":
            reason = "Other features combined"
            effect = f"{'+' if row['contribution'] >= 0 else ''}"
        elif row["col"] == "_PREDICTION":
            reason = "Final prediction"
            effect = ""
        else:
            if na_fill is not None and row["value"] == na_fill:
                reason = f"{row['col']} = MISSING"
            else:
                reason = f"{row['col']} = {row['value']}"

            effect = f"{'+' if row['contribution'] >= 0 else ''}"
        if model_output == "probability":
            effect += str(np.round(100 * row["contribution"], round)) + "%"
        elif model_output == "logodds":
            effect += str(np.round(row["contribution"], round))
        else:
            effect += str(np.round(row["contribution"], round)) + f" {units}"

        contrib_summary_df = append_dict_to_df(
            contrib_summary_df, dict(Reason=reason, Effect=effect)
        )

    return contrib_summary_df.reset_index(drop=True)


def normalize_shap_interaction_values(shap_interaction_values, shap_values=None):
    """
    Normalizes shap_interaction_values to make sure that the rows add up to
    the shap_values.

    This is a workaround for an apparant bug where the diagonals of
    shap_interaction_values of a RandomForestClassifier are set equal to the
    shap_values instead of the main effect.

    I Opened an issue here: https://github.com/slundberg/shap/issues/723

    (so far doesn't seem to be fixed)

    Args:
        shap_interaction_values (np.ndarray): output of shap.Explainer.shap_interaction_values()
        shap_values (np.ndarray): output of shap.Explainer.shap_values()
    """
    siv = shap_interaction_values.copy()

    orig_diags = np.einsum("ijj->ij", siv)
    row_sums = np.einsum("ijk->ij", siv)
    row_diffs = row_sums - orig_diags  # sum of rows excluding diagonal elements

    if shap_values is not None:
        diags = shap_values - row_diffs
    else:
        # if no shap_values provided assume that the original diagonal values
        # were indeed equal to the shap values, and so simply
        diags = orig_diags - row_diffs

    s0, s1, s2 = siv.shape

    # should have commented this bit of code earlier:
    #   (can't really explain it anymore, but it works! :)
    # In any case, it assigns our new diagonal values to siv:
    siv.reshape(s0, -1)[:, :: s2 + 1] = diags
    return siv


def get_decisionpath_df(decision_tree, observation, pos_label=1, class_names=None):
    """summarize the path through a DecisionTree for a specific observation.

    Args:
        decision_tree (DecisionTreeClassifier or DecisionTreeRegressor):
            a fitted DecisionTree model.
        observation ([type]): single row of data to display tree path for.
        pos_label (int, optional): label of positive class. Defaults to 1.
        class_names (list, optional): List of class names for mapping pos_label to class values.
            Defaults to None.

    Returns:
        pd.DataFrame: columns=['node_id', 'average', 'feature',
            'value', 'split', 'direction', 'left', 'right', 'diff']
    """
    # Convert observation to numpy array for dtreeviz's predict_path
    # dtreeviz internally accesses by integer index (node.feature() returns int)
    if isinstance(observation, pd.Series):
        observation_array = observation.values
    elif isinstance(observation, pd.DataFrame):
        observation_array = (
            observation.values[0] if len(observation) == 1 else observation.values
        )
    else:
        observation_array = np.asarray(observation)

    nodes = decision_tree.predict_path(observation_array)

    decisiontree_df = pd.DataFrame(
        columns=[
            "node_id",
            "average",
            "feature",
            "value",
            "split",
            "direction",
            "left",
            "right",
            "diff",
        ]
    )
    if decision_tree.is_classifier():

        def node_pred_proba(node):
            class_counts_raw = node.class_counts()
            # Handle both dict and numpy array return types from class_counts()
            # Newer dtreeviz versions may return numpy arrays instead of dicts
            if isinstance(class_counts_raw, dict):
                class_counts = class_counts_raw
                total = sum(class_counts.values())
                if total == 0:
                    return 0.0

                # Try direct access first (most common case)
                if pos_label in class_counts:
                    return class_counts[pos_label] / total

                # If pos_label not found, try to map it to available class keys
                available_classes = sorted_categorical_values(class_counts.keys())
                if len(available_classes) == 0:
                    return 0.0

                # Map pos_label (index in labels) to actual class value
                if 0 <= pos_label < len(available_classes):
                    class_key = available_classes[pos_label]
                    return class_counts[class_key] / total

                # If pos_label is out of range, clamp it to valid range
                if pos_label >= len(available_classes):
                    class_key = available_classes[-1]
                    return class_counts[class_key] / total

                # Final fallback: use the class with the highest count
                class_key = max(class_counts, key=class_counts.get)
                return class_counts[class_key] / total
            else:
                # Handle numpy array case (newer dtreeviz versions)
                class_counts_array = np.asarray(class_counts_raw)
                total = class_counts_array.sum()
                if total == 0:
                    return 0.0

                # pos_label is an index into the array
                if 0 <= pos_label < len(class_counts_array):
                    return float(class_counts_array[pos_label]) / total
                elif len(class_counts_array) > 0:
                    # Clamp to valid range
                    return float(class_counts_array[-1]) / total
                return 0.0

        for node in nodes:
            if not node.isleaf():
                # Use node.feature() (integer index) to access observation_array
                # Use node.feature_name() (string) for display
                feature_idx = node.feature()
                feature_value = observation_array[feature_idx]
                decisiontree_df = append_dict_to_df(
                    decisiontree_df,
                    {
                        "node_id": node.id,
                        "average": node_pred_proba(node),
                        "feature": node.feature_name(),
                        "value": feature_value,
                        "split": node.split(),
                        "direction": "left"
                        if feature_value < node.split()
                        else "right",
                        "left": node_pred_proba(node.left),
                        "right": node_pred_proba(node.right),
                        "diff": node_pred_proba(node.left) - node_pred_proba(node)
                        if feature_value < node.split()
                        else node_pred_proba(node.right) - node_pred_proba(node),
                    },
                )

    else:

        def node_mean(node):
            return decision_tree.tree_model.tree_.value[node.id].item()

        for node in nodes:
            if not node.isleaf():
                # Use node.feature() (integer index) to access observation_array
                # Use node.feature_name() (string) for display
                feature_idx = node.feature()
                feature_value = observation_array[feature_idx]
                decisiontree_df = append_dict_to_df(
                    decisiontree_df,
                    {
                        "node_id": node.id,
                        "average": node_mean(node),
                        "feature": node.feature_name(),
                        "value": feature_value,
                        "split": node.split(),
                        "direction": "left"
                        if feature_value < node.split()
                        else "right",
                        "left": node_mean(node.left),
                        "right": node_mean(node.right),
                        "diff": node_mean(node.left) - node_mean(node)
                        if feature_value < node.split()
                        else node_mean(node.right) - node_mean(node),
                    },
                )
    return decisiontree_df


def get_decisiontree_summary_df(decisiontree_df, classifier=False, round=2, units=""):
    """generate a pd.DataFrame with a more readable summary of a dataframe
    generated with get_decisiontree_df(...)

    Args:
        decisiontree_df (pd.DataFrame): dataframe generated with get_decisiontree_df(...)
        classifier (bool, optional): model is a classifier. Defaults to False.
        round (int, optional): Rounding to apply to floats. Defaults to 2.
        units (str, optional): units of target to display. Defaults to "".

    Returns:
        pd.DataFrame: columns=['Feature', 'Condition', 'Adjustment', 'New Prediction']
    """
    if classifier:
        base_value = np.round(100 * decisiontree_df.iloc[[0]]["average"].item(), round)
        prediction = np.round(
            100
            * (
                decisiontree_df.iloc[[-1]]["average"].item()
                + decisiontree_df.iloc[[-1]]["diff"].item()
            ),
            round,
        )
    else:
        base_value = np.round(decisiontree_df.iloc[[0]]["average"].item(), round)
        prediction = np.round(
            decisiontree_df.iloc[[-1]]["average"].item()
            + decisiontree_df.iloc[[-1]]["diff"].item(),
            round,
        )

    decisiontree_summary_df = pd.DataFrame(
        columns=["Feature", "Condition", "Adjustment", "New Prediction"]
    )
    decisiontree_summary_df = append_dict_to_df(
        decisiontree_summary_df,
        {
            "Feature": "",
            "Condition": "",
            "Adjustment": "Starting average",
            "New Prediction": str(np.round(base_value, round))
            + ("%" if classifier else f" {units}"),
        },
    )

    for _, row in decisiontree_df.iterrows():
        if classifier:
            decisiontree_summary_df = append_dict_to_df(
                decisiontree_summary_df,
                {
                    "Feature": row["feature"],
                    "Condition": str(row["value"])
                    + str(" >= " if row["direction"] == "right" else " < ")
                    + str(row["split"]).ljust(10),
                    "Adjustment": str("+" if row["diff"] >= 0 else "")
                    + str(np.round(100 * row["diff"], round))
                    + "%",
                    "New Prediction": str(
                        np.round(100 * (row["average"] + row["diff"]), round)
                    )
                    + "%",
                },
            )
        else:
            decisiontree_summary_df = append_dict_to_df(
                decisiontree_summary_df,
                {
                    "Feature": row["feature"],
                    "Condition": str(row["value"])
                    + str(" >= " if row["direction"] == "right" else " < ")
                    + str(row["split"]).ljust(10),
                    "Adjustment": str("+" if row["diff"] >= 0 else "")
                    + str(np.round(row["diff"], round)),
                    "New Prediction": str(
                        np.round((row["average"] + row["diff"]), round)
                    )
                    + f" {units}",
                },
            )

    decisiontree_summary_df = append_dict_to_df(
        decisiontree_summary_df,
        {
            "Feature": "",
            "Condition": "",
            "Adjustment": "Final Prediction",
            "New Prediction": str(np.round(prediction, round))
            + ("%" if classifier else "")
            + f" {units}",
        },
    )

    return decisiontree_summary_df


def get_xgboost_node_dict(xgboost_treedump):
    """Turns the output of a xgboostmodel.get_dump() into a dictionary
    of nodes for easy parsing a prediction path through individual trees
    in the model.

    Args:
        xgboost_treedump (str): a single element of the list output from
            xgboost model.get_dump() that represents a single tree in the
            ensemble.
    Returns:
        dict
    """
    node_dict = {}
    for row in xgboost_treedump.splitlines():
        s = row.strip()
        node = int(re.search(r"^(.*)\:", s).group(1))
        is_leaf = re.search(r":(.*)\=", s).group(1) == "leaf"

        leaf_value = re.search(r"leaf=(.*)$", s).group(1) if is_leaf else None
        feature = re.search(r"\[(.*)\<", s).group(1) if not is_leaf else None
        cutoff = float(re.search(r"\<(.*)\]", s).group(1)) if not is_leaf else None
        left_node = int(re.search(r"yes=(.*)\,no", s).group(1)) if not is_leaf else None
        right_node = int(re.search(r"no=(.*)\,", s).group(1)) if not is_leaf else None
        node_dict[node] = dict(
            node=node,
            is_leaf=is_leaf,
            leaf_value=leaf_value,
            feature=feature,
            cutoff=cutoff,
            left_node=left_node,
            right_node=right_node,
        )
    return node_dict


def get_xgboost_path_df(xgbmodel, X_row, n_tree=None):
    """returns a pd.DataFrame of the prediction path through
    an individual tree in a xgboost ensemble.

    Args:
        xgbmodel: either a fitted xgboost model, or the output of a get_dump()
        X_row: single row from a dataframe (e.g. X_test.iloc[0])
        n_tree: the tree number to display:

    Returns:
        pd.DataFrame
    """
    if isinstance(xgbmodel, str) and xgbmodel.startswith("0:"):
        xgbmodel_treedump = xgbmodel
    elif str(type(xgbmodel)).endswith("xgboost.core.Booster'>"):
        xgbmodel_treedump = xgbmodel.get_dump()[n_tree]
    elif str(type(xgbmodel)).endswith("XGBClassifier'>") or str(
        type(xgbmodel)
    ).endswith("XGBRegressor'>"):
        xgbmodel_treedump = xgbmodel.get_booster().get_dump()[n_tree]
    else:
        raise ValueError(
            "Couldn't extract a treedump. Please pass a fitted xgboost model."
        )
    if isinstance(X_row, pd.DataFrame) and len(X_row) == 1:
        X_row = X_row.squeeze()
    node_dict = get_xgboost_node_dict(xgbmodel_treedump)

    prediction_path_df = pd.DataFrame(columns=["node", "feature", "cutoff", "value"])

    node = node_dict[0]
    while not node["is_leaf"]:
        prediction_path_df = append_dict_to_df(
            prediction_path_df,
            dict(
                node=node["node"],
                feature=node["feature"],
                cutoff=node["cutoff"],
                value=float(X_row[node["feature"]]),
            ),
        )
        if np.isnan(X_row[node["feature"]]) or X_row[node["feature"]] < node["cutoff"]:
            node = node_dict[node["left_node"]]
        else:
            node = node_dict[node["right_node"]]

    if node["is_leaf"]:
        prediction_path_df = append_dict_to_df(
            prediction_path_df,
            dict(node=node["node"], feature="_PREDICTION", value=node["leaf_value"]),
        )
    return prediction_path_df


def get_xgboost_path_summary_df(xgboost_path_df, output="margin"):
    """turn output of get_xgboost_path_df output into a formatted dataframe

    Args:
        xgboost_path_df (pd.DataFrame): output of get_xgboost_path_df
        prediction (str, {'logodds', 'margin'}): Type of output prediction.
            Defaults to "margin".

    Returns:
        pd.DataFrame: dataframe with nodes and split conditions
    """
    xgboost_path_summary_df = pd.DataFrame(columns=["node", "split_condition"])

    for row in xgboost_path_df.itertuples():
        if row.feature == "_PREDICTION":
            xgboost_path_summary_df = append_dict_to_df(
                xgboost_path_summary_df,
                dict(
                    node=row.node,
                    split_condition=f"prediction ({output}) = {row.value}",
                ),
            )
        elif row.value < row.cutoff:
            xgboost_path_summary_df = append_dict_to_df(
                xgboost_path_summary_df,
                dict(
                    node=row.node,
                    split_condition=f"{row.feature} = {row.value} < {row.cutoff}",
                ),
            )
        else:
            xgboost_path_summary_df = append_dict_to_df(
                xgboost_path_summary_df,
                dict(
                    node=row.node,
                    split_condition=f"{row.feature} = {row.value} >= {row.cutoff}",
                ),
            )
    return xgboost_path_summary_df


def get_xgboost_preds_df(xgbmodel, X_row, pos_label=1):
    """returns the marginal contributions of each tree in
    an xgboost ensemble

    Args:
        xgbmodel: a fitted sklearn-comptaible xgboost model
            (i.e. XGBClassifier or XGBRegressor)
        X_row: a single row of data, e.g X_train.iloc[0]
        pos_label: for classifier the label to be used as positive label
            Defaults to 1.

    Returns:
        pd.DataFrame
    """
    if str(type(xgbmodel)).endswith("XGBClassifier'>"):
        is_classifier = True
        n_classes = len(xgbmodel.classes_)
        if n_classes == 2:
            base_score_raw = xgbmodel.get_params()["base_score"]
            base_score_raw = (
                _ensure_numeric_predictions(base_score_raw)
                if base_score_raw is not None
                else None
            )
            if pos_label == 1:
                base_proba = (
                    float(base_score_raw) if base_score_raw is not None else 0.5
                )
            elif pos_label == 0:
                base_proba = 1 - (
                    float(base_score_raw) if base_score_raw is not None else 0.5
                )
            else:
                raise ValueError("pos_label should be either 0 or 1!")
            n_trees = len(xgbmodel.get_booster().get_dump())
            base_score = np.log(base_proba / (1 - base_proba))
        else:
            base_proba = 1.0 / n_classes
            base_score_raw = xgbmodel.get_params()["base_score"]
            base_score_raw = (
                _ensure_numeric_predictions(base_score_raw)
                if base_score_raw is not None
                else None
            )
            base_score = float(base_score_raw) if base_score_raw is not None else 0.5
            n_trees = int(len(xgbmodel.get_booster().get_dump()) / n_classes)

    elif str(type(xgbmodel)).endswith("XGBRegressor'>"):
        is_classifier = False
        base_score_raw = xgbmodel.get_params()["base_score"]
        base_score_raw = _ensure_numeric_predictions(base_score_raw)
        base_score = float(base_score_raw) if base_score_raw is not None else 0.5
        n_trees = len(xgbmodel.get_booster().get_dump())
    else:
        raise ValueError("Pass either an XGBClassifier or XGBRegressor!")

    if is_classifier:
        if n_classes == 2:
            if pos_label == 1:
                preds_raw = [
                    xgbmodel.predict(
                        X_row, iteration_range=(0, i + 1), output_margin=True
                    )[0]
                    for i in range(n_trees)
                ]
            elif pos_label == 0:
                preds_raw = [
                    -xgbmodel.predict(
                        X_row, iteration_range=(0, i + 1), output_margin=True
                    )[0]
                    for i in range(n_trees)
                ]
            # Convert XGBoost 3.0+ string predictions to numeric
            preds = []
            for p in preds_raw:
                p_conv = _ensure_numeric_predictions(p)
                if isinstance(p_conv, np.ndarray):
                    p_conv = p_conv.item() if p_conv.ndim == 0 else float(p_conv[0])
                preds.append(float(p_conv))
            pred_probas = (np.exp(preds) / (1 + np.exp(preds))).tolist()
        else:
            margins_raw = [
                xgbmodel.predict(X_row, iteration_range=(0, i + 1), output_margin=True)[
                    0
                ]
                for i in range(n_trees)
            ]
            # Convert XGBoost 3.0+ string predictions to numeric
            margins = []
            for m in margins_raw:
                m_conv = _ensure_numeric_predictions(m)
                if isinstance(m_conv, np.ndarray):
                    margins.append(m_conv)
                elif isinstance(m_conv, (list, tuple)):
                    margins.append(
                        np.asarray([_ensure_numeric_predictions(x) for x in m_conv])
                    )
                else:
                    margins.append(np.asarray([float(m_conv)]))
            preds = [margin[pos_label] for margin in margins]
            pred_probas = [
                (np.exp(margin) / np.exp(margin).sum())[pos_label] for margin in margins
            ]

    else:
        preds_raw = [
            xgbmodel.predict(X_row, iteration_range=(0, i + 1), output_margin=True)[0]
            for i in range(n_trees)
        ]
        # Convert XGBoost 3.0+ string predictions to numeric
        preds = []
        for p in preds_raw:
            p_conv = _ensure_numeric_predictions(p)
            if isinstance(p_conv, np.ndarray):
                p_conv = p_conv.item() if p_conv.ndim == 0 else float(p_conv[0])
            preds.append(float(p_conv))

    xgboost_preds_df = pd.DataFrame(
        dict(tree=range(-1, n_trees), pred=[base_score] + preds)
    )
    xgboost_preds_df["pred_diff"] = xgboost_preds_df.pred.diff()
    xgboost_preds_df.loc[0, "pred_diff"] = xgboost_preds_df.loc[0, "pred"]

    if is_classifier:
        xgboost_preds_df["pred_proba"] = [base_proba] + pred_probas
        xgboost_preds_df["pred_proba_diff"] = xgboost_preds_df.pred_proba.diff()
        xgboost_preds_df.loc[0, "pred_proba_diff"] = xgboost_preds_df.loc[
            0, "pred_proba"
        ]
    return xgboost_preds_df
