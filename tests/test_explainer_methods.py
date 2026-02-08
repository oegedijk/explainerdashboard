import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from explainerdashboard.explainer_methods import (
    build_pipeline_extraction_warning,
    get_transformed_X,
    infer_cats_from_transformed_X,
    is_binary_like_onehot_column,
    rename_pipeline_columns,
    split_pipeline,
)


def test_is_binary_like_onehot_column_accepts_strict_onehot_values():
    series = pd.Series([0, 1, 0, 1, 0], dtype=float)
    assert is_binary_like_onehot_column(series)


def test_is_binary_like_onehot_column_accepts_scaled_binary_values():
    series = pd.Series([-1.0, 0.5, -1.0, 0.5, -1.0], dtype=float)
    assert is_binary_like_onehot_column(series)


def test_is_binary_like_onehot_column_rejects_non_binary_values():
    series = pd.Series([0.0, 0.5, 1.0, 0.5, 0.0], dtype=float)
    assert not is_binary_like_onehot_column(series)


def test_rename_pipeline_columns_strips_prefixes():
    columns = ["num__age", "cat__city_A", "cat__city_B"]
    renamed = rename_pipeline_columns(columns, strip_pipeline_prefix=True)
    assert renamed == ["age", "city_A", "city_B"]


def test_rename_pipeline_columns_uses_custom_function():
    columns = ["num__age", "cat__city_A"]
    renamed = rename_pipeline_columns(columns, feature_name_fn=lambda c: c.upper())
    assert renamed == ["NUM__AGE", "CAT__CITY_A"]


def test_rename_pipeline_columns_suffixes_duplicates():
    columns = ["num__age", "cat__age"]
    renamed = rename_pipeline_columns(columns, strip_pipeline_prefix=True, verbose=0)
    assert renamed == ["age", "age__2"]


def test_get_transformed_X_preserves_index():
    X = pd.DataFrame(
        {"age": [20, 21, 22, 23], "city": ["A", "B", "A", "C"]},
        index=["row-1", "row-2", "row-3", "row-4"],
    )
    y = np.array([10, 11, 12, 13])
    pipeline = Pipeline(
        [
            (
                "preprocessor",
                ColumnTransformer(
                    [
                        ("num", StandardScaler(), ["age"]),
                        (
                            "cat",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                            ["city"],
                        ),
                    ],
                    sparse_threshold=0,
                ),
            ),
            ("model", LinearRegression()),
        ]
    ).fit(X, y)
    transformer_pipeline, _ = split_pipeline(pipeline, verbose=0)
    transformed_X = get_transformed_X(transformer_pipeline, X, verbose=0)

    assert list(transformed_X.index) == list(X.index)


def test_infer_cats_from_transformed_X_detects_onehot_groups():
    X = pd.DataFrame({"age": [20, 21, 22, 23], "city": ["A", "B", "A", "C"]})
    y = np.array([10, 11, 12, 13])
    pipeline = Pipeline(
        [
            (
                "preprocessor",
                ColumnTransformer(
                    [
                        ("num", StandardScaler(), ["age"]),
                        (
                            "cat",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                            ["city"],
                        ),
                    ],
                    sparse_threshold=0,
                ),
            ),
            ("model", LinearRegression()),
        ]
    ).fit(X, y)
    transformer_pipeline, _ = split_pipeline(pipeline, verbose=0)
    transformed_X = get_transformed_X(transformer_pipeline, X, verbose=0)

    inferred = infer_cats_from_transformed_X(transformed_X, list(X.columns))

    assert inferred == {"cat__city": ["cat__city_A", "cat__city_B", "cat__city_C"]}


def test_build_pipeline_extraction_warning_has_actionable_guidance():
    message = build_pipeline_extraction_warning(RuntimeError("boom"))
    assert "set shap='kernel'" in message
    assert "get_feature_names_out" in message
    assert "Error: boom" in message
