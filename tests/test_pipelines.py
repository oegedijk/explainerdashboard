import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from explainerdashboard.explainer_methods import (
    get_transformed_X,
    parse_cats,
    split_pipeline,
)
from explainerdashboard import RegressionExplainer


def test_pipeline_columns_ranked_by_shap(classifier_pipeline_explainer):
    assert isinstance(classifier_pipeline_explainer.columns_ranked_by_shap(), list)


def test_pipeline_permutation_importances(classifier_pipeline_explainer):
    assert isinstance(
        classifier_pipeline_explainer.get_permutation_importances_df(), pd.DataFrame
    )


def test_pipeline_metrics(classifier_pipeline_explainer):
    assert isinstance(classifier_pipeline_explainer.metrics(), dict)
    assert isinstance(classifier_pipeline_explainer.metrics_descriptions(), dict)


def test_pipeline_mean_abs_shap_df(classifier_pipeline_explainer):
    assert isinstance(
        classifier_pipeline_explainer.get_mean_abs_shap_df(), pd.DataFrame
    )


def test_pipeline_contrib_df(classifier_pipeline_explainer):
    assert isinstance(classifier_pipeline_explainer.get_contrib_df(0), pd.DataFrame)
    assert isinstance(
        classifier_pipeline_explainer.get_contrib_df(
            X_row=classifier_pipeline_explainer.X.iloc[[0]]
        ),
        pd.DataFrame,
    )


def test_pipeline_shap_base_value(classifier_pipeline_explainer):
    assert isinstance(
        classifier_pipeline_explainer.shap_base_value(), (np.floating, float)
    )


def test_pipeline_shap_values_shape(classifier_pipeline_explainer):
    assert classifier_pipeline_explainer.get_shap_values_df().shape == (
        len(classifier_pipeline_explainer),
        len(classifier_pipeline_explainer.merged_cols),
    )


def test_pipeline_shap_values(classifier_pipeline_explainer):
    assert isinstance(classifier_pipeline_explainer.get_shap_values_df(), pd.DataFrame)


def test_pipeline_pdp_df(classifier_pipeline_explainer):
    assert isinstance(classifier_pipeline_explainer.pdp_df("num__age"), pd.DataFrame)
    assert isinstance(classifier_pipeline_explainer.pdp_df("cat__sex"), pd.DataFrame)
    assert isinstance(
        classifier_pipeline_explainer.pdp_df("num__age", index=0), pd.DataFrame
    )
    assert isinstance(
        classifier_pipeline_explainer.pdp_df("cat__sex", index=0), pd.DataFrame
    )


def test_pipeline_kernel_columns_ranked_by_shap(classifier_pipeline_kernel_explainer):
    assert isinstance(
        classifier_pipeline_kernel_explainer.columns_ranked_by_shap(), list
    )


def test_pipeline_kernel_permutation_importances(classifier_pipeline_kernel_explainer):
    assert isinstance(
        classifier_pipeline_kernel_explainer.get_permutation_importances_df(),
        pd.DataFrame,
    )


def test_pipeline_kernel_metrics(classifier_pipeline_kernel_explainer):
    assert isinstance(classifier_pipeline_kernel_explainer.metrics(), dict)
    assert isinstance(classifier_pipeline_kernel_explainer.metrics_descriptions(), dict)


def test_pipeline_kernel_mean_abs_shap_df(classifier_pipeline_kernel_explainer):
    assert isinstance(
        classifier_pipeline_kernel_explainer.get_mean_abs_shap_df(), pd.DataFrame
    )


def test_pipeline_kernel_contrib_df(classifier_pipeline_kernel_explainer):
    assert isinstance(
        classifier_pipeline_kernel_explainer.get_contrib_df(0), pd.DataFrame
    )
    assert isinstance(
        classifier_pipeline_kernel_explainer.get_contrib_df(
            X_row=classifier_pipeline_kernel_explainer.X.iloc[[0]]
        ),
        pd.DataFrame,
    )


def test_pipeline_kernel_shap_base_value(classifier_pipeline_kernel_explainer):
    assert isinstance(
        classifier_pipeline_kernel_explainer.shap_base_value(), (np.floating, float)
    )


def test_pipeline_kernel_shap_values_shape(classifier_pipeline_kernel_explainer):
    assert classifier_pipeline_kernel_explainer.get_shap_values_df().shape == (
        len(classifier_pipeline_kernel_explainer),
        len(classifier_pipeline_kernel_explainer.merged_cols),
    )


def test_pipeline_kernel_shap_values(classifier_pipeline_kernel_explainer):
    assert isinstance(
        classifier_pipeline_kernel_explainer.get_shap_values_df(), pd.DataFrame
    )


def test_pipeline_kernel_pdp_df(classifier_pipeline_kernel_explainer):
    assert isinstance(classifier_pipeline_kernel_explainer.pdp_df("age"), pd.DataFrame)
    assert isinstance(classifier_pipeline_kernel_explainer.pdp_df("sex"), pd.DataFrame)
    assert isinstance(
        classifier_pipeline_kernel_explainer.pdp_df("age", index=0), pd.DataFrame
    )
    assert isinstance(
        classifier_pipeline_kernel_explainer.pdp_df("sex", index=0), pd.DataFrame
    )


def test_get_transformed_X_strip_pipeline_prefix_removes_double_underscores():
    X = pd.DataFrame(
        {
            "age": [20, 21, 22, 23, 24, 25],
            "city": ["A", "B", "A", "C", "B", "A"],
        }
    )
    y = np.array([0, 1, 0, 1, 0, 1])

    preprocessor = ColumnTransformer(
        [
            ("num", StandardScaler(), ["age"]),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ["city"],
            ),
        ],
        sparse_threshold=0,
    )
    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(max_iter=200)),
        ]
    ).fit(X, y)

    transformer_pipeline, _ = split_pipeline(pipeline, verbose=0)
    transformed_X = get_transformed_X(
        transformer_pipeline, X, verbose=0, strip_pipeline_prefix=True
    )

    assert all("__" not in col for col in transformed_X.columns)
    assert {"age", "city_A", "city_B", "city_C"}.issubset(set(transformed_X.columns))


def test_parse_cats_accepts_scaled_binary_like_onehot_columns():
    X = pd.DataFrame(
        {
            "age": [20, 21, 22, 23, 24, 25],
            "city": ["A", "A", "A", "A", "B", "C"],
        }
    )
    y = np.array([0, 1, 0, 1, 0, 1])

    preprocessor = ColumnTransformer(
        [
            ("num", StandardScaler(), ["age"]),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ["city"],
            ),
        ],
        sparse_threshold=0,
    )
    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("post_scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=200)),
        ]
    ).fit(X, y)

    transformer_pipeline, _ = split_pipeline(pipeline, verbose=0)
    transformed_X = get_transformed_X(transformer_pipeline, X, verbose=0)

    onehot_cols, onehot_dict = parse_cats(transformed_X, ["cat__city"])

    assert onehot_cols == ["cat__city"]
    assert set(onehot_dict["cat__city"]) == {
        "cat__city_A",
        "cat__city_B",
        "cat__city_C",
    }


def test_pipeline_feature_name_fn_passthrough_applies_to_explainer_columns():
    X = pd.DataFrame(
        {
            "age": [20, 21, 22, 23, 24, 25],
            "city": ["A", "B", "A", "C", "B", "A"],
        }
    )
    y = pd.Series([10, 11, 12, 13, 14, 15])

    preprocessor = ColumnTransformer(
        [
            ("num", StandardScaler(), ["age"]),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ["city"],
            ),
        ],
        sparse_threshold=0,
    )
    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", LinearRegression()),
        ]
    ).fit(X, y)

    explainer = RegressionExplainer(
        pipeline,
        X,
        y,
        feature_name_fn=lambda c: c.split("__", 1)[1] if "__" in c else c,
    )

    assert "num__age" not in explainer.columns
    assert "age" in explainer.columns


def test_pipeline_auto_detect_cats_groups_onehot_columns():
    X = pd.DataFrame(
        {
            "age": [20, 21, 22, 23, 24, 25],
            "city": ["A", "B", "A", "C", "B", "A"],
        }
    )
    y = pd.Series([10, 11, 12, 13, 14, 15])

    preprocessor = ColumnTransformer(
        [
            ("num", StandardScaler(), ["age"]),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ["city"],
            ),
        ],
        sparse_threshold=0,
    )
    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", LinearRegression()),
        ]
    ).fit(X, y)

    explainer = RegressionExplainer(
        pipeline,
        X,
        y,
        auto_detect_pipeline_cats=True,
    )

    assert "cat__city" in explainer.onehot_cols
    assert set(explainer.onehot_dict["cat__city"]) == {
        "cat__city_A",
        "cat__city_B",
        "cat__city_C",
    }


def test_pipeline_auto_detect_cats_does_not_group_drop_if_binary():
    X = pd.DataFrame(
        {
            "age": [20, 21, 22, 23, 24, 25],
            "city": ["A", "B", "A", "B", "B", "A"],
        }
    )
    y = pd.Series([10, 11, 12, 13, 14, 15])

    preprocessor = ColumnTransformer(
        [
            ("num", StandardScaler(), ["age"]),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore", sparse_output=False, drop="if_binary"
                ),
                ["city"],
            ),
        ],
        sparse_threshold=0,
    )
    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", LinearRegression()),
        ]
    ).fit(X, y)

    explainer = RegressionExplainer(
        pipeline,
        X,
        y,
        auto_detect_pipeline_cats=True,
    )

    assert "cat__city" not in explainer.onehot_cols
