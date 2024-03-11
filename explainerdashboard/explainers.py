__all__ = [
    "BaseExplainer",
    "ClassifierExplainer",
    "RegressionExplainer",
    "RandomForestClassifierExplainer",
    "RandomForestRegressionExplainer",
    "XGBClassifierExplainer",
    "XGBRegressionExplainer",
]

import sys
import inspect
from abc import ABC
import base64
from pathlib import Path
from typing import List, Dict, Union, Callable
from types import MethodType
from functools import wraps
from threading import Lock
import warnings

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

import shap

from dtreeviz import DTreeVizAPI
from dtreeviz.models.shadow_decision_tree import ShadowDecTree

from sklearn.model_selection import KFold
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    roc_curve,
    confusion_matrix,
)
from sklearn.metrics import (
    precision_recall_curve,
    precision_score,
    recall_score,
    log_loss,
)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import average_precision_score


from .explainer_methods import *
from .explainer_plots import *


import plotly.io as pio

pio.templates.default = "none"


def insert_pos_label(func):
    """decorator to insert pos_label=self.pos_label into method call when pos_label=None"""

    @wraps(func)
    def inner(self, *args, **kwargs):
        if not self.is_classifier:
            return func(self, *args, **kwargs)
        if "pos_label" in kwargs:
            if kwargs["pos_label"] is not None:
                # ensure that pos_label is int
                kwargs.update(dict(pos_label=self.pos_label_index(kwargs["pos_label"])))
                return func(self, *args, **kwargs)
            else:
                # insert self.pos_label
                kwargs.update(dict(pos_label=self.pos_label))
                return func(self, *args, **kwargs)
        kwargs.update(
            dict(zip(inspect.getfullargspec(func).args[1 : 1 + len(args)], args))
        )
        if "pos_label" in kwargs:
            if kwargs["pos_label"] is not None:
                kwargs.update(dict(pos_label=self.pos_label_index(kwargs["pos_label"])))
            else:
                kwargs.update(dict(pos_label=self.pos_label))
        else:
            kwargs.update(dict(pos_label=self.pos_label))
        return func(self, **kwargs)

    return inner


class BaseExplainer(ABC):
    """ """

    def __init__(
        self,
        model,
        X: pd.DataFrame,
        y: pd.Series = None,
        permutation_metric: Callable = r2_score,
        shap: str = "guess",
        X_background: pd.DataFrame = None,
        model_output: str = "raw",
        cats: bool = None,
        cats_notencoded: Dict = None,
        idxs: pd.Index = None,
        index_name: str = None,
        target: str = None,
        descriptions: dict = None,
        n_jobs: int = None,
        permutation_cv: int = None,
        cv: int = None,
        na_fill: float = -999,
        precision: str = "float64",
        shap_kwargs: Dict = None,
    ):
        """Defines the basic functionality that is shared by both
        ClassifierExplainer and RegressionExplainer.

        Args:
            model: a model with a scikit-learn compatible .fit and .predict methods
            X (pd.DataFrame): a pd.DataFrame with your model features
            y (pd.Series): Dependent variable of your model, defaults to None
            permutation_metric (function or str): is a scikit-learn compatible
                metric function (or string). Defaults to r2_score
            shap (str): type of shap_explainer to fit: 'tree', 'linear', 'kernel'.
                Defaults to 'guess'.
            X_background (pd.DataFrame): background X to be used by shap
                explainers that need a background dataset (e.g. shap.KernelExplainer
                or shap.TreeExplainer with boosting models and
                model_output='probability').
            model_output (str): model_output of shap values, either 'raw',
                'logodds' or 'probability'. Defaults to 'raw' for regression and
                'probability' for classification.
            cats ({dict, list}): dict of features that have been
                onehotencoded. e.g. cats={'Sex':['Sex_male', 'Sex_female']}.
                If all encoded columns are underscore-seperated (as above), can simply
                pass a list of prefixes: cats=['Sex']. Allows to
                group onehot encoded categorical variables together in
                various plots. Defaults to None.
            cats_notencoded (dict): value to display when all onehot encoded
                columns are equal to zero. Defaults to 'NOT_ENCODED' for each
                onehot col.
            idxs (pd.Series): list of row identifiers. Can be names, id's, etc.
                Defaults to X.index.
            index_name (str): identifier for row indexes. e.g. index_name='Passenger'.
                Defaults to X.index.name or idxs.name.
            target: name of the predicted target, e.g. "Survival",
                "Ticket price", etc. Defaults to y.name.
            n_jobs (int): for jobs that can be parallelized using joblib,
                how many processes to split the job in. For now only used
                for calculating permutation importances. Defaults to None.
            permutation_cv (int): Deprecated! Use parameter cv instead!
                (now also works for calculating metrics)
            cv (int): If not None then permutation importances and metrics
                will get calculated using cross validation across X. Use this
                when you are passing the training set to the explainer.
                Defaults to None.
            na_fill (int): The filler used for missing values, defaults to -999.
            precision: precision with which to store values. Defaults to "float64".
            shap_kwargs(dict): dictionary of keyword arguments to be passed to the shap explainer.
                most typically used to supress an additivity check e.g. `shap_kwargs=dict(check_additivity=False)`
        """
        self._params_dict = dict(
            shap=shap,
            model_output=model_output,
            cats=cats,
            descriptions=descriptions,
            target=target,
            n_jobs=n_jobs,
            permutation_cv=permutation_cv,
            cv=cv,
            na_fill=na_fill,
            precision=precision,
            shap_kwargs=shap_kwargs,
        )

        if permutation_cv is not None:
            warnings.warn(
                "Parameter permutation_cv has been deprecated! Please use "
                "the new parameter `cv` instead! (Which now also works for "
                "calculating cross-validated metrics!)"
            )
            if cv is None:
                cv = permutation_cv

        if safe_isinstance(
            model, "sklearn.pipeline.Pipeline", "imblearn.pipeline.Pipeline"
        ):
            if shap != "kernel":
                try:
                    transformer_pipeline, self.model = split_pipeline(model)
                    self.X = get_transformed_X(transformer_pipeline, X)
                    if X_background is not None:
                        self.X_background = get_transformed_X(
                            transformer_pipeline, X_background
                        )
                    print(
                        "Detected sklearn/imblearn Pipeline and succesfully extracted final "
                        "output dataframe with column names and final model..."
                    )
                except:
                    print(
                        "Warning: Failed to extract a data transformer with column names and final "
                        "model from the Pipeline. So setting shap='kernel' to use "
                        "the (slower and approximate) model-agnostic shap.KernelExplainer "
                        "instead!"
                    )
                    shap = "kernel"

        if not hasattr(self, "X"):
            self.X = X.copy()
        if not hasattr(self, "X_background"):
            if X_background is not None:
                self.X_background = X_background.copy()
            else:
                self.X_background = None
        if not hasattr(self, "model"):
            self.model = model

        if safe_isinstance(model, "xgboost.core.Booster"):
            raise ValueError(
                "For xgboost models, currently only the scikit-learn "
                "compatible wrappers xgboost.sklearn.XGBClassifier and "
                "xgboost.sklearn.XGBRegressor are supported, so please use those "
                "instead of xgboost.Booster!"
            )

        if safe_isinstance(model, "lightgbm.Booster"):
            raise ValueError(
                "For lightgbm, currently only the scikit-learn "
                "compatible wrappers lightgbm.LGBMClassifier and lightgbm.LGBMRegressor "
                "are supported, so please use those instead of lightgbm.Booster!"
            )

        self.onehot_cols, self.onehot_dict = parse_cats(self.X, cats)
        self.encoded_cols, self.regular_cols = get_encoded_and_regular_cols(
            self.X.columns, self.onehot_dict
        )
        self.categorical_cols = [
            col for col in self.regular_cols if not is_numeric_dtype(self.X[col])
        ]
        self.categorical_dict = {
            col: sorted(self.X[col].unique().tolist()) for col in self.categorical_cols
        }
        self.cat_cols = self.onehot_cols + self.categorical_cols
        self.original_cols = self.X.columns
        self.merged_cols = pd.Index(self.regular_cols + self.onehot_cols)

        self.onehot_notencoded = {col: "NOT_ENCODED" for col in self.onehot_cols}
        if cats_notencoded is not None:
            assert isinstance(cats_notencoded, dict), (
                "cats_notencoded should be a dict mapping a onehot col to a "
                " missing value, e.g. cats_notencoded={'Deck': 'Unknown Deck'}...!"
            )
            assert set(cats_notencoded.keys()).issubset(self.onehot_cols), (
                "The following keys in cats_notencoded are not in cats:"
                f"{list(set(cats_notencoded.keys()) - set(self.onehot_cols))}!"
            )
            self.onehot_notencoded.update(cats_notencoded)

        if self.encoded_cols:
            self.X[self.encoded_cols] = self.X[self.encoded_cols].astype(np.int8)

        if self.categorical_cols:
            for col in self.categorical_cols:
                self.X[col] = self.X[col].astype("category")
            if not isinstance(self.model, Pipeline):
                print(
                    f"Warning: Detected the following categorical columns: {self.categorical_cols}. "
                    "Unfortunately for now shap interaction values do not work with"
                    "categorical columns.",
                    flush=True,
                )
                self.interactions_should_work = False

        if y is not None:
            if isinstance(y, pd.DataFrame):
                if len(y.columns) == 1:
                    y = y.squeeze()
                else:
                    raise ValueError(
                        "y should be a pd.Series or np.ndarray not a pd.DataFrame!"
                    )

            self.y = pd.Series(y.squeeze()).astype(precision)
            self.y_missing = False
        else:
            self.y = pd.Series(np.full(len(X), np.nan))
            self.y_missing = True
        if self.y.name is None:
            self.y.name = "Target"

        self.metric = permutation_metric
        self.shap_kwargs = shap_kwargs or {}

        if shap == "guess":
            shap_guess = guess_shap(self.model)
            model_str = (
                str(type(self.model))
                .replace("'", "")
                .replace("<", "")
                .replace(">", "")
                .split(".")[-1]
            )
            if shap_guess is not None:
                self.shap = shap_guess
            else:
                self.shap = "kernel"
                print(
                    "WARNING: Parameter shap='guess', but failed to guess the "
                    f"type of shap explainer to use for {model_str}. "
                    "Defaulting to the model agnostic shap.KernelExplainer "
                    "(shap='kernel'). However this will be slow, so if your model is "
                    "compatible with e.g. shap.TreeExplainer or shap.LinearExplainer "
                    "then pass shap='tree' or shap='linear'!"
                )
        else:
            if shap in {"deep", "torch"}:
                raise ValueError(
                    "ERROR! Only PyTorch neural networks wrapped in a skorch "
                    "sklearn-compatible NeuralNet wrapper are supported for now! "
                    "See https://github.com/skorch-dev/skorch"
                )
            assert shap in ["tree", "linear", "deep", "kernel", "skorch"], (
                "ERROR! Only shap='guess', 'tree', 'linear', ' kernel' or 'skorch' are "
                " supported for now!"
            )
            self.shap = shap
        if self.shap in {"kernel", "skorch", "linear"}:
            print(
                f"WARNING: For shap='{self.shap}', shap interaction values can unfortunately "
                "not be calculated!"
            )
            self.interactions_should_work = False
        if self.shap == "skorch":
            print(
                "WARNING: For shap='skorch' the additivity check tends to fail, "
                "you set set shap_kwargs=dict(check_additivity=False) to supress "
                "this error (at your own risk)!"
            )

        self.model_output = model_output

        if idxs is not None:
            assert len(idxs) == len(self.X) == len(self.y), (
                "idxs should be same length as X but is not: "
                f"len(idxs)={len(idxs)} but  len(X)={len(self.X)}!"
            )
            self.idxs = pd.Index(idxs).astype(str)
        else:
            self.idxs = X.index.astype(str)
        self.X.reset_index(drop=True, inplace=True)
        self.y.reset_index(drop=True, inplace=True)

        self._index_exists_func = None
        self._get_index_list_func = None
        self._get_X_row_func = None
        self._get_y_func = None

        if index_name is None:
            if self.idxs.name is not None:
                self.index_name = self.idxs.name.capitalize()
            else:
                self.index_name = "Index"
        else:
            self.idxs.name = index_name.capitalize()
            self.index_name = index_name.capitalize()
        self.descriptions = {} if descriptions is None else descriptions
        if not isinstance(self.descriptions, dict):
            raise ValueError(
                "ERROR: parameter descriptions should be a dict with feature names as keys, "
                "and feature descriptions as values, but you passed a "
                f"{type(self.descriptions)}!"
            )
        self.target = target if target is not None else self.y.name
        self.n_jobs = n_jobs
        self.cv = cv
        self.na_fill = na_fill
        self.precision = precision
        self.columns = self.X.columns
        self.pos_label = None
        self.units = ""
        self.is_classifier = False
        self.is_regression = False
        if safe_isinstance(self.model, "CatBoostRegressor", "CatBoostClassifier"):
            self.interactions_should_work = False
        if not hasattr(self, "interactions_should_work"):
            self.interactions_should_work = True

        self.__version__ = "0.4.0"

    def get_lock(self):
        if not hasattr(self, "_lock"):
            self._lock = Lock()
        return self._lock

    @classmethod
    def from_file(cls, filepath):
        """Load an Explainer from file. Depending on the suffix of the filepath
        will either load with pickle ('.pkl'), dill ('.dill') or joblib ('joblib').

        If no suffix given, will try with joblib.

        Args:
            filepath {str, Path} the location of the stored Explainer

        returns:
            Explainer object
        """
        filepath = Path(filepath)
        if str(filepath).endswith(".pkl") or str(filepath).endswith(".pickle"):
            import pickle

            return pickle.load(open(filepath, "rb"))
        elif str(filepath).endswith(".dill"):
            import dill

            return dill.load(open(filepath, "rb"))
        else:
            if not filepath.exists():
                if (filepath.parent / (filepath.name + ".joblib")).exists():
                    filepath = filepath.parent / (filepath.name + ".joblib")
                else:
                    raise ValueError(f"Cannot find file: {str(filepath)}")
            import joblib

            return joblib.load(filepath)

    def dump(self, filepath):
        """
        Dump the current Explainer to file. Depending on the suffix of the filepath
        will either dump with pickle ('.pkl'), dill ('.dill') or joblib ('joblib').

        If no suffix given, will dump with joblib and add '.joblib'

        Args:
            filepath (str, Path): filepath where to save the Explainer.
        """
        filepath = Path(filepath)
        if self.shap == "kernel" and not str(filepath).endswith(".dill"):
            print(
                "Warning! KernelExplainer does not work with joblib or pickle, "
                "but only with dill, so specify e.g. filepath='explainer.dill' "
                "to use dill instead of joblib or pickle.",
                flush=True,
            )
        if hasattr(self, "_lock"):
            del self._lock  # Python Locks are not picklable
        if str(filepath).endswith(".pkl") or str(filepath).endswith(".pickle"):
            import pickle

            pickle.dump(self, open(str(filepath), "wb"))
        elif str(filepath).endswith(".dill"):
            import dill

            dill.dump(self, open(str(filepath), "wb"))
        elif str(filepath).endswith(".joblib"):
            import joblib

            joblib.dump(self, filepath)
        else:
            filepath = Path(filepath)
            filepath = filepath.parent / (filepath.name + ".joblib")
            import joblib

            joblib.dump(self, filepath)

    def to_yaml(
        self,
        filepath=None,
        return_dict=False,
        modelfile="model.pkl",
        datafile="data.csv",
        index_col=None,
        target_col=None,
        explainerfile="explainer.joblib",
        dashboard_yaml="dashboard.yaml",
    ):
        """Returns a yaml configuration for the current Explainer
        that can be used by the explainerdashboard CLI. Recommended filename
        is `explainer.yaml`.

        Args:
            filepath ({str, Path}, optional): Filepath to dump yaml. If None
                returns the yaml as a string. Defaults to None.
            return_dict (bool, optional): instead of yaml return dict with config.
            modelfile (str, optional): filename of model dump. Defaults to
                `model.pkl`
            datafile (str, optional): filename of datafile. Defaults to
                `data.csv`.
            index_col (str, optional): column to be used for idxs. Defaults to
                self.idxs.name.
            target_col (str, optional): column to be used for to split X and y
                from datafile. Defaults to self.target.
            explainerfile (str, optional): filename of explainer dump. Defaults
                to `explainer.joblib`.
            dashboard_yaml (str, optional): filename of the dashboard.yaml
                configuration file. This will be used to determine which
                properties to calculate before storing to disk.
                Defaults to `dashboard.yaml`.
        """
        import oyaml as yaml

        yaml_config = dict(
            explainer=dict(
                modelfile=modelfile,
                datafile=datafile,
                explainerfile=explainerfile,
                data_target=self.target,
                data_index=self.idxs.name,
                explainer_type="classifier" if self.is_classifier else "regression",
                dashboard_yaml=dashboard_yaml,
                params=self._params_dict,
            )
        )
        if return_dict:
            return yaml_config

        if filepath is not None:
            yaml.dump(yaml_config, open(filepath, "w"))
            return
        return yaml.dump(yaml_config)

    def __len__(self):
        return len(self.X)

    def __contains__(self, index):
        try:
            if self.get_idx(index) is not None:
                return True
        except IndexNotFoundError:
            return False
        return False

    def get_idx(self, index):
        """Turn str index into an int index

        Args:
          index(str or int):

        Returns:
            int index
        """
        if isinstance(index, int):
            if index >= 0 and index < len(self):
                return index
        elif isinstance(index, str):
            if self.idxs is not None and index in self.idxs:
                return self.idxs.get_loc(index)
        raise IndexNotFoundError(index=index)

    def get_index(self, index):
        """Turn int index into a str index

        Args:
          index(str or int):

        Returns:
            str index
        """
        if isinstance(index, int) and index >= 0 and index < len(self):
            return self.idxs[index]
        elif isinstance(index, str) and index in self.idxs:
            return index
        return None

    @property
    def X_cats(self):
        """X with categorical variables grouped together"""
        if not hasattr(self, "_X_cats"):
            self._X_cats = merge_categorical_columns(
                self.X,
                self.onehot_dict,
                not_encoded_dict=self.onehot_notencoded,
                drop_regular=True,
            )
        return self._X_cats

    @property
    def X_merged(self, index=None):
        if index is None:
            if self.X_cats.empty:
                return self.X[self.merged_cols]
            return self.X.merge(self.X_cats, left_index=True, right_index=True)[
                self.merged_cols
            ]
        else:
            if self.X_cats.empty:
                return self.X[index][self.merged_cols]
            return self.X[index].merge(
                self.X_cats[index], left_index=True, right_index=True
            )[self.merged_cols]

    @property
    def n_features(self):
        """number of features

        Returns:
            int, number of features
        """
        return len(self.merged_cols)

    def index_exists(self, index):
        if isinstance(index, int):
            if index >= 0 and index < len(self):
                return True
        if isinstance(index, str):
            if index in self.idxs:
                return True
            if self._get_index_list_func is not None and index in self.get_index_list():
                return True
            if self._index_exists_func is not None and self._index_exists_func(index):
                return True
        return False

    def set_index_exists_func(self, func):
        """Sets an external function to check whether an index is valid or not.

        func should either be a function that takes a single parameter: def func(index)
        or a method that takes a single parameter: def func(self, index)
        """
        assert callable(
            func
        ), f"{func} is not a callable! pass either a function or a method!"
        argspec = inspect.getfullargspec(func).args
        if argspec == ["self", "index"]:
            self._index_exists_func = MethodType(func, self)
        elif argspec == ["index"]:
            self._index_exists_func = func
        else:
            raise ValueError(
                f"Parameter func should either be a function {func.__name__}(index) "
                f"or a method {func.__name__}(self, index)! Instead you "
                f"passed func={func.__name__}{inspect.signature(func)}"
            )

    def get_index_list(self) -> pd.Series:
        if self._get_index_list_func is not None:
            if not hasattr(self, "_index_list"):
                self._index_list = pd.Index(self._get_index_list_func()).astype(str)
            return self._index_list
        else:
            return self.idxs

    def reset_index_list(self):
        """resets the available indexes using the function provided by explainer.set_index_list_func()"""
        if self._get_index_list_func is not None:
            self._index_list = pd.Index(self._get_index_list_func())

    def set_index_list_func(self, func):
        """Sets an external function all available indexes from an external source.

        func should either be a parameterless function: def func(): ...
        or a parameterless method: def func(self): ...
        """
        assert callable(
            func
        ), f"{func} is not a callable! pass either a function or a method!"
        argspec = inspect.getfullargspec(func).args
        if argspec == ["self"]:
            self._get_index_list_func = MethodType(func, self)
        elif argspec == []:
            self._get_index_list_func = func
        else:
            raise ValueError(
                f"Parameter func should either be a function {func.__name__}() "
                f"or a method {func.__name__}(self)! Instead you "
                f"passed func={func.__name__}{inspect.signature(func)}"
            )

    def get_X_row(self, index, merge=False):
        if index in self.idxs:
            X_row = self.X.iloc[[self.get_idx(index)]]
        elif isinstance(index, int) and index >= 0 and index < len(self):
            X_row = self.X.iloc[[index]]
        elif self._get_X_row_func is not None and self.index_exists(index):
            X_row = self._get_X_row_func(index)
        else:
            raise IndexNotFoundError(index=index)

        if not matching_cols(X_row.columns, self.columns):
            raise ValueError(
                f"columns do not match! Got {X_row.columns}, but was"
                f"expecting {self.columns}"
            )
        if merge:
            X_row = merge_categorical_columns(
                X_row, self.onehot_dict, not_encoded_dict=self.onehot_notencoded
            )[self.merged_cols]
        return X_row

    def set_X_row_func(self, func):
        """Sets an external function to retrieve a row of input data a given index.

        func should either be a function that takes a single parameter: def func(index)
        or a method that takes a single parameter: def func(self, index)
        """
        assert callable(
            func
        ), f"{func} is not a callable! pass either a function or a method!"
        argspec = inspect.getfullargspec(func).args
        if argspec == ["self", "index"]:
            self._get_X_row_func = MethodType(func, self)
        elif argspec == ["index"]:
            self._get_X_row_func = func
        else:
            raise ValueError(
                f"Parameter func should either be a function {func.__name__}(index) "
                f"or a method {func.__name__}(self, index)! Instead you "
                f"passed func={func.__name__}{inspect.signature(func)}"
            )

    def get_y(self, index):
        if index in self.idxs:
            if self.y_missing:
                return None
            return self.y.iloc[[self.get_idx(index)]].item()
        elif isinstance(index, int) and index >= 0 and index < len(self):
            return self.y.iloc[[index]].item()
        elif self._get_y_func is not None and self.index_exists(index):
            y = self._get_y_func(index)
            if isinstance(y, pd.Series) or isinstance(y, np.ndarray):
                try:
                    return y.item()
                except:
                    raise ValueError(f"Can't turn y into a single item: {y}")
        else:
            raise IndexNotFoundError(index=index)

    def set_y_func(self, func):
        """Sets an external function to retrieve an observed label for a given index.

        func should either be a function that takes a single parameter: def func(index)
        or a method that takes a single parameter: def func(self, index)
        """
        assert callable(
            func
        ), f"{func} is not a callable! pass either a function or a method!"
        argspec = inspect.getfullargspec(func).args
        if argspec == ["self", "index"]:
            self._get_y_func = func = MethodType(func, self)
        elif argspec == ["index"]:
            self._get_y_func = func = func
        else:
            raise ValueError(
                f"Parameter func should either be a function {func.__name__}(index) "
                f"or a method {func.__name__}(self, index)! Instead you "
                f"passed func={func.__name__}{inspect.signature(func)}"
            )

    def get_row_from_input(
        self, inputs: List, ranked_by_shap=False, return_merged=False
    ):
        """returns a single row pd.DataFrame from a given list of *inputs"""
        if len(inputs) == 1 and isinstance(inputs[0], list):
            inputs = inputs[0]
        elif len(inputs) == 1 and isinstance(inputs[0], tuple):
            inputs = list(inputs[0])
        else:
            inputs = list(inputs)

        if len(inputs) == len(self.merged_cols):
            cols = self.columns_ranked_by_shap() if ranked_by_shap else self.merged_cols
            with pd.option_context("future.no_silent_downcasting", True):
                df_merged = (
                    pd.DataFrame(dict(zip(cols, inputs)), index=[0])
                    .fillna(self.na_fill)
                    .infer_objects(copy=False)[self.merged_cols]
                )
            if return_merged:
                return df_merged
            else:
                return X_cats_to_X(df_merged, self.onehot_dict, self.columns)

        elif len(inputs) == len(self.columns):
            cols = self.columns
            df = pd.DataFrame(dict(zip(cols, inputs)), index=[0]).fillna(self.na_fill)
            if return_merged:
                return merge_categorical_columns(df, self.onehot_dict, self.merged_cols)
            else:
                return df
        else:
            raise ValueError(
                f"len inputs {len(inputs)} should be the same length as either "
                f"explainer.merged_cols ({len(self.merged_cols)}) or "
                f"explainer.columns ({len(self.columns)})!"
            )

    def get_col(self, col):
        """return pd.Series with values of col

        For categorical feature reverse engineers the onehotencoding.

        Args:
          col: column tof values to be returned

        Returns:
          pd.Series with values of col

        """
        assert col in self.columns or col in self.onehot_cols, f"{col} not in columns!"

        if col in self.onehot_cols:
            return self.X_cats[col]
        else:
            return self.X[col]

    @insert_pos_label
    def get_col_value_plus_prediction(
        self, col, index=None, X_row=None, pos_label=None
    ):
        """return value of col and prediction for either index or X_row

        Args:
          col: feature col
          index (str or int, optional): index row
          X_row (single row pd.DataFrame, optional): single row of features
          pos_label (int): positive label

        Returns:
          value of col, prediction for index

        """
        assert (col in self.X.columns) or (
            col in self.onehot_cols
        ), f"{col} not in columns of dataset"
        if index is not None:
            X_row = self.get_X_row(index)
        if X_row is not None:
            assert X_row.shape[0] == 1, "X_Row should be single row dataframe!"

            if matching_cols(X_row.columns, self.merged_cols):
                col_value = X_row[col].item()
                X_row = X_cats_to_X(X_row, self.onehot_dict, self.columns)
            else:
                assert matching_cols(
                    X_row.columns, self.columns
                ), "X_row should have the same columns as explainer.columns or explainer.merged_cols!"
                if col in self.onehot_cols:
                    col_value = retrieve_onehot_value(
                        X_row, col, self.onehot_dict[col], self.onehot_notencoded[col]
                    ).item()
                else:
                    col_value = X_row[col].item()
            if self.shap == "skorch":
                X_row = X_row.values.astype("float32")
            if self.is_classifier:
                if pos_label is None:
                    pos_label = self.pos_label
                prediction = self.model.predict_proba(X_row)[0][pos_label].squeeze()
                if self.model_output == "probability":
                    prediction = 100 * prediction
            elif self.is_regression:
                prediction = self.model.predict(X_row)[0].squeeze()
            return col_value, prediction
        else:
            raise ValueError("You need to pass either index or X_row!")

    def description(self, col):
        """returns the written out description of what feature col means

        Args:
          col(str): col to get description for

        Returns:
            str, description
        """
        if col in self.descriptions.keys():
            return self.descriptions[col]
        elif col in self.encoded_cols:
            cat_col = [k for k, v in self.onehot_dict.items() if col in v][0]
            if cat_col in self.descriptions.keys():
                return self.descriptions[cat_col]
        return ""

    def description_list(self, cols):
        """returns a list of descriptions of a list of cols

        Args:
          cols(list): cols to be converted to descriptions

        Returns:
            list of descriptions
        """
        return [self.description(col) for col in cols]

    def get_descriptions_df(self, sort: str = "alphabet") -> pd.DataFrame:
        """returns a dataframe with features and their descriptions.

        Args:
            sort (str, optional): sort either by 'alphabet' or be mean absolute
                shap values ('shap')

        Returns:
            pd.DataFrame
        """
        if sort == "alphabet":
            cols = self.merged_cols.sort_values()
        elif sort == "shap":
            cols = self.columns_ranked_by_shap()
        else:
            raise ValueError(
                "get_description_df() parameter sort should be either"
                f"'alphabet' or 'shap', but you passed {sort}!"
            )
        return pd.DataFrame(dict(Feature=cols, Description=self.description_list(cols)))

    def ordered_cats(self, col, topx=None, sort="alphabet", pos_label=None):
        """Return a list of categories in an categorical column, sorted
        by mode.

        Args:
            col (str): Categorical feature to return categories for.
            topx (int, optional): Return topx top categories. Defaults to None.
            sort (str, optional): Sorting method, either alphabetically ('alphabet'),
                by frequency ('freq') or mean absolute shap ('shap').
                Defaults to 'alphabet'.

        Raises:
            ValueError: if sort is other than 'alphabet', 'freq', 'shap

        Returns:
            list
        """
        if pos_label is None:
            pos_label = self.pos_label
        assert col in self.cat_cols, f"{col} is not a categorical feature!"
        if col in self.onehot_cols:
            X = self.X_cats
        else:
            X = self.X

        if sort == "alphabet":
            if topx is None:
                return sorted(X[col].unique().tolist())
            else:
                return sorted(X[col].unique().tolist())[:topx]
        elif sort == "freq":
            if topx is None:
                return X[col].value_counts().index.tolist()
            else:
                return X[col].value_counts().nlargest(topx).index.tolist()
        elif sort == "shap":
            if topx is None:
                return (
                    pd.Series(
                        self.get_shap_values_df(pos_label)[col].values,
                        index=self.get_col(col),
                    )
                    .abs()
                    .groupby(level=0)
                    .mean()
                    .sort_values(ascending=False)
                    .index.tolist()
                )
            else:
                return (
                    pd.Series(
                        self.get_shap_values_df(pos_label)[col].values,
                        index=self.get_col(col),
                    )
                    .abs()
                    .groupby(level=0)
                    .mean()
                    .sort_values(ascending=False)
                    .nlargest(topx)
                    .index.tolist()
                )
        else:
            raise ValueError(
                f"sort='{sort}', but should be in {{'alphabet', 'freq', 'shap'}}"
            )

    @property
    def preds(self):
        """returns model model predictions"""
        if not hasattr(self, "_preds"):
            print("Calculating predictions...", flush=True)
            if self.shap == "skorch":  # skorch model.predict need np.array
                self._preds = (
                    self.model.predict(self.X.values).squeeze().astype(self.precision)
                )
            else:  # Pipelines.predict need pd.DataFrame:
                self._preds = (
                    self.model.predict(self.X).squeeze().astype(self.precision)
                )

        return self._preds

    @insert_pos_label
    def pred_percentiles(self, pos_label=None):
        """returns percentile rank of model predictions"""
        if not hasattr(self, "_pred_percentiles"):
            print("Calculating prediction percentiles...", flush=True)
            self._pred_percentiles = (
                pd.Series(self.preds).rank(method="min").divide(len(self.preds)).values
            ).astype(self.precision)
        return self._pred_percentiles

    @insert_pos_label
    def permutation_importances(self, pos_label=None):
        """Permutation importances"""
        if not hasattr(self, "_perm_imps"):
            print("Calculating importances...", flush=True)
            self._perm_imps = cv_permutation_importances(
                self.model,
                self.X,
                self.y,
                self.metric,
                onehot_dict=self.onehot_dict,
                cv=self.cv,
                n_jobs=self.n_jobs,
                needs_proba=self.is_classifier,
                pass_nparray=(self.shap == "skorch"),
            ).sort_values("Importance", ascending=False)
            self._perm_imps = self._perm_imps
        return self._perm_imps

    @insert_pos_label
    def get_permutation_importances_df(self, topx=None, cutoff=None, pos_label=None):
        """dataframe with features ordered by permutation importance.

        For more about permutation importances.

        see https://explained.ai/rf-importance/index.html

        Args:
          topx(int, optional, optional): only return topx most important
                features, defaults to None
          cutoff(float, optional, optional): only return features with importance
                of at least cutoff, defaults to None
          pos_label:  (Default value = None)

        Returns:
          pd.DataFrame: importance_df

        """
        importance_df = self.permutation_importances(pos_label)

        if topx is None:
            topx = len(importance_df)
        if cutoff is None:
            cutoff = importance_df.Importance.min()
        return importance_df[importance_df.Importance >= cutoff].head(topx)

    @property
    def shap_explainer(self):
        """ """
        if not hasattr(self, "_shap_explainer"):
            X_str = ", X_background" if self.X_background is not None else "X"
            NoX_str = ", X_background" if self.X_background is not None else ""
            if self.shap == "tree":
                print(
                    "Generating self.shap_explainer = "
                    f"shap.TreeExplainer(model{NoX_str})"
                )
                self._shap_explainer = shap.TreeExplainer(self.model)
            elif self.shap == "linear":
                if self.X_background is None:
                    print(
                        "Warning: shap values for shap.LinearExplainer get "
                        "calculated against X_background, but paramater "
                        "X_background=None, so using X instead"
                    )
                print(
                    f"Generating self.shap_explainer = shap.LinearExplainer(model{X_str})..."
                )
                self._shap_explainer = shap.LinearExplainer(
                    self.model,
                    self.X_background if self.X_background is not None else self.X,
                )
            elif self.shap == "deep":
                print(
                    "Generating self.shap_explainer = "
                    "shap.DeepExplainer(model, X_background)"
                )
                print(
                    "Warning: shap values for shap.DeepExplainer get "
                    "calculated against X_background, but paramater "
                    "X_background=None, so using shap.sample(X, 5) instead"
                )
                self._shap_explainer = shap.DeepExplainer(
                    self.model,
                    self.X_background
                    if self.X_background is not None
                    else shap.sample(self.X, 5),
                )
            elif self.shap == "skorch":
                print(
                    "Generating self.shap_explainer = "
                    "shap.DeepExplainer(model, X_background)"
                )
                print(
                    "Warning: shap values for shap.DeepExplainer get "
                    "calculated against X_background, but paramater "
                    "X_background=None, so using shap.sample(X, 5) instead"
                )
                import torch

                self._shap_explainer = shap.DeepExplainer(
                    self.model.module_,
                    torch.tensor(self.X_background.values)
                    if self.X_background is not None
                    else torch.tensor(shap.sample(self.X, 5).values),
                )
            elif self.shap == "kernel":
                if self.X_background is None:
                    print(
                        "Warning: shap values for shap.KernelExplainer get "
                        "calculated against X_background, but paramater "
                        "X_background=None, so using shap.sample(X, 50) instead"
                    )
                print(
                    "Generating self.shap_explainer = "
                    f"shap.KernelExplainer(model, {X_str})..."
                )

                def model_predict(data_asarray):
                    data_asframe = pd.DataFrame(data_asarray, columns=self.columns)
                    preds = self.model.predict(data_asframe)
                    return preds.reshape(len(preds))

                self._shap_explainer = shap.KernelExplainer(
                    model_predict,
                    self.X_background
                    if self.X_background is not None
                    else shap.sample(self.X, 50),
                )
        return self._shap_explainer

    @insert_pos_label
    def shap_base_value(self, pos_label=None):
        """the intercept for the shap values.

        (i.e. 'what would the prediction be if we knew none of the features?')
        """
        if not hasattr(self, "_shap_base_value"):
            # CatBoost needs shap values calculated before expected value
            if not hasattr(self, "_shap_values"):
                _ = self.get_shap_values_df()
            self._shap_base_value = self.shap_explainer.expected_value
            if isinstance(self._shap_base_value, np.ndarray):
                # shap library now returns an array instead of float
                self._shap_base_value = self._shap_base_value.item()
        return self._shap_base_value

    @insert_pos_label
    def get_shap_values_df(self, pos_label=None):
        """SHAP values calculated using the shap library"""
        if not hasattr(self, "_shap_values_df"):
            print("Calculating shap values...", flush=True)
            if self.shap == "skorch":
                import torch

                self._shap_values_df = pd.DataFrame(
                    self.shap_explainer.shap_values(
                        torch.tensor(self.X.values), **self.shap_kwargs
                    ),
                    columns=self.columns,
                )
            else:
                self._shap_values_df = pd.DataFrame(
                    self.shap_explainer.shap_values(self.X, **self.shap_kwargs),
                    columns=self.columns,
                )
            self._shap_values_df = merge_categorical_shap_values(
                self._shap_values_df, self.onehot_dict, self.merged_cols
            ).astype(self.precision)
        return self._shap_values_df

    def set_shap_values(self, base_value: float, shap_values: np.ndarray):
        """Set shap values manually. This is useful if you already have
        shap values calculated, and do not want to calculate them again inside
        the explainer instance. Especially for large models and large datasets
        you may want to calculate shap values on specialized hardware, and then
        add them to the explainer manually.

        Args:
            base_value (float): the shap intercept generated by e.g.
                base_value = shap.TreeExplainer(model).shap_values(X_test).expected_value
            shap_values (np.ndarray]): Generated by e.g.
                shap_values = shap.TreeExplainer(model).shap_values(X_test)
        """
        self._shap_base_value = base_value
        self._shap_values_df = pd.DataFrame(shap_values, columns=self.columns)
        self._shap_values_df = merge_categorical_shap_values(
            self._shap_values_df, self.onehot_dict, self.merged_cols
        ).astype(self.precision)

    @insert_pos_label
    def get_shap_row(self, index=None, X_row=None, pos_label=None):
        if index is not None:
            if index in self.idxs:
                shap_row = self.get_shap_values_df().iloc[[self.idxs.get_loc(index)]]
            elif isinstance(index, int) and index >= 0 and index < len(self):
                shap_row = self.get_shap_values_df().iloc[[index]]
            elif self._get_X_row_func is not None and self.index_exists(index):
                X_row = self._get_X_row_func(index)
                if self.shap == "skorch":
                    import torch

                    X_row = torch.tensor(X_row.values.astype("float32"))
                with self.get_lock():
                    shap_kwargs = (
                        dict(self.shap_kwargs, silent=True)
                        if self.shap == "kernel"
                        else self.shap_kwargs
                    )
                    shap_row = pd.DataFrame(
                        self.shap_explainer.shap_values(X_row, **self.shap_kwargs),
                        columns=self.columns,
                    )
                shap_row = merge_categorical_shap_values(
                    shap_row, self.onehot_dict, self.merged_cols
                )
            else:
                raise IndexNotFoundError(index=index)
        elif X_row is not None:
            if self.shap == "skorch":
                import torch

                X_row = torch.tensor(X_row.values.astype("float32"))
            with self.get_lock():
                shap_kwargs = (
                    dict(self.shap_kwargs, silent=True)
                    if self.shap == "kernel"
                    else self.shap_kwargs
                )
                shap_row = pd.DataFrame(
                    self.shap_explainer.shap_values(X_row, **self.shap_kwargs),
                    columns=self.columns,
                )
            shap_row = merge_categorical_shap_values(
                shap_row, self.onehot_dict, self.merged_cols
            )
        else:
            raise ValueError("you should either pas index or X_row!")
        return shap_row

    @insert_pos_label
    def shap_interaction_values(self, pos_label=None):
        """SHAP interaction values calculated using shap library"""
        assert self.shap != "linear", (
            "Unfortunately shap.LinearExplainer does not provide "
            "shap interaction values! So no interactions tab!"
        )
        if not hasattr(self, "_shap_interaction_values"):
            print("Calculating shap interaction values...", flush=True)
            if self.shap == "tree":
                print(
                    "Reminder: TreeShap computational complexity is O(TLD^2), "
                    "where T is the number of trees, L is the maximum number of"
                    " leaves in any tree and D the maximal depth of any tree. So "
                    "reducing these will speed up the calculation.",
                    flush=True,
                )
            self._shap_interaction_values = self.shap_explainer.shap_interaction_values(
                self.X
            )
            self._shap_interaction_values = merge_categorical_shap_interaction_values(
                self._shap_interaction_values,
                self.columns,
                self.merged_cols,
                self.onehot_dict,
            ).astype(self.precision)
        return self._shap_interaction_values

    def set_shap_interaction_values(self, shap_interaction_values: np.ndarray):
        """Manually set shap interaction values in case you have already pre-computed
        these elsewhere and do not want to re-calculate them again inside the
        explainer instance.

        Args:
            shap_interaction_values (np.ndarray): shap interactions values of shape (n, m, m)

        """
        if not isinstance(shap_interaction_values, np.ndarray):
            raise ValueError("shap_interaction_values should be a numpy array")
        if not shap_interaction_values.shape == (
            len(self.X),
            len(self.original_cols),
            len(self.original_cols),
        ):
            raise ValueError(
                "shap interaction_values should be of shape "
                f"({len(self.X)}, {len(self.original_cols)}, {len(self.original_cols)})!"
            )

        self._shap_interaction_values = merge_categorical_shap_interaction_values(
            shap_interaction_values, self.columns, self.merged_cols, self.onehot_dict
        ).astype(self.precision)

    @insert_pos_label
    def mean_abs_shap_df(self, pos_label=None):
        """Mean absolute SHAP values per feature."""
        if not hasattr(self, "_mean_abs_shap_df"):
            self._mean_abs_shap_df = (
                self.get_shap_values_df(pos_label)[self.merged_cols]
                .abs()
                .mean()
                .sort_values(ascending=False)
                .to_frame()
                .rename_axis(index="Feature")
                .reset_index()
                .rename(columns={0: "MEAN_ABS_SHAP"})
            )
        return self._mean_abs_shap_df

    @insert_pos_label
    def columns_ranked_by_shap(self, pos_label=None):
        """returns the columns of X, ranked by mean abs shap value

        Args:
        cats: Group categorical together (Default value = False)
        pos_label:  (Default value = None)

        Returns:
        list of columns

        """
        return self.mean_abs_shap_df(pos_label).Feature.tolist()

    @insert_pos_label
    def get_mean_abs_shap_df(self, topx=None, cutoff=None, pos_label=None):
        """sorted dataframe with mean_abs_shap

        returns a pd.DataFrame with the mean absolute shap values per features,
        sorted rom highest to lowest.

        Args:
          topx(int, optional, optional): Only return topx most importance
            features, defaults to None
          cutoff(float, optional, optional): Only return features with mean
            abs shap of at least cutoff, defaults to None
          pos_label:  (Default value = None)

        Returns:
          pd.DataFrame: shap_df

        """
        shap_df = self.mean_abs_shap_df(pos_label)
        if topx is None:
            topx = len(shap_df)
        if cutoff is None:
            cutoff = shap_df["MEAN_ABS_SHAP"].min()
        return shap_df[shap_df["MEAN_ABS_SHAP"] >= cutoff].head(topx)

    @insert_pos_label
    def top_shap_interactions(self, col, topx=None, pos_label=None):
        """returns the features that interact with feature col in descending order.

        if shap interaction values have already been calculated, use those.
        Otherwise use shap approximate_interactions or simply mean abs shap.

        Args:
          col(str): feature for which you want to get the interactions
          topx(int, optional, optional): Only return topx features, defaults to None
          cats(bool, optional, optional): Group categorical features, defaults to False
          pos_label:  (Default value = None)

        Returns:
          list: top_interactions

        """
        if hasattr(self, "_shap_interaction_values"):
            col_idx = self.merged_cols.get_loc(col)
            order = np.argsort(
                -np.abs(self.shap_interaction_values(pos_label)[:, col_idx, :]).mean(0)
            )
            top_interactions = self.merged_cols[order].tolist()
        else:
            top_interactions = self.columns_ranked_by_shap()
            top_interactions.insert(
                0, top_interactions.pop(top_interactions.index(col))
            )  # put col first

        if topx is None:
            return top_interactions
        else:
            return top_interactions[:topx]

    @insert_pos_label
    def shap_interaction_values_for_col(self, col, interact_col=None, pos_label=None):
        """returns the shap interaction values[np.array(N,N)] for feature col

        Args:
          col(str): features for which you'd like to get the interaction value
          pos_label:  (Default value = None)

        Returns:
          np.array(N,N): shap_interaction_values

        """
        assert col in self.merged_cols, f"{col} not in self.merged_cols!"
        if interact_col is None:
            return self.shap_interaction_values(pos_label)[
                :, self.merged_cols.get_loc(col), :
            ]
        else:
            assert (
                interact_col in self.merged_cols
            ), f"{interact_col} not in self.merged_cols!"
            return self.shap_interaction_values(pos_label)[
                :, self.merged_cols.get_loc(col), self.merged_cols.get_loc(interact_col)
            ]

    def calculate_properties(self, include_interactions=True):
        """Explicitely calculates all lazily calculated properties.
        Useful so that properties are not calculate multiple times in
        parallel when starting a dashboard.

        Args:
          include_interactions(bool, optional, optional): shap interaction values can take a long
        time to compute for larger datasets with more features. Therefore you
        can choose not to calculate these, defaults to True

        Returns:

        """
        _ = (
            self.preds,
            self.pred_percentiles(),
            self.shap_base_value(),
            self.get_shap_values_df(),
            self.get_mean_abs_shap_df(),
        )
        if not self.y_missing:
            _ = self.get_permutation_importances_df()
        if self.onehot_cols:
            _ = self.X_cats
        if self.interactions_should_work and include_interactions:
            _ = self.shap_interaction_values

    def memory_usage(self, cutoff=0):
        """returns a pd.DataFrame witht the memory usage of each attribute of
        this explainer object"""

        def get_size(obj):
            def get_inner_size(obj):
                if isinstance(obj, pd.DataFrame):
                    return obj.memory_usage().sum()
                elif isinstance(obj, pd.Series):
                    return obj.memory_usage()
                elif isinstance(obj, pd.Index):
                    return obj.memory_usage()
                elif isinstance(obj, np.ndarray):
                    return obj.nbytes
                else:
                    return sys.getsizeof(obj)

            if isinstance(obj, list):
                return sum([get_inner_size(o) for o in obj])
            elif isinstance(obj, dict):
                return sum([get_inner_size(o) for o in obj.values()])
            else:
                return get_inner_size(obj)

        def size_to_string(num, suffix="B"):
            for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
                if np.abs(num) < 1024.0:
                    return "%3.1f%s%s" % (num, unit, suffix)
                num /= 1024.0
            return "%.1f%s%s" % (num, "Yi", suffix)

        memory_df = pd.DataFrame(columns=["property", "type", "bytes", "size"])
        for k, v in self.__dict__.items():
            memory_df = append_dict_to_df(
                memory_df,
                dict(
                    property=f"self.{k}",
                    type=v.__class__.__name__,
                    bytes=get_size(v),
                    size=size_to_string(get_size(v)),
                ),
            )

        print(
            "Explainer total memory usage (approximate): ",
            size_to_string(memory_df.bytes.sum()),
            flush=True,
        )
        return (
            memory_df[memory_df.bytes > cutoff]
            .sort_values("bytes", ascending=False)
            .reset_index(drop=True)
        )

    def random_index(
        self,
        y_min=None,
        y_max=None,
        pred_min=None,
        pred_max=None,
        return_str=False,
        **kwargs,
    ):
        """random index following constraints

        Args:
          y_min:  (Default value = None)
          y_max:  (Default value = None)
          pred_min:  (Default value = None)
          pred_max:  (Default value = None)
          return_str:  (Default value = False)
          **kwargs:

        Returns:
          if y_values is given select an index for which y in y_values
          if return_str return str index from self.idxs
        """

        if pred_min is None:
            pred_min = self.preds.min()
        if pred_max is None:
            pred_max = self.preds.max()

        if not self.y_missing:
            if y_min is None:
                y_min = self.y.min()
            if y_max is None:
                y_max = self.y.max()

            potential_idxs = self.y[
                (self.y >= y_min)
                & (self.y <= y_max)
                & (self.preds >= pred_min)
                & (self.preds <= pred_max)
            ].index
        else:
            potential_idxs = self.y[
                (self.preds >= pred_min) & (self.preds <= pred_max)
            ].index

        if len(potential_idxs) > 0:
            idx = np.random.choice(potential_idxs)
        else:
            return None
        if return_str:
            return self.idxs[idx]
        return idx

    def metrics(self, *args, **kwargs):
        """returns a dict of metrics.

        Implemented by either ClassifierExplainer or RegressionExplainer
        """
        return {}

    @insert_pos_label
    def get_importances_df(self, kind="shap", topx=None, cutoff=None, pos_label=None):
        """wrapper function for get_mean_abs_shap_df() and get_permutation_importance_df()

        Args:
          kind(str): 'shap' or 'permutations'  (Default value = "shap")
          topx: only display topx highest features (Default value = None)
          cutoff: only display features above cutoff (Default value = None)
          pos_label: Positive class (Default value = None)

        Returns:
          pd.DataFrame

        """
        assert (
            kind == "shap" or kind == "permutation"
        ), "kind should either be 'shap' or 'permutation'!"
        if kind == "permutation":
            return self.get_permutation_importances_df(topx, cutoff, pos_label)
        elif kind == "shap":
            return self.get_mean_abs_shap_df(topx, cutoff, pos_label)

    @insert_pos_label
    def get_contrib_df(
        self, index=None, X_row=None, topx=None, cutoff=None, sort="abs", pos_label=None
    ):
        """shap value contributions to the prediction for index.

        Used as input for the plot_contributions() method.

        Args:
          index(int or str): index for which to calculate contributions
          X_row (pd.DataFrame, single row): single row of feature for which
                to calculate contrib_df. Can us this instead of index
          topx(int, optional): Only return topx features, remainder
                    called REST, defaults to None
          cutoff(float, optional): only return features with at least
                    cutoff contributions, defaults to None
          sort({'abs', 'high-to-low', 'low-to-high', 'importance'}, optional): sort by
                    absolute shap value, or from high to low, low to high, or
                    ordered by the global shap importances.
                    Defaults to 'abs'.
          pos_label:  (Default value = None)

        Returns:
          pd.DataFrame: contrib_df

        """
        if index is None and X_row is None:
            raise ValueError("Either index or X_row should be passed!")
        if sort == "importance":
            if cutoff is None:
                cols = self.columns_ranked_by_shap()
            else:
                cols = (
                    self.mean_abs_shap_df()
                    .query(f"MEAN_ABS_SHAP > {cutoff}")
                    .Feature.tolist()
                )
            if topx is not None:
                cols = cols[:topx]
        else:
            cols = None
        if index is not None:
            X_row_merged = self.get_X_row(index, merge=True)
            shap_values = self.get_shap_row(index, pos_label=pos_label)
        elif X_row is not None:
            if matching_cols(X_row.columns, self.merged_cols):
                X_row_merged = X_row
                X_row = X_cats_to_X(X_row, self.onehot_dict, self.X.columns)
            else:
                assert matching_cols(
                    X_row.columns, self.columns
                ), "X_row should have the same columns as self.X or self.merged_cols!"
                X_row_merged = merge_categorical_columns(
                    X_row,
                    self.onehot_dict,
                    not_encoded_dict=self.onehot_notencoded,
                    drop_regular=False,
                )[self.merged_cols]
            shap_values = self.get_shap_row(X_row=X_row, pos_label=pos_label)

        return get_contrib_df(
            shap_base_value=self.shap_base_value(pos_label),
            shap_values=shap_values.values[0],
            X_row=remove_cat_names(
                X_row_merged, self.onehot_dict, self.onehot_notencoded
            ),
            topx=topx,
            cutoff=cutoff,
            sort=sort,
            cols=cols,
        )

    @insert_pos_label
    def get_contrib_summary_df(
        self,
        index=None,
        X_row=None,
        topx=None,
        cutoff=None,
        round=2,
        sort="abs",
        pos_label=None,
    ):
        """Takes a contrib_df, and formats it to a more human readable format

        Args:
          index: index to show contrib_summary_df for
          X_row (pd.DataFrame, single row): single row of feature for which
                to calculate contrib_df. Can us this instead of index
          topx: Only show topx highest features(Default value = None)
          cutoff: Only show features above cutoff (Default value = None)
          round: round figures (Default value = 2)
          sort({'abs', 'high-to-low', 'low-to-high', 'importance'}, optional): sort by
                    absolute shap value, or from high to low, or low to high, or
                    ordered by the global shap importances.
                    Defaults to 'abs'.
          pos_label: Positive class (Default value = None)

        Returns:
          pd.DataFrame
        """
        contrib_df = self.get_contrib_df(index, X_row, topx, cutoff, sort, pos_label)
        return get_contrib_summary_df(
            contrib_df,
            model_output=self.model_output,
            round=round,
            units=self.units,
            na_fill=self.na_fill,
        )

    @insert_pos_label
    def get_interactions_df(self, col, topx=None, cutoff=None, pos_label=None):
        """dataframe of mean absolute shap interaction values for col

        Args:
          col: Feature to get interactions_df for
          topx: Only display topx most important features (Default value = None)
          cutoff: Only display features with mean abs shap of at least cutoff (Default value = None)
          pos_label: Positive class  (Default value = None)

        Returns:
          pd.DataFrame

        """
        importance_df = get_mean_absolute_shap_df(
            self.merged_cols,
            self.shap_interaction_values_for_col(col, pos_label=pos_label),
        )

        if topx is None:
            topx = len(importance_df)
        if cutoff is None:
            cutoff = importance_df["MEAN_ABS_SHAP"].min()
        return importance_df[importance_df["MEAN_ABS_SHAP"] >= cutoff].head(topx)

    @insert_pos_label
    def pdp_df(
        self,
        col,
        index=None,
        X_row=None,
        drop_na=True,
        sample=500,
        n_grid_points=10,
        pos_label=None,
        sort="freq",
    ):
        """Return a pdp_df for generating partial dependence plots.

        Args:
            col (str): Feature to generate partial dependence for.
            index ({int, str}, optional): Index to include on first row
                of pdp_df. Defaults to None.
            X_row (pd.DataFrame, optional): Single row to put on first row of pdp_df.
                Defaults to None.
            drop_na (bool, optional): Drop self.na_fill values. Defaults to True.
            sample (int, optional): Sample size for pdp_df. Defaults to 500.
            n_grid_points (int, optional): Number of grid points on x axis.
                Defaults to 10.
            pos_label ([type], optional): [description]. Defaults to None.
            sort (str, optional): For categorical features: how to sort:
             'alphabet', 'freq', 'shap'. Defaults to 'freq'.

        Returns:
            pd.DataFrame
        """
        assert (
            col in self.X.columns or col in self.onehot_cols
        ), f"{col} not in columns of dataset"
        if col in self.onehot_cols:
            grid_values = self.ordered_cats(col, n_grid_points, sort)
            if index is not None or X_row is not None:
                val, pred = self.get_col_value_plus_prediction(col, index, X_row)
                if val not in grid_values:
                    grid_values[-1] = val
            features = self.onehot_dict[col]
        elif col in self.categorical_cols:
            features = col
            grid_values = self.ordered_cats(col, n_grid_points, sort)
            if index is not None or X_row is not None:
                val, pred = self.get_col_value_plus_prediction(col, index, X_row)
                if val not in grid_values:
                    grid_values[-1] = val
        else:
            features = col
            if drop_na:
                vals = np.delete(
                    self.X[col].values,
                    np.where(self.X[col].values == self.na_fill),
                    axis=0,
                )
                grid_values = get_grid_points(vals, n_grid_points=n_grid_points)
            else:
                grid_values = get_grid_points(
                    self.X[col].values, n_grid_points=n_grid_points
                )
            if index is not None or X_row is not None:
                val, pred = self.get_col_value_plus_prediction(col, index, X_row)
                if val not in grid_values:
                    grid_values = np.sort(np.append(grid_values, val))

        if index is not None:
            X_row = self.get_X_row(index)
        if X_row is not None:
            if matching_cols(X_row.columns, self.merged_cols):
                X_row = X_cats_to_X(X_row, self.onehot_dict, self.X.columns)
            else:
                assert matching_cols(
                    X_row.columns, self.columns
                ), "X_row should have the same columns as self.X or self.merged_cols!"

            if isinstance(features, str) and drop_na:  # regular col, not onehotencoded
                sample_size = min(
                    sample, len(self.X[(self.X[features] != self.na_fill)]) - 1
                )
                sampleX = pd.concat(
                    [
                        X_row,
                        self.X[(self.X[features] != self.na_fill)].sample(sample_size),
                    ],
                    ignore_index=True,
                    axis=0,
                )
            else:
                sample_size = min(sample, len(self.X) - 1)
                sampleX = pd.concat(
                    [X_row, self.X.sample(sample_size)], ignore_index=True, axis=0
                )
        else:
            if isinstance(features, str) and drop_na:  # regular col, not onehotencoded
                sample_size = min(
                    sample, len(self.X[(self.X[features] != self.na_fill)]) - 1
                )
                sampleX = self.X[(self.X[features] != self.na_fill)].sample(sample_size)
            else:
                sampleX = self.X.sample(min(sample, len(self.X)))

        pdp_df = get_pdp_df(
            model=self.model,
            X_sample=sampleX,
            feature=features,
            n_grid_points=n_grid_points,
            pos_label=pos_label,
            grid_values=grid_values,
            is_classifier=self.is_classifier,
            cast_to_float32=(self.shap == "skorch"),
        )

        if all([str(c).startswith(col + "_") for c in pdp_df.columns]):
            pdp_df.columns = [str(c)[len(col) + 1 :] for c in pdp_df.columns]
        if self.is_classifier and self.model_output == "probability":
            pdp_df = pdp_df.multiply(100)
        return pdp_df

    @insert_pos_label
    def plot_importances(self, kind="shap", topx=None, round=3, pos_label=None):
        """plot barchart of importances in descending order.

        Args:
          type(str, optional): shap' for mean absolute shap values, 'permutation' for
                    permutation importances, defaults to 'shap'
          topx(int, optional, optional): Only return topx features, defaults to None
          kind:  (Default value = 'shap')
          round:  (Default value = 3)
          pos_label:  (Default value = None)

        Returns:
          plotly.fig: fig

        """
        importances_df = self.get_importances_df(
            kind=kind, topx=topx, pos_label=pos_label
        )
        if kind == "shap":
            if self.target:
                title = f"Average impact on predicted {self.target}<br>(mean absolute SHAP value)"
            else:
                title = "Average impact on prediction<br>(mean absolute SHAP value)"

            units = self.units
        else:
            title = f"Permutation Importances <br>(decrease in metric '{self.metric.__name__}'' with randomized feature)"
            units = ""
        if self.descriptions:
            descriptions = self.description_list(importances_df.Feature)
            return plotly_importances_plot(
                importances_df, descriptions, round=round, units=units, title=title
            )
        else:
            return plotly_importances_plot(
                importances_df, round=round, units=units, title=title
            )

    @insert_pos_label
    def plot_importances_detailed(
        self,
        highlight_index=None,
        topx=None,
        max_cat_colors=5,
        plot_sample=None,
        pos_label=None,
    ):
        """Plot barchart of mean absolute shap value.

        Displays all individual shap value for each feature in a horizontal
        scatter chart in descending order by mean absolute shap value.

        Args:
          highlight_index (str or int): index to highlight
          topx(int, optional): Only display topx most important features,
            defaults to None
          max_cat_colors (int, optional): for categorical features, maximum number
            of categories to label with own color. Defaults to 5.
          plot_sample (int, optional): Instead of all points only plot a random
            sample of points. Defaults to None (=all points)
          pos_label: positive class (Default value = None)

        Returns:
          plotly.Fig

        """
        plot_idxs = self.get_idx_sample(plot_sample, highlight_index)
        highlight_index = self.get_index(highlight_index)

        if self.is_classifier:
            pos_label_str = self.labels[pos_label]
            if self.model_output == "probability":
                if self.target:
                    title = f"Impact of feature on predicted probability {self.target}={pos_label_str} <br> (SHAP values)"
                else:
                    title = (
                        "Impact of Feature on Prediction probability <br> (SHAP values)"
                    )
            elif self.model_output == "logodds":
                title = "Impact of Feature on predicted logodds <br> (SHAP values)"
        elif self.is_regression:
            if self.target:
                title = (
                    f"Impact of Feature on Predicted {self.target} <br> (SHAP values)"
                )
            else:
                title = "Impact of Feature on Prediction<br> (SHAP values)"

        cols = self.get_importances_df(kind="shap", topx=topx, pos_label=pos_label)[
            "Feature"
        ].values.tolist()

        return plotly_shap_scatter_plot(
            self.X_merged[cols].iloc[plot_idxs],
            self.get_shap_values_df(pos_label)[cols].iloc[plot_idxs],
            cols,
            idxs=self.idxs[plot_idxs],
            highlight_index=highlight_index,
            title=title,
            na_fill=self.na_fill,
            max_cat_colors=max_cat_colors,
        )

    @insert_pos_label
    def plot_contributions(
        self,
        index=None,
        X_row=None,
        topx=None,
        cutoff=None,
        sort="abs",
        orientation="vertical",
        higher_is_better=True,
        round=2,
        pos_label=None,
    ):
        """plot waterfall plot of shap value contributions to the model prediction for index.

        Args:
            index(int or str): index for which to display prediction
            X_row (pd.DataFrame single row): a single row of a features to plot
                shap contributions for. Can use this instead of index for
                what-if scenarios.
            topx(int, optional, optional): Only display topx features,
                        defaults to None
            cutoff(float, optional, optional): Only display features with at least
                        cutoff contribution, defaults to None
            sort({'abs', 'high-to-low', 'low-to-high', 'importance'}, optional):
                sort by absolute shap value, or from high to low,
                or low to high, or by order of shap feature importance.
                Defaults to 'abs'.
            orientation({'vertical', 'horizontal'}): Horizontal or vertical bar chart.
                    Horizontal may be better if you have lots of features.
                    Defaults to 'vertical'.
            higher_is_better (bool): if True, up=green, down=red. If false reversed.
                Defaults to True.
            round(int, optional, optional): round contributions to round precision,
                        defaults to 2
            pos_label:  (Default value = None)

        Returns:
            plotly.Fig: fig

        """
        assert orientation in ["vertical", "horizontal"]
        contrib_df = self.get_contrib_df(
            index=index,
            X_row=X_row,
            topx=topx,
            cutoff=cutoff,
            sort=sort,
            pos_label=pos_label,
        )
        return plotly_contribution_plot(
            contrib_df,
            model_output=self.model_output,
            orientation=orientation,
            round=round,
            higher_is_better=higher_is_better,
            target=self.target,
            units=self.units,
        )

    def get_idx_sample(
        self,
        sample_size=None,
        include_index=None,
        outlier_array1=None,
        outlier_array2=None,
    ):
        """returns a random sample of integer indexes, making sure that
        include_index is included. Outlier indexes can be excluded.

        Args:
            sample_size: Number of (random) samples to return
            include_index: index that has to be included, independent of random draw
            outlier_array1: array to exclude all indexes with values <> 1.5*IQR from.
            outlier_array2: array to exclude all indexes with values <> 1.5*IQR from.
        """

        idx_sample = np.arange(0, len(self))
        if sample_size is None and outlier_array1 is None and outlier_array2 is None:
            return idx_sample
        else:
            if outlier_array1 is not None:
                q1, q3 = np.nanpercentile(outlier_array1, [25, 75])
                lb, ub = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
                idx_sample = idx_sample[(outlier_array1 >= lb) & (outlier_array1 <= ub)]
            if outlier_array2 is not None:
                q1, q3 = np.nanpercentile(outlier_array2[idx_sample], [25, 75])
                lb, ub = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
                idx_sample = idx_sample[
                    (outlier_array2[idx_sample] >= lb)
                    & (outlier_array2[idx_sample] <= ub)
                ]

            if sample_size is not None and sample_size < len(idx_sample):
                assert sample_size >= 0, "sample_size should be a positive integer!"
                idx_sample = np.random.choice(idx_sample, sample_size, replace=False)

            if include_index is not None:
                if isinstance(include_index, str):
                    if include_index not in self.idxs:
                        raise ValueError(f"{include_index} could not be found in idxs!")
                    include_index = self.idxs.get_loc(include_index)
                if include_index not in idx_sample and include_index < len(self):
                    idx_sample = np.append(idx_sample, include_index)
            return idx_sample

    @insert_pos_label
    def plot_dependence(
        self,
        col,
        color_col=None,
        highlight_index=None,
        topx=None,
        sort="alphabet",
        max_cat_colors=5,
        round=3,
        plot_sample=None,
        remove_outliers=False,
        pos_label=None,
    ):
        """plot shap dependence

        Plots a shap dependence plot:
            - on the x axis the possible values of the feature `col`
            - on the y axis the associated individual shap values

        Args:
          col(str): feature to be displayed
          color_col(str): if color_col provided then shap values colored (blue-red)
                    according to feature color_col (Default value = None)
          highlight_index: individual observation to be highlighed in the plot.
                    (Default value = None)
          topx (int, optional): for categorical features only display topx
                categories.
          sort (str): for categorical features, how to sort the categories:
                alphabetically 'alphabet', most frequent first 'freq',
                highest mean absolute value first 'shap'. Defaults to 'alphabet'.
          max_cat_colors (int, optional): for categorical features, maximum number
                of categories to label with own color. Defaults to 5.
          round (int, optional): rounding to apply to floats. Defaults to 3.
          plot_sample (int, optional): Instead of all points only plot a random
            sample of points. Defaults to None (=all points)
          remove_outliers (bool, optional): remove observations that are >1.5*IQR
            in either col or color_col. Defaults to False.
          pos_label: positive class (Default value = None)

        Returns:

        """

        plot_idxs = self.get_idx_sample(
            plot_sample,
            highlight_index,
            self.get_col(col).values
            if remove_outliers and col not in self.cat_cols
            else None,
            self.get_col(color_col).values
            if remove_outliers
            and color_col is not None
            and color_col not in self.cat_cols
            else None,
        )
        highlight_index = self.get_index(highlight_index)

        if color_col is None:
            X_color_col = None
        else:
            X_color_col = self.get_col(color_col).iloc[plot_idxs]

        if col in self.cat_cols:
            return plotly_shap_violin_plot(
                self.get_col(col).iloc[plot_idxs],
                self.get_shap_values_df(pos_label)[col].iloc[plot_idxs].values,
                X_color_col,
                highlight_index=highlight_index,
                idxs=self.idxs[plot_idxs],
                round=round,
                cats_order=self.ordered_cats(col, topx, sort),
                max_cat_colors=max_cat_colors,
            )
        else:
            return plotly_dependence_plot(
                self.get_col(col).iloc[plot_idxs],
                self.get_shap_values_df(pos_label)[col].iloc[plot_idxs].values,
                X_color_col,
                na_fill=self.na_fill,
                units=self.units,
                highlight_index=highlight_index,
                idxs=self.idxs[plot_idxs],
                round=round,
            )

    @insert_pos_label
    def plot_interaction(
        self,
        col,
        interact_col,
        highlight_index=None,
        topx=10,
        sort="alphabet",
        max_cat_colors=5,
        plot_sample=None,
        remove_outliers=False,
        pos_label=None,
    ):
        """plots a dependence plot for shap interaction effects

        Args:
          col(str): feature for which to find interaction values
          interact_col(str): feature for which interaction value are displayed
          highlight_index(str, optional): index that will be highlighted, defaults to None
          topx (int, optional): number of categorical features to display in violin plots.
          sort (str, optional): how to sort categorical features in violin plots.
                Should be in {'alphabet', 'freq', 'shap'}.
          max_cat_colors (int, optional): for categorical features, maximum number
                of categories to label with own color. Defaults to 5.
          plot_sample (int, optional): Instead of all points only plot a random
            sample of points. Defaults to None (=all points)
          remove_outliers (bool, optional): remove observations that are >1.5*IQR
            in either col or color_col. Defaults to False.
          pos_label:  (Default value = None)

        Returns:
          plotly.Fig: Plotly Fig

        """
        plot_idxs = self.get_idx_sample(
            plot_sample,
            highlight_index,
            self.get_col(col).values
            if remove_outliers and col not in self.cat_cols
            else None,
            self.get_col(interact_col).values
            if remove_outliers and interact_col not in self.cat_cols
            else None,
        )
        highlight_index = self.get_index(highlight_index)

        if col in self.cat_cols:
            return plotly_shap_violin_plot(
                self.get_col(col).iloc[plot_idxs],
                self.shap_interaction_values_for_col(
                    col, interact_col, pos_label=pos_label
                )[plot_idxs],
                self.get_col(interact_col).iloc[plot_idxs],
                interaction=True,
                units=self.units,
                highlight_index=highlight_index,
                idxs=self.idxs[plot_idxs],
                cats_order=self.ordered_cats(col, topx, sort),
                max_cat_colors=max_cat_colors,
            )
        else:
            return plotly_dependence_plot(
                self.get_col(col).iloc[plot_idxs],
                self.shap_interaction_values_for_col(
                    col, interact_col, pos_label=pos_label
                )[plot_idxs],
                self.get_col(interact_col).iloc[plot_idxs],
                interaction=True,
                units=self.units,
                highlight_index=highlight_index,
                idxs=self.idxs[plot_idxs],
            )

    @insert_pos_label
    def plot_interactions_importance(self, col, topx=None, pos_label=None):
        """plot mean absolute shap interaction value for col.

        Args:
          col: column for which to generate shap interaction value
          topx(int, optional, optional): Only return topx features, defaults to None
          pos_label:  (Default value = None)

        Returns:
          plotly.fig: fig

        """
        interactions_df = self.get_interactions_df(col, topx=topx, pos_label=pos_label)
        title = f"Average interaction shap values for {col}"
        return plotly_importances_plot(interactions_df, units=self.units, title=title)

    @insert_pos_label
    def plot_interactions_detailed(
        self,
        col,
        highlight_index=None,
        topx=None,
        max_cat_colors=5,
        plot_sample=None,
        pos_label=None,
    ):
        """Plot barchart of mean absolute shap interaction values

        Displays all individual shap interaction values for each feature in a
        horizontal scatter chart in descending order by mean absolute shap value.

        Args:
          col(type]): feature for which to show interactions summary
          highlight_index (str or int): index to highlight
          topx(int, optional): only show topx most important features, defaults to None
          max_cat_colors (int, optional): for categorical features, maximum number
            of categories to label with own color. Defaults to 5.
          plot_sample (int, optional): Instead of all points only plot a random
            sample of points. Defaults to None (=all points)
          pos_label: positive class (Default value = None)

        Returns:
          fig
        """
        plot_idxs = self.get_idx_sample(plot_sample, highlight_index)
        highlight_index = self.get_index(highlight_index)

        interact_cols = self.top_shap_interactions(col, pos_label=pos_label)
        shap_df = pd.DataFrame(
            self.shap_interaction_values_for_col(col, pos_label=pos_label),
            columns=self.merged_cols,
        ).iloc[plot_idxs]
        if topx is None:
            topx = len(interact_cols)
        title = f"Shap interaction values for {col}"
        return plotly_shap_scatter_plot(
            self.X_merged.iloc[plot_idxs],
            shap_df,
            interact_cols[:topx],
            title=title,
            idxs=self.idxs[plot_idxs],
            highlight_index=highlight_index,
            na_fill=self.na_fill,
            max_cat_colors=max_cat_colors,
        )

    @insert_pos_label
    def plot_pdp(
        self,
        col,
        index=None,
        X_row=None,
        drop_na=True,
        sample=100,
        gridlines=100,
        gridpoints=10,
        sort="freq",
        round=2,
        pos_label=None,
    ):
        """plot partial dependence plot (pdp)

        returns plotly fig for a partial dependence plot showing ice lines
        for num_grid_lines rows, average pdp based on sample of sample.
        If index is given, display pdp for this specific index.

        Args:
          col(str): feature to display pdp graph for
          index(int or str, optional, optional): index to highlight in pdp graph,
                    defaults to None
          X_row (pd.Dataframe, single row, optional): a row of features to highlight
                predictions for. Alternative to passing index.
          drop_na(bool, optional, optional): if true drop samples with value
                    equal to na_fill, defaults to True
          sample(int, optional, optional): sample size on which the average
                    pdp will be calculated, defaults to 100
          gridlines(int, optional): number of ice lines to display,
                    defaults to 100
          gridpoints(ints: int, optional): number of points on the x axis
                    to calculate the pdp for, defaults to 10
          sort (str, optional): For categorical features: how to sort:
             'alphabet', 'freq', 'shap'. Defaults to 'freq'.
          round (int, optional): round float prediction to number of digits.
            Defaults to 2.
          pos_label:  (Default value = None)

        Returns:
          plotly.Fig: fig

        """
        pdp_df = self.pdp_df(
            col,
            index,
            X_row,
            drop_na=drop_na,
            sample=sample,
            n_grid_points=gridpoints,
            pos_label=pos_label,
            sort=sort,
        )
        units = "Predicted %" if self.model_output == "probability" else self.units
        if index is not None or X_row is not None:
            col_value, pred = self.get_col_value_plus_prediction(
                col, index=index, X_row=X_row, pos_label=pos_label
            )
            if (
                col in self.cat_cols
                and col_value not in pdp_df.columns
                and col_value[len(col) + 1 :] in pdp_df.columns
            ):
                col_value = col_value[len(col) + 1 :]
            return plotly_pdp(
                pdp_df,
                display_index=0,  # the idx to be displayed is always set to the first row by self.pdp_df()
                index_feature_value=col_value,
                index_prediction=pred,
                feature_name=col,
                num_grid_lines=min(gridlines, sample, len(self.X)),
                round=round,
                target=self.target,
                units=units,
            )
        else:
            return plotly_pdp(
                pdp_df,
                feature_name=col,
                num_grid_lines=min(gridlines, sample, len(self.X)),
                round=round,
                target=self.target,
                units=units,
            )


class ClassifierExplainer(BaseExplainer):
    """ """

    def __init__(
        self,
        model,
        X: pd.DataFrame,
        y: pd.Series = None,
        permutation_metric: Callable = roc_auc_score,
        shap: str = "guess",
        X_background: pd.DataFrame = None,
        model_output: str = "probability",
        cats: Union[List, Dict] = None,
        cats_notencoded: Dict = None,
        idxs: pd.Index = None,
        index_name: str = None,
        target: str = None,
        descriptions: Dict = None,
        n_jobs: int = None,
        permutation_cv: int = None,
        cv: int = None,
        na_fill: float = -999,
        precision: str = "float64",
        shap_kwargs: Dict = None,
        labels: List = None,
        pos_label: int = 1,
    ):
        """
        Explainer for classification models. Defines the shap values for
        each possible class in the classification.

        You assign the positive label class afterwards with e.g. explainer.pos_label=0

        In addition defines a number of plots specific to classification problems
        such as a precision plot, confusion matrix, roc auc curve and pr auc curve.

        Compared to BaseExplainer defines two additional parameters

        Args:
            model: a model with a scikit-learn compatible .fit and .predict methods
            X (pd.DataFrame): a pd.DataFrame with your model features
            y (pd.Series): Dependent variable of your model, defaults to None
            permutation_metric (function or str): is a scikit-learn compatible
                metric function (or string). Defaults to r2_score
            shap (str): type of shap_explainer to fit: 'tree', 'linear', 'kernel'.
                Defaults to 'guess'.
            X_background (pd.DataFrame): background X to be used by shap
                explainers that need a background dataset (e.g. shap.KernelExplainer
                or shap.TreeExplainer with boosting models and
                model_output='probability').
            model_output (str): model_output of shap values, either 'raw',
                'logodds' or 'probability'. Defaults to 'raw' for regression and
                'probability' for classification.
            cats ({dict, list}): dict of features that have been
                onehotencoded. e.g. cats={'Sex':['Sex_male', 'Sex_female']}.
                If all encoded columns are underscore-seperated (as above), can simply
                pass a list of prefixes: cats=['Sex']. Allows to
                group onehot encoded categorical variables together in
                various plots. Defaults to None.
            cats_notencoded (dict): value to display when all onehot encoded
                columns are equal to zero. Defaults to 'NOT_ENCODED' for each
                onehot col.
            idxs (pd.Series): list of row identifiers. Can be names, id's, etc.
                Defaults to X.index.
            index_name (str): identifier for row indexes. e.g. index_name='Passenger'.
                Defaults to X.index.name or idxs.name.
            target: name of the predicted target, e.g. "Survival",
                "Ticket price", etc. Defaults to y.name.
            n_jobs (int): for jobs that can be parallelized using joblib,
                how many processes to split the job in. For now only used
                for calculating permutation importances. Defaults to None.
            permutation_cv (int): Deprecated! Use parameter cv instead!
                (now also works for calculating metrics)
            cv (int): If not None then permutation importances and metrics
                will get calculated using cross validation across X. Use this
                when you are passing the training set to the explainer.
                Defaults to None.
            na_fill (int): The filler used for missing values, defaults to -999.
            precision: precision with which to store values. Defaults to "float64".
            shap_kwargs(dict): dictionary of keyword arguments to be passed to the shap explainer.
                most typically used to supress an additivity check e.g. `shap_kwargs=dict(check_additivity=False)`
            labels(list): list of str labels for the different classes,
                        defaults to e.g. ['0', '1'] for a binary classification
            pos_label: class that should be used as the positive class,
                        defaults to 1
        """
        super().__init__(
            model,
            X,
            y,
            permutation_metric,
            shap,
            X_background,
            model_output,
            cats,
            cats_notencoded,
            idxs,
            index_name,
            target,
            descriptions,
            n_jobs,
            permutation_cv,
            cv,
            na_fill,
            precision,
            shap_kwargs,
        )

        assert hasattr(model, "predict_proba"), (
            "for ClassifierExplainer, model should be a scikit-learn "
            "compatible *classifier* model that has a predict_proba(...) "
            f"method, so not a {type(model)}! If you are using e.g an SVM "
            "with hinge loss (which does not support predict_proba), you "
            "can try the following monkey patch:\n\n"
            "import types\n"
            "def predict_proba(self, X):\n"
            "    pred = self.predict(X)\n"
            "    return np.array([1-pred, pred]).T\n"
            "model.predict_proba = types.MethodType(predict_proba, model)\n"
        )

        self._params_dict = {
            **self._params_dict,
            **dict(labels=labels, pos_label=pos_label),
        }

        if not self.y_missing:
            self.y = self.y.astype("int16")
        if (
            self.categorical_cols
            and model_output == "probability"
            and not isinstance(self.model, Pipeline)
        ):
            print(
                "Warning: Models that deal with categorical features directly "
                f"such as {self.model.__class__.__name__} are incompatible with model_output='probability'"
                " for now. So setting model_output='logodds'...",
                flush=True,
            )
            self.model_output = "logodds"
        if labels is not None:
            self.labels = labels
        elif hasattr(self.model, "classes_"):
            self.labels = [str(cls) for cls in self.model.classes_]
        else:
            self.labels = [str(i) for i in range(self.y.nunique())]
        self.pos_label = pos_label
        self.is_classifier = True
        if safe_isinstance(
            self.model, "RandomForestClassifier", "ExtraTreesClassifier"
        ):
            print(
                "Detected RandomForestClassifier model: "
                "Changing class type to RandomForestClassifierExplainer...",
                flush=True,
            )
            self.__class__ = RandomForestClassifierExplainer
        if str(type(self.model)).endswith("XGBClassifier'>"):
            print(
                "Detected XGBClassifier model: "
                "Changing class type to XGBClassifierExplainer...",
                flush=True,
            )
            self.__class__ = XGBClassifierExplainer
            if len(self.labels) > 2 and self.model_output == "probability":
                print(
                    "model_output=='probability' does not work with multiclass "
                    "XGBClassifier models, so settings model_output='logodds'..."
                )
                self.model_output = "logodds"

        _ = self.shap_explainer

    @property
    def pos_label(self):
        return self._pos_label

    @pos_label.setter
    def pos_label(self, label):
        if label is None or (
            isinstance(label, int) and label >= 0 and label < len(self.labels)
        ):
            self._pos_label = label
        elif isinstance(label, str) and label in self.labels:
            self._pos_label = self.pos_label_index(label)
        else:
            raise ValueError(f"'{label}' not in labels")

    @property
    def pos_label_str(self):
        """return str label of self.pos_label"""
        return self.labels[self.pos_label]

    def pos_label_index(self, pos_label):
        """return int index of pos_label_str"""
        if isinstance(pos_label, int):
            assert pos_label >= 0 and pos_label <= len(
                self.labels
            ), f"pos_label={pos_label}, but should be >= 0 and <= {len(self.labels)-1}!"
            return pos_label
        elif isinstance(pos_label, str):
            assert (
                pos_label in self.labels
            ), f"Unknown pos_label. {pos_label} not in self.labels!"
            return self.labels.index(pos_label)
        raise ValueError("pos_label should either be int or str in self.labels!")

    @insert_pos_label
    def y_binary(self, pos_label):
        """for multiclass problems returns one-vs-rest array of [1,0] pos_label"""
        if not hasattr(self, "_y_binaries"):
            if not self.y_missing:
                self._y_binaries = [
                    np.where(self.y.values == i, 1, 0) for i in range(self.y.nunique())
                ]
            else:
                self._y_binaries = [self.y.values for i in range(len(self.labels))]
        return self._y_binaries[pos_label]

    @property
    def pred_probas_raw(self):
        """returns pred_probas with probability for each class"""
        if not hasattr(self, "_pred_probas"):
            print("Calculating prediction probabilities...", flush=True)
            assert hasattr(
                self.model, "predict_proba"
            ), "model does not have a predict_proba method!"
            if self.shap == "skorch":
                self._pred_probas = self.model.predict_proba(self.X.values).astype(
                    self.precision
                )
            else:
                warnings.filterwarnings("ignore", category=UserWarning)
                self._pred_probas = self.model.predict_proba(self.X).astype(
                    self.precision
                )
                warnings.filterwarnings("default", category=UserWarning)
        return self._pred_probas

    @property
    def pred_percentiles_raw(self):
        """ """
        if not hasattr(self, "_pred_percentiles_raw"):
            print("Calculating pred_percentiles...", flush=True)
            self._pred_percentiles_raw = (
                pd.DataFrame(self.pred_probas_raw)
                .rank(method="min")
                .divide(len(self.pred_probas_raw))
                .values
            )
        return self._pred_percentiles_raw

    @insert_pos_label
    def pred_probas(self, pos_label=None):
        """returns pred_proba for pos_label class"""
        return self.pred_probas_raw[:, pos_label]

    @insert_pos_label
    def pred_percentiles(self, pos_label=None):
        """returns ranks for pos_label class"""
        return self.pred_percentiles_raw[:, pos_label]

    @insert_pos_label
    def permutation_importances(self, pos_label=None):
        """Permutation importances"""
        if not hasattr(self, "_perm_imps"):
            print(
                "Calculating permutation importances (if slow, try setting n_jobs parameter)...",
                flush=True,
            )
            self._perm_imps = [
                cv_permutation_importances(
                    self.model,
                    self.X,
                    self.y,
                    self.metric,
                    onehot_dict=self.onehot_dict,
                    cv=self.cv,
                    needs_proba=self.is_classifier,
                    pos_label=label,
                    pass_nparray=(self.shap == "skorch"),
                ).sort_values("Importance", ascending=False)
                for label in range(len(self.labels))
            ]

        return self._perm_imps[pos_label]

    @property
    def shap_explainer(self):
        """Initialize SHAP explainer.

        Taking into account model type and model_output
        """
        if not hasattr(self, "_shap_explainer"):
            model_str = (
                str(type(self.model))
                .replace("'", "")
                .replace("<", "")
                .replace(">", "")
                .split(".")[-1]
            )
            if self.shap == "tree":
                if safe_isinstance(
                    self.model,
                    "XGBClassifier",
                    "LGBMClassifier",
                    "CatBoostClassifier",
                    "GradientBoostingClassifier",
                    "HistGradientBoostingClassifier",
                ):
                    if self.model_output == "probability":
                        if self.X_background is None:
                            print(
                                f"Note: model_output=='probability'. For {model_str} shap values normally get "
                                "calculated against X_background, but paramater X_background=None, "
                                "so using X instead"
                            )
                        print(
                            "Generating self.shap_explainer = shap.TreeExplainer(model, "
                            f"{'X_background' if self.X_background is not None else 'X'}"
                            ", model_output='probability', feature_perturbation='interventional')..."
                        )
                        print(
                            "Note: Shap interaction values will not be available. "
                            "If shap values in probability space are not necessary you can "
                            "pass model_output='logodds' to get shap values in logodds without the need for "
                            "a background dataset and also working shap interaction values..."
                        )
                        self._shap_explainer = shap.TreeExplainer(
                            self.model,
                            self.X_background
                            if self.X_background is not None
                            else self.X,
                            model_output="probability",
                            feature_perturbation="interventional",
                        )
                        self.interactions_should_work = False
                    else:
                        self.model_output = "logodds"
                        print(
                            f"Generating self.shap_explainer = shap.TreeExplainer(model{', X_background' if self.X_background is not None else ''})"
                        )
                        self._shap_explainer = shap.TreeExplainer(
                            self.model, self.X_background
                        )
                else:
                    if self.model_output == "probability":
                        print(
                            f"Note: model_output=='probability', so assuming that raw shap output of {model_str} is in probability space..."
                        )
                    print(
                        f"Generating self.shap_explainer = shap.TreeExplainer(model{', X_background' if self.X_background is not None else ''})"
                    )
                    self._shap_explainer = shap.TreeExplainer(
                        self.model, self.X_background
                    )

            elif self.shap == "linear":
                if self.model_output == "probability":
                    print(
                        "Note: model_output='probability' is currently not supported for linear classifiers "
                        "models with shap. So defaulting to model_output='logodds' "
                        "If you really need probability outputs use shap='kernel' instead."
                    )
                    self.model_output = "logodds"
                if self.X_background is None:
                    print(
                        "Note: shap values for shap='linear' get calculated against "
                        "X_background, but paramater X_background=None, so using X instead..."
                    )
                print(
                    "Generating self.shap_explainer = shap.LinearExplainer(model, "
                    f"{'X_background' if self.X_background is not None else 'X'})..."
                )

                self._shap_explainer = shap.LinearExplainer(
                    self.model,
                    self.X_background if self.X_background is not None else self.X,
                )
            elif self.shap == "deep":
                print(
                    "Generating self.shap_explainer = "
                    "shap.DeepExplainer(model, X_background)"
                )
                print(
                    "Warning: shap values for shap.DeepExplainer get "
                    "calculated against X_background, but paramater "
                    "X_background=None, so using shap.sample(X, 5) instead"
                )
                self._shap_explainer = shap.DeepExplainer(
                    self.model,
                    self.X_background
                    if self.X_background is not None
                    else shap.sample(self.X, 5),
                )
            elif self.shap == "skorch":
                import torch

                print(
                    "Generating self.shap_explainer = "
                    "shap.DeepExplainer(model, X_background)"
                )
                print(
                    "Warning: shap values for shap.DeepExplainer get "
                    "calculated against X_background, but paramater "
                    "X_background=None, so using shap.sample(X, 5) instead"
                )
                self._shap_explainer = shap.DeepExplainer(
                    self.model.module_,
                    torch.tensor(
                        self.X_background.values
                        if self.X_background is not None
                        else shap.sample(self.X, 5).values
                    ),
                )
            elif self.shap == "kernel":
                if self.X_background is None:
                    print(
                        "Note: shap values for shap='kernel' normally get calculated against "
                        "X_background, but paramater X_background=None, so setting "
                        "X_background=shap.sample(X, 50)..."
                    )
                if self.model_output != "probability":
                    print(
                        "Note: for ClassifierExplainer shap='kernel' defaults to model_output='probability"
                    )
                    self.model_output = "probability"
                print(
                    "Generating self.shap_explainer = shap.KernelExplainer(model, "
                    f"{'X_background' if self.X_background is not None else 'X'}"
                    ", link='identity')"
                )

                def model_predict(data_asarray):
                    data_asframe = pd.DataFrame(data_asarray, columns=self.columns)
                    return self.model.predict_proba(data_asframe)

                self._shap_explainer = shap.KernelExplainer(
                    model_predict,
                    self.X_background
                    if self.X_background is not None
                    else shap.sample(self.X, 50),
                    link="identity",
                )
        return self._shap_explainer

    @insert_pos_label
    def shap_base_value(self, pos_label=None):
        """SHAP base value: average outcome of population"""
        if not hasattr(self, "_shap_base_value"):
            _ = self.get_shap_values_df()  # CatBoost needs to have shap values calculated before expected value for some reason
            self._shap_base_value = self.shap_explainer.expected_value
            if (
                isinstance(self._shap_base_value, np.ndarray)
                and len(self._shap_base_value) == 1
            ):
                self._shap_base_value = self._shap_base_value[0]
            if isinstance(self._shap_base_value, np.ndarray):
                self._shap_base_value = list(self._shap_base_value)
            if len(self.labels) == 2 and isinstance(
                self._shap_base_value, (np.floating, float)
            ):
                if self.model_output == "probability":
                    self._shap_base_value = [
                        1 - self._shap_base_value,
                        self._shap_base_value,
                    ]
                else:  # assume logodds
                    self._shap_base_value = [
                        -self._shap_base_value,
                        self._shap_base_value,
                    ]
            assert len(self._shap_base_value) == len(self.labels), (
                f"len(shap_explainer.expected_value)={len(self._shap_base_value)}"
                + f"and len(labels)={len(self.labels)} do not match!"
            )
            if self.model_output == "probability":
                for shap_base_value in self._shap_base_value:
                    assert shap_base_value >= 0.0 and shap_base_value <= 1.0, (
                        f"Shap base value does not look like a probability: {self._shap_base_value}. "
                        "Try setting model_output='logodds'."
                    )
        return self._shap_base_value[pos_label]

    @insert_pos_label
    def get_shap_values_df(self, pos_label=None):
        """SHAP Values"""
        if not hasattr(self, "_shap_values_df"):
            print("Calculating shap values...", flush=True)
            if self.shap == "skorch":
                import torch

                _shap_values = self.shap_explainer.shap_values(
                    torch.tensor(self.X.values.astype("float32")), **self.shap_kwargs
                )
            else:
                _shap_values = self.shap_explainer.shap_values(
                    self.X.values, **self.shap_kwargs
                )

            if len(self.labels) == 2:
                if (
                    isinstance(_shap_values, np.ndarray)
                    and len(_shap_values.shape) == 3
                    and _shap_values.shape[2] == 2
                ):
                    # for binary classifier only keep positive class:
                    _shap_values = _shap_values[:, :, 1]
                elif (
                    isinstance(_shap_values, np.ndarray)
                    and len(_shap_values.shape) == 3
                    and _shap_values.shape[2] > 2
                ):
                    raise Exception(
                        f"len(self.label)={len(self.labels)}, but "
                        f"shap returned shap values for {len(_shap_values)} classes! "
                        "Adjust the labels parameter accordingly!"
                    )

                if isinstance(_shap_values, list) and len(_shap_values) == 2:
                    # for binary classifier only keep positive class
                    _shap_values = _shap_values[1]
                elif isinstance(_shap_values, list) and len(_shap_values) > 2:
                    raise Exception(
                        f"len(self.label)={len(self.labels)}, but "
                        f"shap returned shap values for {len(_shap_values)} classes! "
                        "Adjust the labels parameter accordingly!"
                    )
            else:
                if (
                    isinstance(_shap_values, np.ndarray)
                    and len(_shap_values.shape) == 3
                ):
                    _shap_values = [
                        _shap_values[:, :, i] for i in range(_shap_values.shape[2])
                    ]
                assert len(_shap_values) == len(self.labels), (
                    f"len(self.label)={len(self.labels)}, but "
                    f"shap returned shap values for {len(_shap_values)} classes! "
                    "Adjust the labels parameter accordingly!"
                )
            if self.model_output == "probability":
                pass
                # for shap_values in _shap_values:
                #     assert np.all(shap_values >= -1.0) , \
                #         (f"model_output=='probability but some shap values are < 1.0!"
                #          "Try setting model_output='logodds'.")
                # for shap_values in _shap_values:
                #     assert np.all(shap_values <= 1.0) , \
                #         (f"model_output=='probability but some shap values are > 1.0!"
                #          "Try setting model_output='logodds'.")
            if len(self.labels) > 2:
                self._shap_values_df = [
                    pd.DataFrame(sv, columns=self.columns) for sv in _shap_values
                ]
                self._shap_values_df = [
                    merge_categorical_shap_values(
                        df, self.onehot_dict, self.merged_cols
                    ).astype(self.precision)
                    for df in self._shap_values_df
                ]
            else:
                self._shap_values_df = merge_categorical_shap_values(
                    pd.DataFrame(_shap_values, columns=self.columns),
                    self.onehot_dict,
                    self.merged_cols,
                ).astype(self.precision)

        if len(self.labels) > 2:
            if isinstance(self._shap_values_df, list):
                return self._shap_values_df[pos_label]
            else:
                return self._shap_values_df
        else:
            if pos_label == 1:
                return self._shap_values_df
            elif pos_label == 0:
                return self._shap_values_df.multiply(-1)
            else:
                raise ValueError(f"pos_label={pos_label}, but should be either 1 or 0!")

    def set_shap_values(self, base_value: List[float], shap_values: List):
        """Set shap values manually. This is useful if you already have
        shap values calculated, and do not want to calculate them again inside
        the explainer instance. Especially for large models and large datasets
        you may want to calculate shap values on specialized hardware, and then
        add them to the explainer manually.

        Args:
            base_value (list[float]): list of shap intercept generated by e.g.
                base_value = shap.TreeExplainer(model).shap_values(X_test).expected_value.
                Should be a list with a float for each class. For binary classification
                and some models shap only provides the base value for the positive class,
                in which case you need to provide [1-base_value, base_value] or [-base_value, base_value]
                depending on whether the shap values are for probabilities or logodds.
            shap_values (list[np.ndarray]): Generated by e.g.
                shap_values = shap.TreeExplainer(model).shap_values(X_test)
                For binary classification
                and some models shap only provides the shap values for the positive class,
                in which case you need to provide [1-shap_values, shap_values] or [-shap_values, shap_values]
                depending on whether the shap values are for probabilities or logodds.
        """
        if isinstance(base_value, np.ndarray) and base_value.shape == (
            len(self.labels),
        ):
            base_value = list(base_value)
        if not isinstance(base_value, list):
            raise ValueError(
                "base_value should be a list of floats with an expected value for each class"
            )
        if not len(base_value) == len(self.labels):
            raise ValueError(
                "base value should be a list with an expected value "
                f"for each class, so should be length {len(self.labels)}"
            )
        self._shap_base_value = base_value

        self._shap_values_df = []
        if not isinstance(shap_values, list):
            raise ValueError(
                "shap_values should be a list of np.ndarray with shap values for each class"
            )
        if len(shap_values) != len(self.labels):
            raise ValueError(
                "shap_values be a list with a np.ndarray of shap values "
                f"for each class, so should be length {len(self.labels)}"
            )
        for sv in shap_values:
            if not isinstance(sv, np.ndarray):
                raise ValueError("each element of shap_values should be an np.ndarray!")
            if sv.shape[0] != len(self.X):
                raise ValueError(f"Expected shap values to have {len(self.X)} rows!")
            if sv.shape[1] != len(self.original_cols):
                raise ValueError(
                    f"Expected shap values to have {len(self.original_columns)} columns!"
                )
            self._shap_values_df.append(
                merge_categorical_shap_values(
                    pd.DataFrame(sv, columns=self.columns),
                    self.onehot_dict,
                    self.merged_cols,
                ).astype(self.precision)
            )
        if len(self.labels) == 2:
            self._shap_values_df = self._shap_values_df[1]

    @insert_pos_label
    def get_shap_row(self, index=None, X_row=None, pos_label=None):
        def X_row_to_shap_row(X_row):
            if self.shap == "skorch":
                import torch

                X_row = torch.tensor(X_row.values.astype("float32"))
            with self.get_lock():
                shap_kwargs = (
                    dict(self.shap_kwargs, silent=True)
                    if self.shap == "kernel"
                    else self.shap_kwargs
                )
                sv = self.shap_explainer.shap_values(X_row, **shap_kwargs)
            if isinstance(sv, np.ndarray) and len(sv.shape) > 2:
                shap_row = pd.DataFrame(sv[:, :, pos_label], columns=self.columns)
            elif isinstance(sv, list) and len(sv) > 1:
                shap_row = pd.DataFrame(sv[pos_label], columns=self.columns)
            elif (
                len(self.labels) == 2
                and isinstance(sv, np.ndarray)
                and len(sv.shape) == 2
            ):
                if pos_label == 1:
                    shap_row = pd.DataFrame(sv, columns=self.columns)
                elif pos_label == 0:
                    shap_row = pd.DataFrame(-sv, columns=self.columns)
                else:
                    raise ValueError(
                        "binary classifier only except pos_label in {0, 1}!"
                    )
            else:
                raise ValueError(
                    "Shap values returned are neither a list nor 2d array for positive class!"
                )
            shap_row = merge_categorical_shap_values(
                shap_row, self.onehot_dict, self.merged_cols
            )
            return shap_row

        if index is not None:
            if index in self.idxs:
                shap_row = self.get_shap_values_df(pos_label=pos_label).iloc[
                    [self.idxs.get_loc(index)]
                ]
            elif isinstance(index, int) and index >= 0 and index < len(self):
                shap_row = self.get_shap_values_df(pos_label=pos_label).iloc[[index]]
            elif self._get_X_row_func is not None and self.index_exists(index):
                return X_row_to_shap_row(self._get_X_row_func(index))
            else:
                raise IndexNotFoundError(index=index)
        elif X_row is not None:
            return X_row_to_shap_row(X_row)
        else:
            raise ValueError("you should either pas index or X_row!")
        return shap_row

    @insert_pos_label
    def shap_interaction_values(self, pos_label=None):
        """SHAP interaction values"""
        if not hasattr(self, "_shap_interaction_values"):
            _ = self.get_shap_values_df()  # make sure shap values have been calculated
            print(
                "Calculating shap interaction values... (this may take a while)",
                flush=True,
            )
            if self.shap == "tree":
                print(
                    "Reminder: TreeShap computational complexity is O(TLD^2), "
                    "where T is the number of trees, L is the maximum number of"
                    " leaves in any tree and D the maximal depth of any tree. So "
                    "reducing these will speed up the calculation.",
                    flush=True,
                )
            self._shap_interaction_values = self.shap_explainer.shap_interaction_values(
                self.X
            )
            if len(self.labels) == 2:
                if (
                    isinstance(self._shap_interaction_values, np.ndarray)
                    and len(self._shap_interaction_values.shape) == 4
                    and self._shap_interaction_values.shape[3] == 2
                ):
                    # for binary classifier only keep positive class:
                    self._shap_interaction_values = [
                        self._shap_interaction_values[:, :, :, 1]
                    ]
                elif (
                    isinstance(self._shap_interaction_values, np.ndarray)
                    and len(self._shap_interaction_values.shape) == 3
                ):
                    # for binary classifier only keep positive class:
                    self._shap_interaction_values = [self._shap_interaction_values]
                elif (
                    isinstance(self._shap_interaction_values, list)
                    and len(self._shap_interaction_values) == 2
                ):
                    # for binary classifier only keep positive class
                    self._shap_interaction_values = [self._shap_interaction_values[1]]
                else:
                    raise Exception(
                        f"len(self.label)={len(self.labels)}, but "
                        f"shap returned shap interaction values for "
                        f"{len(self._shap_interaction_values)} classes! "
                        "Adjust the labels parameter accordingly!"
                    )
            else:
                if (
                    isinstance(self._shap_interaction_values, np.ndarray)
                    and len(self._shap_interaction_values.shape) == 4
                    and self._shap_interaction_values.shape[3] > 2
                ):
                    self._shap_interaction_values = [
                        self._shap_interaction_values[:, :, :, i]
                        for i in range(self._shap_interaction_values.shape[3])
                    ]
                assert len(self._shap_interaction_values) == len(self.labels), (
                    f"len(self.label)={len(self.labels)}, but "
                    f"shap returned shap values for {len(self._shap_interaction_values)} classes! "
                    "Adjust the labels parameter accordingly!"
                )

            self._shap_interaction_values = [
                merge_categorical_shap_interaction_values(
                    siv, self.columns, self.merged_cols, self.onehot_dict
                ).astype(self.precision)
                for siv in self._shap_interaction_values
            ]
            if len(self._shap_interaction_values) == 1:
                self._shap_interaction_values = self._shap_interaction_values[0]

        if len(self.labels) > 2:
            if isinstance(self._shap_interaction_values, list):
                return self._shap_interaction_values[pos_label]
            else:
                return self._shap_interaction_values
        else:
            if pos_label == 1:
                return self._shap_interaction_values
            elif pos_label == 0:
                return self._shap_interaction_values * -1
            else:
                raise ValueError(f"pos_label={pos_label}, but should be either 1 or 0!")

    def set_shap_interaction_values(self, shap_interaction_values: List[np.ndarray]):
        """Manually set shap interaction values in case you have already pre-computed
        these elsewhere and do not want to re-calculate them again inside the
        explainer instance.

        Args:
            shap_interaction_values (np.ndarray): shap interactions values of shape (n, m, m)

        """
        self._shap_interaction_values = []
        if not isinstance(shap_interaction_values, list):
            raise ValueError(
                "shap_interaction_values should be a list of np.ndarray with shap interaction values for each class"
            )
        if len(shap_interaction_values) != len(self.labels):
            raise ValueError(
                "shap_interaction_values should be a list with a np.ndarray of shap interaction values "
                f"for each class, so should be length {len(self.labels)}"
            )
        for siv in shap_interaction_values:
            if not isinstance(siv, np.ndarray):
                raise ValueError("each element of shap_values should be an np.ndarray!")
            if siv.shape != (
                len(self.X),
                len(self.original_cols),
                len(self.original_cols),
            ):
                raise ValueError(
                    f"Expected shap interaction values to have shape of "
                    f"({len(self.X)}, {len(self.original_cols)}, {len(self.original_cols)})"
                )
            self._shap_interaction_values.append(
                merge_categorical_shap_interaction_values(
                    siv, self.columns, self.merged_cols, self.onehot_dict
                ).astype(self.precision)
            )
        if len(self.labels) == 2:
            self._shap_interaction_values = self._shap_interaction_values[1]

    @insert_pos_label
    def mean_abs_shap_df(self, pos_label=None):
        """mean absolute SHAP values"""
        if not hasattr(self, "_mean_abs_shap_df"):
            _ = self.get_shap_values_df()
            self._mean_abs_shap_df = [
                self.get_shap_values_df(pos_label)[self.merged_cols]
                .abs()
                .mean()
                .sort_values(ascending=False)
                .to_frame()
                .rename_axis(index="Feature")
                .reset_index()
                .rename(columns={0: "MEAN_ABS_SHAP"})
                for pos_label in self.labels
            ]
        return self._mean_abs_shap_df[pos_label]

    @insert_pos_label
    def keep_shap_pos_label_only(self, pos_label=None):
        """drops the shap values and shap_interaction values for all labels
        except pos_label in order to save on memory usage for multi class classifiers"""
        assert len(self.labels) > 2, (
            "It is not necessary to drop shap values for binary classifiers! "
            "ClassifierExplainer only store a single label anyway and return "
            "negative shap_values for the negative class..."
        )
        if hasattr(self, "_shap_values_df"):
            self._shap_values_df = self.get_shap_values_df(pos_label)
        if hasattr(self, "_shap_interaction_values"):
            self._shap_interaction_values = self.shap_interaction_values(pos_label)

    @insert_pos_label
    def cutoff_from_percentile(self, percentile, pos_label=None):
        """The cutoff equivalent to the percentile given

        For example if you want the cutoff that splits the highest 20%
        pred_proba from the lowest 80%, you would set percentile=0.8
        and get the correct cutoff.

        Args:
          percentile(float):  percentile to convert to cutoff
          pos_label: positive class (Default value = None)

        Returns:
          cutoff

        """
        return (
            pd.Series(self.pred_probas(pos_label))
            .nlargest(int((1 - percentile) * len(self)))
            .min()
        )

    @insert_pos_label
    def percentile_from_cutoff(self, cutoff, pos_label=None):
        """The percentile equivalent to the cutoff given

        For example if set the cutoff at 0.8, then what percentage
        of pred_proba is above this cutoff?

        Args:
          cutoff (float):  cutoff to convert to percentile
          pos_label: positive class (Default value = None)

        Returns:
          percentile

        """
        if cutoff is None:
            return None
        return 1 - (self.pred_probas(pos_label) < cutoff).mean()

    @insert_pos_label
    def metrics(
        self,
        cutoff: float = 0.5,
        show_metrics: List[Union[str, Callable]] = None,
        pos_label: int = None,
    ):
        """returns a dict with useful metrics for your classifier:

        accuracy, precision, recall, f1, roc auc, pr auc, log loss

        Args:
          cutoff(float): cutoff used to calculate metrics (Default value = 0.5)
          show_metrics (List): list of metrics to display in order. Defaults
                to None, displaying all metrics.
          pos_label: positive class (Default value = None)

        Returns:
          dict

        """
        if self.y_missing:
            raise ValueError(
                "No y was passed to explainer, so cannot calculate metrics!"
            )

        def get_metrics(cutoff, pos_label):
            y_true = self.y_binary(pos_label)
            y_pred = np.where(self.pred_probas(pos_label) > cutoff, 1, 0)

            metrics_dict = {
                "accuracy": accuracy_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred),
                "f1": f1_score(y_true, y_pred),
                "roc_auc_score": roc_auc_score(y_true, self.pred_probas(pos_label)),
                "pr_auc_score": average_precision_score(
                    y_true, self.pred_probas(pos_label)
                ),
                "log_loss": log_loss(y_true, self.pred_probas(pos_label)),
            }
            return metrics_dict

        def get_cv_metrics(n_splits):
            cv_metrics = {}
            for label in range(len(self.labels)):
                cv_metrics[label] = dict()
                for cut in np.linspace(1, 99, 99, dtype=int):
                    cv_metrics[label][cut] = {
                        "accuracy": [],
                        "precision": [],
                        "recall": [],
                        "f1": [],
                        "roc_auc_score": [],
                        "pr_auc_score": [],
                        "log_loss": [],
                    }
            for train_index, test_index in KFold(n_splits=n_splits, shuffle=True).split(
                self.X
            ):
                X_train, X_test = self.X.iloc[train_index], self.X.iloc[test_index]
                y_train, y_test = self.y.iloc[train_index], self.y.iloc[test_index]
                preds = clone(self.model).fit(X_train, y_train).predict_proba(X_test)
                for label in range(len(self.labels)):
                    for cut in np.linspace(1, 99, 99, dtype=int):
                        y_true = np.where(y_test == label, 1, 0)
                        y_pred = np.where(preds[:, label] > 0.01 * cut, 1, 0)
                        cv_metrics[label][cut]["accuracy"].append(
                            accuracy_score(y_true, y_pred)
                        )
                        cv_metrics[label][cut]["precision"].append(
                            precision_score(y_true, y_pred, zero_division=0)
                        )
                        cv_metrics[label][cut]["recall"].append(
                            recall_score(y_true, y_pred)
                        )
                        cv_metrics[label][cut]["f1"].append(f1_score(y_true, y_pred))
                        cv_metrics[label][cut]["roc_auc_score"].append(
                            roc_auc_score(y_true, preds[:, label])
                        )
                        cv_metrics[label][cut]["pr_auc_score"].append(
                            average_precision_score(y_true, preds[:, label])
                        )
                        cv_metrics[label][cut]["log_loss"].append(
                            log_loss(y_true, preds[:, label])
                        )
            for label in range(len(self.labels)):
                for cut in np.linspace(1, 99, 99, dtype=int):
                    cv_metrics[label][cut] = {
                        k: np.mean(v) for k, v in cv_metrics[label][cut].items()
                    }
            return cv_metrics

        if not hasattr(self, "_metrics"):
            _ = self.pred_probas()
            print("Calculating metrics...", flush=True)
            if self.cv is None:
                self._metrics = dict()
                for label in range(len(self.labels)):
                    self._metrics[label] = dict()
                    for cut in np.linspace(1, 99, 99, dtype=int):
                        self._metrics[label][cut] = get_metrics(0.01 * cut, label)
            else:
                self._metrics = get_cv_metrics(self.cv)

        if int(cutoff * 100) in self._metrics[pos_label]:
            metrics_dict = self._metrics[pos_label][int(cutoff * 100)]
        else:
            metrics_dict = get_metrics(cutoff, pos_label)

        if not show_metrics:
            return metrics_dict

        show_metrics_dict = {}
        for m in show_metrics:
            if callable(m):
                if self.cv is not None:
                    raise ValueError(
                        "custom metrics do not work with permutation_cv for now!"
                    )
                metric_args = inspect.signature(m).parameters.keys()
                metric_kwargs = {}
                if "pos_label" in metric_args:
                    y_true = self.y
                    y_pred = self.pred_probas_raw
                    metric_kwargs["pos_label"] = pos_label
                else:
                    y_true = self.y_binary(pos_label)
                    y_pred = self.pred_probas(pos_label)

                if "cutoff" in metric_args:
                    metric_kwargs["cutoff"] = cutoff
                else:
                    y_pred = np.where(y_pred > cutoff, 1, 0)
                try:
                    show_metrics_dict[m.__name__] = m(y_true, y_pred, **metric_kwargs)
                except:
                    raise Exception(
                        f"Failed to calculate metric {m.__name__}! "
                        "Make sure it takes arguments y_true and y_pred, and "
                        "optionally cutoff and pos_label!"
                    )
            elif m in metrics_dict:
                show_metrics_dict[m] = metrics_dict[m]
        return show_metrics_dict

    @insert_pos_label
    def metrics_descriptions(self, cutoff=0.5, round=3, pos_label=None):
        """Returns a metrics dict with the value replaced with a
        description/interpretation of the value

        Args:
            cutoff (float, optional): Cutoff for calculating the metrics. Defaults to 0.5.
            round (int, optional): Round to apply to floats. Defaults to 3.
            pos_label (None, optional): positive label. Defaults to None.

        Returns:
            dict
        """
        metrics_dict = self.metrics(cutoff=cutoff, pos_label=pos_label)
        metrics_descriptions_dict = {}
        for k, v in metrics_dict.items():
            if k == "accuracy":
                metrics_descriptions_dict[
                    k
                ] = f"{100*v:.{round}f}% of predicted labels was predicted correctly."
            if k == "precision":
                metrics_descriptions_dict[
                    k
                ] = f"{100*v:.{round}f}% of predicted positive labels was predicted correctly."
            if k == "recall":
                metrics_descriptions_dict[
                    k
                ] = f"{100*v:.{round}f}% of positive labels was predicted correctly."
            if k == "f1":
                metrics_descriptions_dict[
                    k
                ] = f"The weighted average of precision and recall is {v:.{round}f}"
            if k == "roc_auc_score":
                metrics_descriptions_dict[
                    k
                ] = f"The probability that a random positive label has a higher score than a random negative label is {100*v:.2f}%"
            if k == "pr_auc_score":
                metrics_descriptions_dict[
                    k
                ] = f"The average precision score calculated for each recall threshold is {v:.{round}f}. This ignores true negatives."
            if k == "log_loss":
                metrics_descriptions_dict[
                    k
                ] = f"A measure of how far the predicted label is from the true label on average in log space {v:.{round}f}"
        return metrics_descriptions_dict

    @insert_pos_label
    def random_index(
        self,
        y_values=None,
        return_str=False,
        pred_proba_min=None,
        pred_proba_max=None,
        pred_percentile_min=None,
        pred_percentile_max=None,
        pos_label=None,
    ):
        """random index satisfying various constraint

        Args:
          y_values: list of labels to include (Default value = None)
          return_str: return str from self.idxs (Default value = False)
          pred_proba_min: minimum pred_proba (Default value = None)
          pred_proba_max: maximum pred_proba (Default value = None)
          pred_percentile_min: minimum pred_proba percentile (Default value = None)
          pred_percentile_max: maximum pred_proba percentile (Default value = None)
          pos_label: positive class (Default value = None)

        Returns:
          index

        """
        # if pos_label is None: pos_label = self.pos_label
        if (
            y_values is None
            and pred_proba_min is None
            and pred_proba_max is None
            and pred_percentile_min is None
            and pred_percentile_max is None
        ):
            potential_idxs = self.idxs.values
        else:
            pred_probas = self.pred_probas(pos_label)
            pred_percentiles = self.pred_percentiles(pos_label)
            if pred_proba_min is None:
                pred_proba_min = pred_probas.min()
            if pred_proba_max is None:
                pred_proba_max = pred_probas.max()
            if pred_percentile_min is None:
                pred_percentile_min = 0.0
            if pred_percentile_max is None:
                pred_percentile_max = 1.0

            if not self.y_missing:
                if y_values is None:
                    y_values = self.y.unique().astype(str).tolist()
                if not isinstance(y_values, list):
                    y_values = [y_values]
                y_values = [
                    y if isinstance(y, int) else self.labels.index(str(y))
                    for y in y_values
                ]

                potential_idxs = self.idxs[
                    (self.y.isin(y_values))
                    & (pred_probas >= pred_proba_min)
                    & (pred_probas <= pred_proba_max)
                    & (pred_percentiles > pred_percentile_min)
                    & (pred_percentiles <= pred_percentile_max)
                ].values

            else:
                potential_idxs = self.idxs[
                    (pred_probas >= pred_proba_min)
                    & (pred_probas <= pred_proba_max)
                    & (pred_percentiles > pred_percentile_min)
                    & (pred_percentiles <= pred_percentile_max)
                ].values

        if len(potential_idxs) > 0:
            idx = np.random.choice(potential_idxs)
        else:
            return None
        if return_str:
            return idx
        return self.idxs.get_loc(idx)

    def prediction_result_df(
        self, index=None, X_row=None, add_star=True, logodds=False, round=3
    ):
        """returns a table with the predicted probability for each label for index

        Args:
            index ({int, str}): index
            add_star(bool): add a star to the observed label
            round (int): rounding to apply to pred_proba float

        Returns:
            pd.DataFrame
        """
        if index is None and X_row is None:
            raise ValueError("You need to either pass an index or X_row!")
        if index is not None:
            X_row = self.get_X_row(index)
        if X_row is not None:
            if matching_cols(X_row.columns, self.merged_cols):
                X_row = X_cats_to_X(X_row, self.onehot_dict, self.X.columns)
            if self.shap == "skorch":
                X_row = X_row.values.astype("float32")
            pred_probas = self.model.predict_proba(X_row)[0, :].squeeze()

        preds_df = pd.DataFrame(dict(label=self.labels, probability=pred_probas))
        if logodds and all(preds_df.probability < 1 - np.finfo(np.float64).eps):
            preds_df.loc[:, "logodds"] = preds_df.probability.apply(
                lambda p: np.log(p / (1 - p))
            )
        if index is not None:
            try:
                y_true = self.pos_label_index(self.get_y(index))
                preds_df.iloc[y_true, 0] = f"{preds_df.iloc[y_true, 0]}*"
            except Exception as e:
                print(e)

        return preds_df.round(round)

    @insert_pos_label
    def get_precision_df(
        self, bin_size=None, quantiles=None, multiclass=False, round=3, pos_label=None
    ):
        """dataframe with predicted probabilities and precision

        Args:
          bin_size(float, optional, optional): group predictions in bins of size bin_size, defaults to 0.1
          quantiles(int, optional, optional): group predictions in evenly sized quantiles of size quantiles, defaults to None
          multiclass(bool, optional, optional): whether to calculate precision for every class (Default value = False)
          round:  (Default value = 3)
          pos_label:  (Default value = None)

        Returns:
          pd.DataFrame: precision_df

        """
        if self.y_missing:
            raise ValueError(
                "No y was passed to explainer, so cannot calculate precision_df!"
            )
        assert self.pred_probas is not None

        if bin_size is None and quantiles is None:
            bin_size = 0.1  # defaults to bin_size=0.1
        if multiclass:
            return get_precision_df(
                self.pred_probas_raw,
                self.y,
                bin_size,
                quantiles,
                round=round,
                pos_label=pos_label,
            )
        else:
            return get_precision_df(
                self.pred_probas(pos_label),
                self.y_binary(pos_label),
                bin_size,
                quantiles,
                round=round,
            )

    @insert_pos_label
    def get_liftcurve_df(self, pos_label=None):
        """returns a pd.DataFrame with data needed to build a lift curve

        Args:
          pos_label:  (Default value = None)

        Returns:

        """
        if not hasattr(self, "_liftcurve_dfs"):
            print("Calculating liftcurve_dfs...", flush=True)
            self._liftcurve_dfs = [
                get_liftcurve_df(self.pred_probas(label), self.y, label)
                for label in range(len(self.labels))
            ]
        return self._liftcurve_dfs[pos_label]

    @insert_pos_label
    def get_classification_df(self, cutoff=0.5, pos_label=None):
        """Returns a dataframe with number of observations in each class above
        and below the cutoff.

        Args:
            cutoff (float, optional): Cutoff to split on. Defaults to 0.5.
            pos_label (int, optional): Pos label to generate dataframe for.
                Defaults to self.pos_label.

        Returns:
            pd.DataFrame
        """

        def get_clas_df(cutoff, pos_label):
            clas_df = pd.DataFrame(index=pd.RangeIndex(0, len(self.labels)))
            clas_df["below"] = self.y[
                self.pred_probas(pos_label) < cutoff
            ].value_counts()
            clas_df["above"] = self.y[
                self.pred_probas(pos_label) >= cutoff
            ].value_counts()
            clas_df = clas_df.fillna(0)
            clas_df["total"] = clas_df.sum(axis=1)
            clas_df.index = self.labels
            return clas_df

        if not hasattr(self, "_classification_dfs"):
            _ = self.pred_probas()
            print("Calculating classification_dfs...", flush=True)
            self._classification_dfs = dict()
            for label in range(len(self.labels)):
                self._classification_dfs[label] = dict()
                for cut in np.linspace(0.01, 0.99, 99):
                    self._classification_dfs[label][np.round(cut, 2)] = get_clas_df(
                        cut, label
                    )
        if cutoff in self._classification_dfs[pos_label]:
            return self._classification_dfs[pos_label][cutoff]
        else:
            return get_clas_df(cutoff, pos_label)

    @insert_pos_label
    def roc_auc_curve(self, pos_label=None):
        """Returns a dict with output from sklearn.metrics.roc_curve() for pos_label:
        fpr, tpr, thresholds, score"""

        if not hasattr(self, "_roc_auc_curves"):
            print("Calculating roc auc curves...", flush=True)
            self._roc_auc_curves = []
            for i in range(len(self.labels)):
                fpr, tpr, thresholds = roc_curve(self.y_binary(i), self.pred_probas(i))
                score = roc_auc_score(self.y_binary(i), self.pred_probas(i))
                self._roc_auc_curves.append(
                    dict(fpr=fpr, tpr=tpr, thresholds=thresholds, score=score)
                )
        return self._roc_auc_curves[pos_label]

    @insert_pos_label
    def pr_auc_curve(self, pos_label=None):
        """Returns a dict with output from sklearn.metrics.precision_recall_curve() for pos_label:
        fpr, tpr, thresholds, score"""

        if not hasattr(self, "_pr_auc_curves"):
            print("Calculating pr auc curves...", flush=True)
            self._pr_auc_curves = []
            for i in range(len(self.labels)):
                precision, recall, thresholds = precision_recall_curve(
                    self.y_binary(i), self.pred_probas(i)
                )
                score = average_precision_score(self.y_binary(i), self.pred_probas(i))
                self._pr_auc_curves.append(
                    dict(
                        precision=precision,
                        recall=recall,
                        thresholds=thresholds,
                        score=score,
                    )
                )
        return self._pr_auc_curves[pos_label]

    @insert_pos_label
    def confusion_matrix(self, cutoff=0.5, binary=True, pos_label=None):
        def get_binary_cm(y, pred_probas, cutoff, pos_label):
            return confusion_matrix(
                np.where(y == pos_label, 1, 0),
                np.where(pred_probas[:, pos_label] >= cutoff, 1, 0),
            )

        if not hasattr(self, "_confusion_matrices"):
            print("Calculating confusion matrices...", flush=True)
            self._confusion_matrices = dict()
            self._confusion_matrices["binary"] = dict()
            for label in range(len(self.labels)):
                self._confusion_matrices["binary"][label] = dict()
                for cut in np.linspace(0.01, 0.99, 99):
                    self._confusion_matrices["binary"][label][
                        np.round(cut, 2)
                    ] = get_binary_cm(self.y, self.pred_probas_raw, cut, label)
            self._confusion_matrices["multi"] = confusion_matrix(
                self.y, self.pred_probas_raw.argmax(axis=1)
            )
        if binary:
            if cutoff in self._confusion_matrices["binary"][pos_label]:
                return self._confusion_matrices["binary"][pos_label][cutoff]
            else:
                return get_binary_cm(self.y, self.pred_probas_raw, cutoff, pos_label)
        else:
            return self._confusion_matrices["multi"]

    @insert_pos_label
    def plot_precision(
        self,
        bin_size=None,
        quantiles=None,
        cutoff=None,
        multiclass=False,
        pos_label=None,
    ):
        """plot precision vs predicted probability

        plots predicted probability on the x-axis and observed precision (fraction of actual positive
        cases) on the y-axis.

        Should pass either bin_size fraction of number of quantiles, but not both.

        Args:
          bin_size(float, optional):  size of the bins on x-axis (e.g. 0.05 for 20 bins)
          quantiles(int, optional): number of equal sized quantiles to split
                    the predictions by e.g. 20, optional)
          cutoff: cutoff of model to include in the plot (Default value = None)
          multiclass: whether to display all classes or only positive class,
                    defaults to False
          pos_label: positive label to display, defaults to self.pos_label

        Returns:
          Plotly fig

        """
        if bin_size is None and quantiles is None:
            bin_size = 0.1  # defaults to bin_size=0.1
        precision_df = self.get_precision_df(
            bin_size=bin_size,
            quantiles=quantiles,
            multiclass=multiclass,
            pos_label=pos_label,
        )
        return plotly_precision_plot(
            precision_df, cutoff=cutoff, labels=self.labels, pos_label=pos_label
        )

    @insert_pos_label
    def plot_cumulative_precision(self, percentile=None, pos_label=None):
        """plot cumulative precision

        returns a cumulative precision plot, which is a slightly different
        representation of a lift curve.

        Args:
          pos_label: positive label to display, defaults to self.pos_label

        Returns:
          plotly fig

        """
        return plotly_cumulative_precision_plot(
            self.get_liftcurve_df(pos_label=pos_label),
            labels=self.labels,
            percentile=percentile,
            pos_label=pos_label,
        )

    @insert_pos_label
    def plot_confusion_matrix(
        self,
        cutoff=0.5,
        percentage=False,
        normalize="all",
        binary=False,
        pos_label=None,
    ):
        """plot of a confusion matrix.

        Args:
          cutoff(float, optional, optional): cutoff of positive class to
                    calculate confusion matrix for, defaults to 0.5
          percentage(bool, optional, optional): display percentages instead
                    of counts , defaults to False
          normalize (str[‘observed’, ‘pred’, ‘all’]): normalizes confusion matrix over
            the observed (rows), predicted (columns) conditions or all the population.
            Defaults to all.
          binary(bool, optional, optional): if multiclass display one-vs-rest
                    instead, defaults to False
          pos_label: positive label to display, defaults to self.pos_label


        Returns:
          plotly fig

        """
        if self.y_missing:
            raise ValueError(
                "No y was passed to explainer, so cannot plot confusion matrix!"
            )
        pos_label_str = self.labels[pos_label]
        if binary:
            if len(self.labels) == 2:

                def order_binary_labels(labels, pos_label):
                    pos_index = self.pos_label_index(pos_label)
                    return [labels[1 - pos_index], labels[pos_index]]

                labels = order_binary_labels(self.labels, pos_label_str)
            else:
                labels = ["Not " + pos_label_str, pos_label_str]

            return plotly_confusion_matrix(
                self.confusion_matrix(cutoff, binary, pos_label),
                percentage=percentage,
                labels=labels,
                normalize=normalize,
            )
        else:
            return plotly_confusion_matrix(
                self.confusion_matrix(cutoff, binary, pos_label),
                percentage=percentage,
                normalize=normalize,
                labels=self.labels,
            )

    @insert_pos_label
    def plot_lift_curve(
        self, cutoff=None, percentage=False, add_wizard=True, round=2, pos_label=None
    ):
        """plot of a lift curve.

        Args:
          cutoff(float, optional): cutoff of positive class to calculate lift
                    (Default value = None)
          percentage(bool, optional): display percentages instead of counts,
                    defaults to False
          add_wizard (bool, optional): Add a line indicating how a perfect model
                    would perform ("the wizard"). Defaults to True.
          round: number of digits to round to (Default value = 2)
          pos_label: positive label to display, defaults to self.pos_label

        Returns:
          plotly fig

        """
        return plotly_lift_curve(
            self.get_liftcurve_df(pos_label),
            cutoff=cutoff,
            percentage=percentage,
            add_wizard=add_wizard,
            round=round,
        )

    @insert_pos_label
    def plot_classification(self, cutoff=0.5, percentage=True, pos_label=None):
        """plot showing a barchart of the classification result for cutoff

        Args:
          cutoff(float, optional): cutoff of positive class to calculate lift
                    (Default value = 0.5)
          percentage(bool, optional): display percentages instead of counts,
                    defaults to True
          pos_label: positive label to display, defaults to self.pos_label

        Returns:
          plotly fig

        """
        return plotly_classification_plot(
            self.get_classification_df(cutoff=cutoff, pos_label=pos_label),
            percentage=percentage,
        )

    @insert_pos_label
    def plot_roc_auc(self, cutoff=0.5, pos_label=None):
        """plots ROC_AUC curve.

        The TPR and FPR of a particular cutoff is displayed in crosshairs.

        Args:
          cutoff: cutoff value to be included in plot (Default value = 0.5)
          pos_label:  (Default value = None)

        Returns:

        """
        if self.y_missing:
            raise ValueError("No y was passed to explainer, so cannot plot roc auc!")
        roc_dict = self.roc_auc_curve(pos_label)
        return plotly_roc_auc_curve(
            roc_dict["fpr"],
            roc_dict["tpr"],
            roc_dict["thresholds"],
            roc_dict["score"],
            cutoff=cutoff,
        )

    @insert_pos_label
    def plot_pr_auc(self, cutoff=0.5, pos_label=None):
        """plots PR_AUC curve.

        the precision and recall of particular cutoff is displayed in crosshairs.

        Args:
          cutoff: cutoff value to be included in plot (Default value = 0.5)
          pos_label:  (Default value = None)

        Returns:

        """
        if self.y_missing:
            raise ValueError("No y was passed to explainer, so cannot plot PR AUC!")
        pr_dict = self.pr_auc_curve(pos_label)
        return plotly_pr_auc_curve(
            pr_dict["precision"],
            pr_dict["recall"],
            pr_dict["thresholds"],
            pr_dict["score"],
            cutoff=cutoff,
        )

    def plot_prediction_result(self, index=None, X_row=None, showlegend=True):
        """Returns a piechart with the predicted probabilities distribution

        Args:
            index ({int, str}): Index for which to display prediction
            X_row (pd.DataFrame): single row of an input dataframe, e.g.
                explainer.X.iloc[[0]]
            showlegend (bool, optional): Display legend. Defaults to False.

        Returns:
            plotly.fig
        """
        preds_df = self.prediction_result_df(index, X_row)
        return plotly_prediction_piechart(preds_df, showlegend=showlegend)

    def calculate_properties(self, include_interactions=True):
        """calculate all lazily calculated properties of explainer

        Args:
          include_interactions:  (Default value = True)

        Returns:
            None

        """
        _ = self.pred_probas()
        if not self.y_missing:
            _ = self.y_binary()
            _ = self.metrics(), self.get_classification_df()
            _ = self.roc_auc_curve(), self.pr_auc_curve()
        super().calculate_properties(include_interactions=include_interactions)


class RegressionExplainer(BaseExplainer):
    """ """

    def __init__(
        self,
        model,
        X: pd.DataFrame,
        y: pd.Series = None,
        permutation_metric: Callable = r2_score,
        shap: str = "guess",
        X_background: pd.DataFrame = None,
        model_output: str = "raw",
        cats: Union[List, Dict] = None,
        cats_notencoded: Dict = None,
        idxs: pd.Index = None,
        index_name: str = None,
        target: str = None,
        descriptions: Dict = None,
        n_jobs: int = None,
        permutation_cv: int = None,
        cv: int = None,
        na_fill: float = -999,
        precision: str = "float64",
        shap_kwargs: Dict = None,
        units: str = "",
    ):
        """Explainer for regression models.

        In addition to BaseExplainer defines a number of plots specific to
        regression problems such as a predicted vs actual and residual plots.

        Combared to BaseExplainerBunch defines two additional parameters.

        Args:
            model: a model with a scikit-learn compatible .fit and .predict methods
            X (pd.DataFrame): a pd.DataFrame with your model features
            y (pd.Series): Dependent variable of your model, defaults to None
            permutation_metric (function or str): is a scikit-learn compatible
                metric function (or string). Defaults to r2_score
            shap (str): type of shap_explainer to fit: 'tree', 'linear', 'kernel'.
                Defaults to 'guess'.
            X_background (pd.DataFrame): background X to be used by shap
                explainers that need a background dataset (e.g. shap.KernelExplainer
                or shap.TreeExplainer with boosting models and
                model_output='probability').
            model_output (str): model_output of shap values, either 'raw',
                'logodds' or 'probability'. Defaults to 'raw' for regression and
                'probability' for classification.
            cats ({dict, list}): dict of features that have been
                onehotencoded. e.g. cats={'Sex':['Sex_male', 'Sex_female']}.
                If all encoded columns are underscore-seperated (as above), can simply
                pass a list of prefixes: cats=['Sex']. Allows to
                group onehot encoded categorical variables together in
                various plots. Defaults to None.
            cats_notencoded (dict): value to display when all onehot encoded
                columns are equal to zero. Defaults to 'NOT_ENCODED' for each
                onehot col.
            idxs (pd.Series): list of row identifiers. Can be names, id's, etc.
                Defaults to X.index.
            index_name (str): identifier for row indexes. e.g. index_name='Passenger'.
                Defaults to X.index.name or idxs.name.
            target: name of the predicted target, e.g. "Survival",
                "Ticket price", etc. Defaults to y.name.
            n_jobs (int): for jobs that can be parallelized using joblib,
                how many processes to split the job in. For now only used
                for calculating permutation importances. Defaults to None.
            permutation_cv (int): Deprecated! Use parameter cv instead!
                (now also works for calculating metrics)
            cv (int): If not None then permutation importances and metrics
                will get calculated using cross validation across X. Use this
                when you are passing the training set to the explainer.
                Defaults to None.
            na_fill (int): The filler used for missing values, defaults to -999.
            precision: precision with which to store values. Defaults to "float64".
            shap_kwargs(dict): dictionary of keyword arguments to be passed to the shap explainer.
                most typically used to supress an additivity check e.g. `shap_kwargs=dict(check_additivity=False)`
            units(str): units to display for regression quantity
        """
        super().__init__(
            model,
            X,
            y,
            permutation_metric,
            shap,
            X_background,
            model_output,
            cats,
            cats_notencoded,
            idxs,
            index_name,
            target,
            descriptions,
            n_jobs,
            permutation_cv,
            cv,
            na_fill,
            precision,
            shap_kwargs,
        )

        self._params_dict = {**self._params_dict, **dict(units=units)}
        self.units = units
        self.is_regression = True

        if safe_isinstance(model, "RandomForestRegressor", "ExtraTreesRegressor"):
            print(
                "Changing class type to RandomForestRegressionExplainer...", flush=True
            )
            self.__class__ = RandomForestRegressionExplainer
        if safe_isinstance(model, "XGBRegressor"):
            print("Changing class type to XGBRegressionExplainer...", flush=True)
            self.__class__ = XGBRegressionExplainer

        _ = self.shap_explainer

    @property
    def residuals(self):
        """residuals: y-preds"""
        if not hasattr(self, "_residuals"):
            print("Calculating residuals...")
            self._residuals = (self.y - self.preds).astype(self.precision)
        return self._residuals

    @property
    def abs_residuals(self):
        """absolute residuals"""
        if not hasattr(self, "_abs_residuals"):
            print("Calculating absolute residuals...")
            self._abs_residuals = np.abs(self.residuals).astype(self.precision)
        return self._abs_residuals

    def random_index(
        self,
        y_min=None,
        y_max=None,
        pred_min=None,
        pred_max=None,
        residuals_min=None,
        residuals_max=None,
        abs_residuals_min=None,
        abs_residuals_max=None,
        return_str=False,
        **kwargs,
    ):
        """random index following to various exclusion criteria

        Args:
          y_min:  (Default value = None)
          y_max:  (Default value = None)
          pred_min:  (Default value = None)
          pred_max:  (Default value = None)
          residuals_min:  (Default value = None)
          residuals_max:  (Default value = None)
          abs_residuals_min:  (Default value = None)
          abs_residuals_max:  (Default value = None)
          return_str:  return the str index from self.idxs (Default value = False)
          **kwargs:

        Returns:
          a random index that fits the exclusion criteria

        """
        if self.y_missing:
            if pred_min is None:
                pred_min = self.preds.min()
            if pred_max is None:
                pred_max = self.preds.max()
            potential_idxs = self.idxs[
                (self.preds >= pred_min) & (self.preds <= pred_max)
            ].values
        else:
            if y_min is None:
                y_min = self.y.min()
            if y_max is None:
                y_max = self.y.max()
            if pred_min is None:
                pred_min = self.preds.min()
            if pred_max is None:
                pred_max = self.preds.max()
            if residuals_min is None:
                residuals_min = self.residuals.min()
            if residuals_max is None:
                residuals_max = self.residuals.max()
            if abs_residuals_min is None:
                abs_residuals_min = self.abs_residuals.min()
            if abs_residuals_max is None:
                abs_residuals_max = self.abs_residuals.max()

            potential_idxs = self.idxs[
                (self.y >= y_min)
                & (self.y <= y_max)
                & (self.preds >= pred_min)
                & (self.preds <= pred_max)
                & (self.residuals >= residuals_min)
                & (self.residuals <= residuals_max)
                & (self.abs_residuals >= abs_residuals_min)
                & (self.abs_residuals <= abs_residuals_max)
            ].values

        if len(potential_idxs) > 0:
            idx = np.random.choice(potential_idxs)
        else:
            return None
        if return_str:
            return idx
        return self.idxs.get_loc(idx)

    def prediction_result_df(self, index=None, X_row=None, round=3):
        """prediction result in dataframe format

        Args:
            index:  row index to be predicted
            round (int):  rounding applied to floats (defaults to 3)

        Returns:
            pd.DataFrame

        """
        if index is None and X_row is None:
            raise ValueError("You need to either pass an index or X_row!")
        if index is not None:
            X_row = self.get_X_row(index)
        if X_row is not None:
            if matching_cols(X_row.columns, self.merged_cols):
                X_row = X_cats_to_X(X_row, self.onehot_dict, self.X.columns)
        if self.shap == "skorch":
            X_row = X_row.values.astype("float32")
        pred = self.model.predict(X_row).item()
        preds_df = pd.DataFrame(columns=["", self.target])
        preds_df = append_dict_to_df(
            preds_df, {"": "Predicted", self.target: f"{pred:.{round}f} {self.units}"}
        )
        if index is not None:
            try:
                y_true = self.get_y(index)
                preds_df = append_dict_to_df(
                    preds_df,
                    {"": "Observed", self.target: f"{y_true:.{round}f} {self.units}"},
                )
                preds_df = append_dict_to_df(
                    preds_df,
                    {
                        "": "Residual",
                        self.target: f"{(y_true-pred):.{round}f} {self.units}",
                    },
                )
            except Exception:
                pass
        return preds_df

    def metrics(self, show_metrics: List[str] = None):
        """dict of performance metrics: root_mean_squared_error, mean_absolute_error and R-squared

        Args:
            show_metrics (List): list of metrics to display in order. Defaults
                to None, displaying all metrics.
        """

        if self.y_missing:
            raise ValueError(
                "No y was passed to explainer, so cannot calculate metrics!"
            )
        if self.cv is None:
            metrics_dict = {
                "mean-squared-error": mean_squared_error(self.y, self.preds),
                "root-mean-squared-error": np.sqrt(
                    mean_squared_error(self.y, self.preds)
                ),
                "mean-absolute-error": mean_absolute_error(self.y, self.preds),
                "mean-absolute-percentage-error": mape_score(self.y, self.preds),
                "R-squared": r2_score(self.y, self.preds),
            }
        else:
            metrics_dict = {
                "mean-squared-error": [],
                "root-mean-squared-error": [],
                "mean-absolute-error": [],
                "mean-absolute-percentage-error": [],
                "R-squared": [],
            }
            for train_index, test_index in KFold(n_splits=self.cv, shuffle=True).split(
                self.X
            ):
                X_train, X_test = self.X.iloc[train_index], self.X.iloc[test_index]
                y_train, y_test = self.y.iloc[train_index], self.y.iloc[test_index]
                preds = clone(self.model).fit(X_train, y_train).predict(X_test)
                metrics_dict["mean-squared-error"].append(
                    mean_squared_error(y_test, preds)
                )
                metrics_dict["root-mean-squared-error"].append(
                    np.sqrt(mean_squared_error(y_test, preds))
                )
                metrics_dict["mean-absolute-error"].append(
                    mean_absolute_error(y_test, preds)
                )
                metrics_dict["mean-absolute-percentage-error"].append(
                    mape_score(y_test, preds)
                )
                metrics_dict["R-squared"].append(r2_score(y_test, preds))
            metrics_dict = {k: np.mean(v) for k, v in metrics_dict.items()}

        if metrics_dict["mean-absolute-percentage-error"] > 2:
            print(
                "Warning: mean-absolute-percentage-error is very large "
                f"({metrics_dict['mean-absolute-percentage-error']}), you can hide "
                "it from the metrics by passing parameter show_metrics...",
                flush=True,
            )
        if not show_metrics:
            return metrics_dict
        show_metrics_dict = {}
        for m in show_metrics:
            if callable(m):
                if self.cv is not None:
                    raise ValueError(
                        "custom metrics do not work with permutation_cv for now!"
                    )
                show_metrics_dict[m.__name__] = m(self.y, self.preds)
            elif m in metrics_dict:
                show_metrics_dict[m] = metrics_dict[m]
        return show_metrics_dict

    def metrics_descriptions(self, round=2):
        """Returns a metrics dict, with the metric values replaced by a descriptive
        string, explaining/interpreting the value of the metric

        Returns:
            dict
        """
        metrics_dict = self.metrics()
        metrics_descriptions_dict = {}
        for k, v in metrics_dict.items():
            if k == "mean-squared-error":
                metrics_descriptions_dict[k] = (
                    "A measure of how close "
                    "predicted value fits true values, where large deviations "
                    "are punished more heavily. So the lower this number the "
                    "better the model."
                )
            if k == "root-mean-squared-error":
                metrics_descriptions_dict[k] = (
                    "A measure of how close "
                    "predicted value fits true values, where large deviations "
                    "are punished more heavily. So the lower this number the "
                    "better the model."
                )
            if k == "mean-absolute-error":
                metrics_descriptions_dict[k] = (
                    f"On average predictions deviate "
                    f"{v:.{round}f} {self.units} off the observed value of "
                    f"{self.target} (can be both above or below)"
                )
            if k == "mean-absolute-percentage-error":
                metrics_descriptions_dict[k] = (
                    f"On average predictions deviate "
                    f"{100*v:.{round}f}% off the observed value of "
                    f"{self.target} (can be both above or below)"
                )
            if k == "R-squared":
                metrics_descriptions_dict[k] = (
                    f"{100*v:.{round}f}% of all "
                    f"variation in {self.target} was explained by the model."
                )
        return metrics_descriptions_dict

    def plot_predicted_vs_actual(
        self, round=2, logs=False, log_x=False, log_y=False, plot_sample=None, **kwargs
    ):
        """plot with predicted value on x-axis and actual value on y axis.

        Args:
          round(int, optional): rounding to apply to outcome, defaults to 2
          logs (bool, optional): log both x and y axis, defaults to False
          log_y (bool, optional): only log x axis. Defaults to False.
          log_x (bool, optional): only log y axis. Defaults to False.
          plot_sample (int, optional): Instead of all points only plot a random
            sample of points. Defaults to None (=all points)
          **kwargs:

        Returns:
          Plotly fig

        """
        plot_idxs = self.get_idx_sample(plot_sample)
        if self.y_missing:
            raise ValueError(
                "No y was passed to explainer, so cannot plot predicted vs actual!"
            )
        return plotly_predicted_vs_actual(
            self.y[plot_idxs],
            self.preds[plot_idxs],
            target=self.target,
            units=self.units,
            idxs=self.idxs[plot_idxs],
            logs=logs,
            log_x=log_x,
            log_y=log_y,
            round=round,
            index_name=self.index_name,
        )

    def plot_residuals(
        self, vs_actual=False, round=2, residuals="difference", plot_sample=None
    ):
        """plot of residuals. x-axis is the predicted outcome by default

        Args:
          vs_actual(bool, optional): use actual value for x-axis,
                    defaults to False
          round(int, optional): rounding to perform on values, defaults to 2
          residuals (str, {'difference', 'ratio', 'log-ratio'} optional):
                    How to calcualte residuals. Defaults to 'difference'.
          plot_sample (int, optional): Instead of all points only plot a random
            sample of points. Defaults to None (=all points)

        Returns:
          Plotly fig

        """
        if self.y_missing:
            raise ValueError("No y was passed to explainer, so cannot plot residuals!")

        plot_idxs = self.get_idx_sample(plot_sample)
        return plotly_plot_residuals(
            self.y[plot_idxs],
            self.preds[plot_idxs],
            idxs=self.idxs[plot_idxs],
            vs_actual=vs_actual,
            target=self.target,
            units=self.units,
            residuals=residuals,
            round=round,
            index_name=self.index_name,
        )

    def plot_residuals_vs_feature(
        self,
        col,
        residuals="difference",
        round=2,
        dropna=True,
        points=True,
        winsor=0,
        topx=None,
        sort="alphabet",
        plot_sample=None,
    ):
        """Plot residuals vs individual features

        Args:
          col(str): Plot against feature col
          residuals (str, {'difference', 'ratio', 'log-ratio'} optional):
                    How to calcualte residuals. Defaults to 'difference'.
          round(int, optional): rounding to perform on residuals, defaults to 2
          dropna(bool, optional): drop missing values from plot, defaults to True.
          points (bool, optional): display point cloud next to violin plot.
                    Defaults to True.
          winsor (int, 0-50, optional): percentage of outliers to winsor out of
                    the y-axis. Defaults to 0.
          plot_sample (int, optional): Instead of all points only plot a random
            sample of points. Defaults to None (=all points)

        Returns:
          plotly fig
        """
        if self.y_missing:
            raise ValueError("No y was passed to explainer, so cannot plot residuals!")
        assert col in self.merged_cols, f"{col} not in explainer.merged_cols!"

        plot_idxs = self.get_idx_sample(plot_sample)
        col_vals = self.get_col(col).iloc[plot_idxs]
        na_mask = (
            col_vals != self.na_fill if dropna else np.array([True] * len(col_vals))
        )
        if col in self.cat_cols:
            return plotly_residuals_vs_col(
                self.y[plot_idxs][na_mask],
                self.preds[plot_idxs][na_mask],
                col_vals[plot_idxs][na_mask],
                residuals=residuals,
                idxs=self.idxs[plot_idxs].values[na_mask],
                points=points,
                round=round,
                winsor=winsor,
                index_name=self.index_name,
                cats_order=self.ordered_cats(col, topx, sort),
            )
        else:
            return plotly_residuals_vs_col(
                self.y[plot_idxs][na_mask],
                self.preds[plot_idxs][na_mask],
                col_vals[plot_idxs][na_mask],
                residuals=residuals,
                idxs=self.idxs[plot_idxs].values[na_mask],
                points=points,
                round=round,
                winsor=winsor,
                index_name=self.index_name,
            )

    def plot_y_vs_feature(
        self,
        col,
        residuals="difference",
        round=2,
        dropna=True,
        points=True,
        winsor=0,
        topx=None,
        sort="alphabet",
        plot_sample=None,
    ):
        """Plot y vs individual features

        Args:
          col(str): Plot against feature col
          round(int, optional): rounding to perform on residuals, defaults to 2
          dropna(bool, optional): drop missing values from plot, defaults to True.
          points (bool, optional): display point cloud next to violin plot.
                    Defaults to True.
          winsor (int, 0-50, optional): percentage of outliers to winsor out of
                    the y-axis. Defaults to 0.
          plot_sample (int, optional): Instead of all points only plot a random
            sample of points. Defaults to None (=all points)

        Returns:
          plotly fig
        """
        if self.y_missing:
            raise ValueError(
                "No y was passed to explainer, so cannot plot y vs feature!"
            )
        assert col in self.merged_cols, f"{col} not in explainer.merged_cols!"

        plot_idxs = self.get_idx_sample(plot_sample)
        col_vals = self.get_col(col).iloc[plot_idxs]
        na_mask = (
            col_vals != self.na_fill if dropna else np.array([True] * len(col_vals))
        )
        if col in self.cat_cols:
            return plotly_actual_vs_col(
                self.y[plot_idxs][na_mask],
                self.preds[plot_idxs][na_mask],
                col_vals[plot_idxs][na_mask],
                idxs=self.idxs[plot_idxs].values[na_mask],
                points=points,
                round=round,
                winsor=winsor,
                units=self.units,
                target=self.target,
                index_name=self.index_name,
                cats_order=self.ordered_cats(col, topx, sort),
            )
        else:
            return plotly_actual_vs_col(
                self.y[plot_idxs][na_mask],
                self.preds[plot_idxs][na_mask],
                col_vals[plot_idxs][na_mask],
                idxs=self.idxs[plot_idxs].values[na_mask],
                points=points,
                round=round,
                winsor=winsor,
                units=self.units,
                target=self.target,
                index_name=self.index_name,
            )

    def plot_preds_vs_feature(
        self,
        col,
        residuals="difference",
        round=2,
        dropna=True,
        points=True,
        winsor=0,
        topx=None,
        sort="alphabet",
        plot_sample=None,
    ):
        """Plot y vs individual features

        Args:
          col(str): Plot against feature col
          round(int, optional): rounding to perform on residuals, defaults to 2
          dropna(bool, optional): drop missing values from plot, defaults to True.
          points (bool, optional): display point cloud next to violin plot.
                    Defaults to True.
          winsor (int, 0-50, optional): percentage of outliers to winsor out of
                    the y-axis. Defaults to 0.
          plot_sample (int, optional): Instead of all points only plot a random
            sample of points. Defaults to None (=all points)

        Returns:
          plotly fig
        """
        assert col in self.merged_cols, f"{col} not in explainer.merged_cols!"

        plot_idxs = self.get_idx_sample(plot_sample)
        col_vals = self.get_col(col).iloc[plot_idxs]
        na_mask = (
            col_vals != self.na_fill if dropna else np.array([True] * len(col_vals))
        )
        if col in self.cat_cols:
            return plotly_preds_vs_col(
                self.y[plot_idxs][na_mask],
                self.preds[plot_idxs][na_mask],
                col_vals[plot_idxs][na_mask],
                idxs=self.idxs[plot_idxs].values[na_mask],
                points=points,
                round=round,
                winsor=winsor,
                units=self.units,
                target=self.target,
                index_name=self.index_name,
                cats_order=self.ordered_cats(col, topx, sort),
            )
        else:
            return plotly_preds_vs_col(
                self.y[plot_idxs][na_mask],
                self.preds[plot_idxs][na_mask],
                col_vals[plot_idxs][na_mask],
                idxs=self.idxs[plot_idxs].values[na_mask],
                points=points,
                round=round,
                winsor=winsor,
                units=self.units,
                target=self.target,
                index_name=self.index_name,
            )


class TreeExplainer(BaseExplainer):
    @property
    def is_tree_explainer(self):
        """this is a TreeExplainer"""
        return True

    @property
    def no_of_trees(self):
        """The number of trees in the RandomForest model"""
        raise NotImplementedError

    @property
    def graphviz_available(self):
        """ """
        if not hasattr(self, "_graphviz_available"):
            try:
                import graphviz.backend.execute as be

                cmd = ["dot", "-V"]
                be.run_check(cmd, capture_output=True, check=True, quiet=True)
            except Exception:
                print(
                    """
                WARNING: you don't seem to have graphviz in your path (cannot run 'dot -V'), 
                so no dtreeviz visualisation of decision trees will be shown on the shadow trees tab.

                See https://github.com/parrt/dtreeviz for info on how to properly install graphviz 
                for dtreeviz. 
                """
                )
                self._graphviz_available = False
            else:
                self._graphviz_available = True
        return self._graphviz_available

    @property
    def shadow_trees(self):
        """a list of ShadowDecTree objects"""
        raise NotImplementedError

    @insert_pos_label
    def get_decisionpath_df(self, tree_idx, index, pos_label=None):
        """dataframe with all decision nodes of a particular decision tree
        for a particular observation.

        Args:
          tree_idx: the n'th tree in the random forest
          index: row index
          pos_label:  positive class (Default value = None)

        Returns:
          dataframe with summary of the decision tree path

        """
        assert (
            tree_idx >= 0 and tree_idx < len(self.shadow_trees)
        ), f"tree index {tree_idx} outside 0 and number of trees ({len(self.decision_trees)}) range"
        X_row = self.get_X_row(index)
        if self.is_classifier:
            return get_decisionpath_df(
                self.shadow_trees[tree_idx], X_row.squeeze(), pos_label=pos_label
            )
        else:
            return get_decisionpath_df(self.shadow_trees[tree_idx], X_row.squeeze())

    @insert_pos_label
    def get_decisionpath_summary_df(self, tree_idx, index, round=2, pos_label=None):
        """formats decisiontree_df in a slightly more human readable format.

        Args:
          tree_idx: the n'th tree in the random forest or boosted ensemble
          index: index
          round: rounding to apply to floats (Default value = 2)
          pos_label:  positive class (Default value = None)

        Returns:
          dataframe with summary of the decision tree path

        """
        return get_decisiontree_summary_df(
            self.get_decisionpath_df(tree_idx, index, pos_label=pos_label),
            classifier=self.is_classifier,
            round=round,
            units=self.units,
        )

    def decisiontree_view(self, tree_idx, index, show_just_path=False):
        """get a dtreeviz visualization of a particular tree in the random forest.

        Args:
          tree_idx: the n'th tree in the random forest
          index: row index
          show_just_path (bool, optional): show only the path not rest of the
                    tree. Defaults to False.

        Returns:
          DTreeVizRender

        """
        if not self.graphviz_available:
            print("No graphviz 'dot' executable available!")
            return None

        viz = DTreeVizAPI(self.shadow_trees[tree_idx])

        return viz.view(
            x=self.get_X_row(index).squeeze(),
            fancy=False,
            show_node_labels=False,
            show_just_path=show_just_path,
        )

    def decisiontree_file(self, tree_idx, index, show_just_path=False):
        return self.decisiontree_view(tree_idx, index, show_just_path).save_svg()

    def decisiontree(self, tree_idx, index, show_just_path=False):
        """get a dtreeviz visualization of a particular tree in the random forest.

        Args:
          tree_idx: the n'th tree in the random forest
          index: row index
          show_just_path (bool, optional): show only the path not rest of the
                    tree. Defaults to False.

        Returns:
          a IPython display SVG object for e.g. jupyter notebook.

        """
        from IPython.display import SVG

        return SVG(self.decisiontree_view(tree_idx, index, show_just_path).svg())

    def decisiontree_encoded(self, tree_idx, index, show_just_path=False):
        """get a dtreeviz visualization of a particular tree in the random forest.

        Args:
          tree_idx: the n'th tree in the random forest
          index: row index
          show_just_path (bool, optional): show only the path not rest of the
                    tree. Defaults to False.

        Returns:
          a base64 encoded image, for inclusion in websites (e.g. dashboard)


        """
        if not self.graphviz_available:
            print("No graphviz 'dot' executable available!")
            return None
        svg = open(self.decisiontree_file(tree_idx, index, show_just_path), "rb").read()
        encoded = base64.b64encode(svg)
        svg_encoded = "data:image/svg+xml;base64,{}".format(encoded.decode())
        return svg_encoded

    @insert_pos_label
    def plot_trees(
        self, index, highlight_tree=None, round=2, higher_is_better=True, pos_label=None
    ):
        """plot barchart predictions of each individual prediction tree

        Args:
          index: index to display predictions for
          highlight_tree:  tree to highlight in plot (Default value = None)
          round: rounding of numbers in plot (Default value = 2)
          higher_is_better (bool): flip red and green. Dummy bool for compatibility
                with gbm plot_trees().
          pos_label: positive class (Default value = None)

        Returns:

        """

        raise NotImplementedError

    def calculate_properties(self, include_interactions=True):
        """

        Args:
          include_interactions:  If False do not calculate shap interaction value
            (Default value = True)

        Returns:

        """
        _ = self.shadow_trees
        super().calculate_properties(include_interactions=include_interactions)


class RandomForestExplainer(TreeExplainer):
    """RandomForestExplainer allows for the analysis of individual DecisionTrees that
    make up the RandomForestClassifier or RandomForestRegressor."""

    @property
    def no_of_trees(self):
        """The number of trees in the RandomForest model"""
        return len(self.model.estimators_)

    @property
    def shadow_trees(self):
        """a list of ShadowDecTree objects"""
        if not hasattr(self, "_shadow_trees"):
            print(
                "Calculating ShadowDecTree for each individual decision tree...",
                flush=True,
            )
            assert hasattr(
                self.model, "estimators_"
            ), """self.model does not have an estimators_ attribute, so probably not
                actually a sklearn RandomForest?"""
            y = self.y if self.y_missing else self.y.astype("int16")
            self._shadow_trees = [
                ShadowDecTree.get_shadow_tree(
                    decision_tree,
                    self.X,
                    y,
                    feature_names=self.X.columns.tolist(),
                    target_name="target",
                    class_names=self.labels if self.is_classifier else None,
                )
                for decision_tree in self.model.estimators_
            ]
        return self._shadow_trees

    @insert_pos_label
    def plot_trees(
        self, index, highlight_tree=None, round=2, higher_is_better=True, pos_label=None
    ):
        """plot barchart predictions of each individual prediction tree

        Args:
          index: index to display predictions for
          highlight_tree:  tree to highlight in plot (Default value = None)
          round: rounding of numbers in plot (Default value = 2)
          higher_is_better (bool): flip red and green. Dummy bool for compatibility
                with gbm plot_trees().
          pos_label: positive class (Default value = None)

        Returns:

        """

        X_row = self.get_X_row(index)
        y = self.get_y(index)

        if self.is_classifier:
            pos_label = self.pos_label_index(pos_label)
            if y is not None:
                y = 100 * int(y == pos_label)
            return plotly_rf_trees(
                self.model,
                X_row,
                y,
                highlight_tree=highlight_tree,
                round=round,
                pos_label=pos_label,
                target=self.target,
            )
        else:
            return plotly_rf_trees(
                self.model,
                X_row,
                y,
                highlight_tree=highlight_tree,
                round=round,
                target=self.target,
                units=self.units,
            )


class XGBExplainer(TreeExplainer):
    """XGBExplainer allows for the analysis of individual DecisionTrees that
    make up the xgboost model.
    """

    @property
    def model_dump_list(self):
        if not hasattr(self, "_model_dump_list"):
            print("Generating xgboost model dump...", flush=True)
            self._model_dump_list = self.model.get_booster().get_dump()
        return self._model_dump_list

    @property
    def no_of_trees(self):
        """The number of trees in the RandomForest model"""
        if self.is_classifier and len(self.labels) > 2:
            # for multiclass classification xgboost generates a seperate
            # tree for each class
            return int(len(self.model_dump_list) / len(self.labels))
        return len(self.model_dump_list)

    @property
    def shadow_trees(self):
        """a list of ShadowDecTree objects"""
        if not hasattr(self, "_shadow_trees"):
            print(
                "Calculating ShadowDecTree for each individual decision tree...",
                flush=True,
            )

            self._shadow_trees = [
                ShadowDecTree.get_shadow_tree(
                    self.model.get_booster(),
                    self.X,
                    self.y.astype("int32"),
                    feature_names=self.X.columns.tolist(),
                    target_name="target",
                    class_names=self.labels if self.is_classifier else None,
                    tree_index=i,
                )
                for i in range(len(self.model_dump_list))
            ]
        return self._shadow_trees

    @insert_pos_label
    def get_decisionpath_df(self, tree_idx, index, pos_label=None):
        """dataframe with all decision nodes of a particular decision tree

        Args:
          tree_idx: the n'th tree in the random forest
          index: row index
          round:  (Default value = 2)
          pos_label:  positive class (Default value = None)

        Returns:
          dataframe with summary of the decision tree path

        """
        assert (
            tree_idx >= 0 and tree_idx < self.no_of_trees
        ), f"tree index {tree_idx} outside 0 and number of trees ({len(self.decision_trees)}) range"

        if self.is_classifier:
            if len(self.labels) > 2:
                # for multiclass classification xgboost generates a seperate
                # tree for each class
                tree_idx = tree_idx * len(self.labels) + pos_label
        return get_xgboost_path_df(
            self.model_dump_list[tree_idx], self.get_X_row(index)
        )

    def get_decisionpath_summary_df(self, tree_idx, index, round=2, pos_label=None):
        """formats decisiontree_df in a slightly more human readable format.
        Args:
          tree_idx: the n'th tree in the random forest
          index: row index
          round:  (Default value = 2)
          pos_label:  positive class (Default value = None)
        Returns:
          dataframe with summary of the decision tree path
        """
        return get_xgboost_path_summary_df(
            self.get_decisionpath_df(tree_idx, index, pos_label=pos_label)
        )

    @insert_pos_label
    def decisiontree_view(self, tree_idx, index, show_just_path=False, pos_label=None):
        """get a dtreeviz visualization of a particular tree in the random forest.

        Args:
          tree_idx: the n'th tree in the random forest
          index: row index
          show_just_path (bool, optional): show only the path not rest of the
                    tree. Defaults to False.
          pos_label: for classifiers, positive label class

        Returns:
          the path where the .svg file is stored.

        """
        if not self.graphviz_available:
            print("No graphviz 'dot' executable available!")
            return None

        if self.is_classifier:
            if len(self.labels) > 2:
                tree_idx = tree_idx * len(self.labels) + pos_label

        viz = DTreeVizAPI(self.shadow_trees[tree_idx])

        return viz.view(
            x=self.get_X_row(index).squeeze(),
            fancy=False,
            show_node_labels=False,
            show_just_path=show_just_path,
        )

    @insert_pos_label
    def plot_trees(
        self, index, highlight_tree=None, round=2, higher_is_better=True, pos_label=None
    ):
        """plot barchart predictions of each individual prediction tree

        Args:
          index: index to display predictions for
          highlight_tree:  tree to highlight in plot (Default value = None)
          round: rounding of numbers in plot (Default value = 2)
          higher_is_better (bool, optional): up is green, down is red. If False
            flip the colors.
          pos_label: positive class (Default value = None)

        Returns:

        """
        if self.is_classifier:
            pos_label = self.pos_label_index(pos_label)
            y = self.get_y(index)
            y = int(y == pos_label) if y is not None else y
            xgboost_preds_df = get_xgboost_preds_df(
                self.model, self.get_X_row(index), pos_label=pos_label
            )
            return plotly_xgboost_trees(
                xgboost_preds_df,
                y=y,
                highlight_tree=highlight_tree,
                target=self.target,
                higher_is_better=higher_is_better,
            )
        else:
            X_row = self.get_X_row(index)
            y = self.get_y(index)
            xgboost_preds_df = get_xgboost_preds_df(self.model, X_row)
            return plotly_xgboost_trees(
                xgboost_preds_df,
                y=y,
                highlight_tree=highlight_tree,
                target=self.target,
                units=self.units,
                higher_is_better=higher_is_better,
            )

    def calculate_properties(self, include_interactions=True):
        """

        Args:
          include_interactions:  If False do not calculate shap interaction value
            (Default value = True)

        Returns:
        """
        _ = self.shadow_trees, self.model_dump_list
        super().calculate_properties(include_interactions=include_interactions)


class RandomForestClassifierExplainer(RandomForestExplainer, ClassifierExplainer):
    """RandomForestClassifierExplainer inherits from both RandomForestExplainer and
    ClassifierExplainer.
    """

    pass


class RandomForestRegressionExplainer(RandomForestExplainer, RegressionExplainer):
    """RandomForestRegressionExplainer inherits from both RandomForestExplainer and
    RegressionExplainer.
    """

    pass


class XGBClassifierExplainer(XGBExplainer, ClassifierExplainer):
    """RandomForestClassifierBunch inherits from both RandomForestExplainer and
    ClassifierExplainer.
    """

    pass


class XGBRegressionExplainer(XGBExplainer, RegressionExplainer):
    """XGBRegressionExplainer inherits from both XGBExplainer and
    RegressionExplainer.
    """

    pass
