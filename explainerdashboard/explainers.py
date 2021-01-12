__all__ = ['BaseExplainer', 
            'ClassifierExplainer', 
            'RegressionExplainer', 
            'RandomForestClassifierExplainer', 
            'RandomForestRegressionExplainer',
            'XGBClassifierExplainer',
            'XGBRegressionExplainer',
            'ClassifierBunch', # deprecated
            'RegressionBunch', # deprecated
            'RandomForestClassifierBunch', # deprecated
            'RandomForestRegressionBunch', # deprecated
            ]

from abc import ABC
import base64
from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_string_dtype

import shap

from dtreeviz.trees import ShadowDecTree, dtreeviz

from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.metrics import precision_score, recall_score, log_loss
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import average_precision_score

from .explainer_methods import *
from .explainer_plots import *
from .make_callables import make_callable, default_list, default_2darray

import plotly.io as pio
pio.templates.default = "none"

class BaseExplainer(ABC):
    """ """
    def __init__(self, model, X, y=None, permutation_metric=r2_score, 
                    shap="guess", X_background=None, model_output="raw",
                    cats=None, idxs=None, index_name=None, target=None,
                    descriptions=None, 
                    n_jobs=None, permutation_cv=None, na_fill=-999):
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
            idxs (pd.Series): list of row identifiers. Can be names, id's, etc. 
                Defaults to X.index.
            index_name (str): identifier for row indexes. e.g. index_name='Passenger'.
                Defaults to X.index.name or idxs.name.
            target: name of the predicted target, e.g. "Survival", 
                "Ticket price", etc. Defaults to y.name.
            n_jobs (int): for jobs that can be parallelized using joblib,
                how many processes to split the job in. For now only used
                for calculating permutation importances. Defaults to None.
            permutation_cv (int): If not None then permutation importances 
                will get calculated using cross validation across X. 
                This is for calculating permutation importances against
                X_train. Defaults to None
            na_fill (int): The filler used for missing values, defaults to -999.
        """
        self._params_dict = dict(
            shap=shap, model_output=model_output, cats=cats, 
            descriptions=descriptions, target=target, n_jobs=n_jobs, 
            permutation_cv=n_jobs, na_fill=na_fill)

        if isinstance(model, Pipeline):
            self.X, self.model = split_pipeline(model, X)
            self.X_background, _ = split_pipeline(model, X_background, verbose=0)
        else:
            self.X, self.X_background = X, X_background
            self.model = model

        if safe_is_instance(model, "xgboost.core.Booster"):
            raise ValueError("For xgboost models, currently only the scikit-learn "
                "compatible wrappers xgboost.sklearn.XGBClassifier and "
                "xgboost.sklearn.XGBRegressor are supported, so please use those "
                "instead of xgboost.Booster!")

        if safe_is_instance(model, "lightgbm.Booster"):
            raise ValueError("For lightgbm, currently only the scikit-learn "
                "compatible wrappers lightgbm.LGBMClassifier and lightgbm.LGBMRegressor "
                "are supported, so please use those instead of lightgbm.Booster!")

        self.onehot_cols, self.onehot_dict = parse_cats(self.X, cats)
        self.categorical_cols = [col for col in X.columns if not is_numeric_dtype(X[col])]
        self.categorical_dict = {col:sorted(X[col].unique().tolist()) for col in self.categorical_cols}
        self.cat_cols = self.onehot_cols + self.categorical_cols
        if self.categorical_cols:
            print(f"Warning: Detected the following categorical columns: {self.categorical_cols}."
                    "Unfortunately for now shap interaction values do not work with"
                    "categorical columns.", flush=True)
            self.interactions_should_work = False

        if y is not None:
            self.y = pd.Series(y)
            self.y_missing = False
        else:
            self.y = pd.Series(np.full(len(X), np.nan))
            self.y_missing = True
        if self.y.name is None: self.y.name = 'Target'
        
        self.metric = permutation_metric

        if shap == "guess":
            shap_guess = guess_shap(self.model)
            if shap_guess is not None:
                model_str = str(type(self.model))\
                    .replace("'", "").replace("<", "").replace(">", "")\
                    .split(".")[-1]
                print(f"Note: shap=='guess' so guessing for {model_str}"
                      f" shap='{shap_guess}'...")
                self.shap = shap_guess
            else:
                raise ValueError(
                    "Parameter shap='gues'', but failed to to guess the type of "
                    "shap explainer to use. "
                    "Please explicitly pass a `shap` parameter to the explainer, "
                    "e.g. shap='tree', shap='linear', etc.")
        else:
            assert shap in ['tree', 'linear', 'deep', 'kernel'], \
                "Only shap='guess', 'tree', 'linear', 'deep', or ' kernel' allowed."
            self.shap = shap

        self.model_output = model_output

        if idxs is not None:
            assert len(idxs) == len(self.X) == len(self.y), \
                ("idxs should be same length as X but is not: "
                f"len(idxs)={len(idxs)} but  len(X)={len(self.X)}!")
            self.idxs = pd.Index(idxs, dtype=str)
        else:
            self.idxs = X.index.astype(str)
        self.X.index = self.idxs
        self.y.index = self.idxs

        if index_name is None:
            if self.idxs.name is not None:
                self.index_name = self.idxs.name.capitalize()
            else:
                self.index_name = "Index"
        else:
            self.index_name = index_name.capitalize()

        self.descriptions = {} if descriptions is None else descriptions
        self.target = target if target is not None else self.y.name
        self.n_jobs = n_jobs
        self.permutation_cv = permutation_cv
        self.na_fill = na_fill
        self.columns = self.X.columns.tolist()
        self.pos_label = None
        self.units = ""
        self.is_classifier = False
        self.is_regression = False
        self.interactions_should_work = True
        if safe_is_instance(self.model, "CatBoostRegressor", "CatBoostClassifier"):
            self.interactions_should_work = False
        else:
            self.interactions_should_work = True

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

    def to_yaml(self, filepath=None, return_dict=False,
                    modelfile="model.pkl",
                    datafile="data.csv",
                    index_col=None,
                    target_col=None,
                    explainerfile="explainer.joblib",
                    dashboard_yaml="dashboard.yaml"):
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
                params=self._params_dict))
        if return_dict:
            return yaml_config

        if filepath is not None:
            yaml.dump(yaml_config, open(filepath, "w"))
            return
        return yaml.dump(yaml_config)

    def __len__(self):
        return len(self.X)

    def __contains__(self, index):
        if self.get_int_idx(index) is not None:
            return True
        return False

    def check_cats(self, col1, col2=None):
        """check whether should use cats=True based on col1 (and col2)

        Args:
          col1: First column
          col2:  Second column (Default value = None)

        Returns:
          Boolean whether cats should be True

        """
        if col2 is None:
            if col1 in self.columns:
                return False
            elif col1 in self.columns_cats:
                return True
            raise ValueError(f"Can't find {col1}.")
        
        if col1 not in self.columns and col1 not in self.columns_cats:
            raise ValueError(f"Can't find {col1}.")
        if col2 not in self.columns and col2 not in self.columns_cats:
            raise ValueError(f"Can't find {col2}.")
        
        if col1 in self.columns and col2 in self.columns:
            return False
        if col1 in self.columns_cats and col2 in self.columns_cats:
            return True
        if col1 in self.columns_cats and not col2 in self.columns_cats:
            raise ValueError(
                f"{col1} is categorical but {col2} is not in columns_cats")
        if col2 in self.columns_cats and not col1 in self.columns_cats:
            raise ValueError(
                f"{col2} is categorical but {col1} is not in columns_cats")

    @property
    def shap_explainer(self):
        """ """
        if not hasattr(self, '_shap_explainer'):
            X_str = ", X_background" if self.X_background is not None else 'X'
            NoX_str = ", X_background" if self.X_background is not None else ''
            if self.shap == 'tree':
                print("Generating self.shap_explainer = "
                      f"shap.TreeExplainer(model{NoX_str})")
                self._shap_explainer = shap.TreeExplainer(self.model)
            elif self.shap=='linear':
                if self.X_background is None:
                    print(
                        "Warning: shap values for shap.LinearExplainer get "
                        "calculated against X_background, but paramater "
                        "X_background=None, so using X instead")
                print(f"Generating self.shap_explainer = shap.LinearExplainer(model{X_str})...")
                self._shap_explainer = shap.LinearExplainer(self.model, 
                    self.X_background if self.X_background is not None else self.X)
            elif self.shap=='deep':
                print(f"Generating self.shap_explainer = "
                      f"shap.DeepExplainer(model{NoX_str})")
                self._shap_explainer = shap.DeepExplainer(self.model)
            elif self.shap=='kernel': 
                if self.X_background is None:
                    print(
                        "Warning: shap values for shap.LinearExplainer get "
                        "calculated against X_background, but paramater "
                        "X_background=None, so using X instead")
                print("Generating self.shap_explainer = "
                        f"shap.KernelExplainer(model, {X_str})...")
                self._shap_explainer = shap.KernelExplainer(self.model, 
                    self.X_background if self.X_background is not None else self.X)
        return self._shap_explainer

    def get_int_idx(self, index):
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
        return None

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

    def random_index(self, y_min=None, y_max=None, pred_min=None, pred_max=None, 
                        return_str=False, **kwargs):
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
            if y_min is None: y_min = self.y.min()
            if y_max is None: y_max = self.y.max()

            potential_idxs = self.y[(self.y>=y_min) & 
                                    (self.y <= y_max) & 
                                    (self.preds>=pred_min) &    
                                    (self.preds <= pred_max)].index
        else:
            potential_idxs = self.y[(self.preds>=pred_min) &    
                                    (self.preds <= pred_max)].index

        if len(potential_idxs) > 0:
            idx = np.random.choice(potential_idxs)
        else:
            return None
        if return_str:
            return idx
        return idxs.get_loc(idx)

    @property
    def preds(self):
        """returns model model predictions"""
        if not hasattr(self, '_preds'):
            print("Calculating predictions...", flush=True)
            self._preds = self.model.predict(self.X).astype(np.float64)
            
        return self._preds
    
    @property
    def pred_percentiles(self):
        """returns percentile rank of model predictions"""
        if not hasattr(self, '_pred_percentiles'):
            print("Calculating prediction percentiles...", flush=True)
            self._pred_percentiles = (pd.Series(self.preds)
                                .rank(method='min')
                                .divide(len(self.preds))
                                .values)
        return make_callable(self._pred_percentiles)

    def columns_ranked_by_shap(self, cats=False, pos_label=None):
        """returns the columns of X, ranked by mean abs shap value

        Args:
          cats: Group categorical together (Default value = False)
          pos_label:  (Default value = None)

        Returns:
          list of columns

        """
        if cats:
            return self.mean_abs_shap_cats(pos_label).Feature.tolist()
        else:
            return self.mean_abs_shap(pos_label).Feature.tolist()

    def n_features(self, cats=False):
        """number of features with cats=True or cats=False

        Args:
          cats:  (Default value = False)

        Returns:
            int, number of features

        """
        if cats:
            return len(self.columns_cats)
        else:
            return len(self.columns)

    def equivalent_col(self, col):
        """Find equivalent col in columns_cats or columns
        
        if col in self.columns, return equivalent col in self.columns_cats,
                e.g. equivalent_col('Gender_Male') -> 'Gender'
        if col in self.columns_cats, return first one hot encoded col,
                e.g. equivalent_col('Gender') -> 'Gender_Male'
        
        (useful for switching between cats=True and cats=False, while
            maintaining column selection)

        Args:
          col:  col to get equivalent col for

        Returns:
          col
        """
        if col in self.columns_cats:
            # first onehot-encoded columns
            return self.onehot_dict[col][0]
        elif col in self.columns:
            # the cat that the col belongs to
            return [k for k, v in self.onehot_dict.items() if col in v][0]
        return None

    def ordered_cats(self, col, topx=None, sort='alphabet'):
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
        assert col in self.cat_cols, \
            f"{col} is not a categorical feature!"
        if sort=='alphabet':
            if topx is None:
                return sorted(self.X_cats[col].unique().tolist())
            else:
                return sorted(self.X_cats[col].unique().tolist())[:topx]
        elif sort=='freq':
            if topx is None:
                return self.X_cats[col].value_counts().index.tolist()
            else:
                return self.X_cats[col].value_counts().nlargest(topx).index.tolist()
        elif sort=='shap':
            if topx is None:
                return (pd.Series(self.shap_values_cats[:, self.columns_cats.index(col)], 
                        index=self.X_cats[col]).abs().groupby(level=0).mean()
                            .sort_values(ascending=False).index.tolist())
            else:
                return (pd.Series(self.shap_values_cats[:, self.columns_cats.index(col)], 
                        index=self.X_cats[col]).abs().groupby(level=0).mean()
                            .sort_values(ascending=False).nlargest(topx).index.tolist())
        else:
            raise ValueError(f"sort='{sort}', but should be in {{'alphabet', 'freq', 'shap'}}")

    def get_row_from_input(self, inputs:List, ranked_by_shap=False):
        """returns a single row pd.DataFrame from a given list of *inputs"""
        if len(inputs)==1 and isinstance(inputs[0], list):
            inputs = inputs[0]
        elif len(inputs)==1 and isinstance(inputs[0], tuple):
            inputs = list(inputs[0])
        else:
            inputs = list(inputs)
        if len(inputs) == len(self.columns_cats):
            cols = self.columns_ranked_by_shap(cats=True) if ranked_by_shap else self.columns_cats
            df = pd.DataFrame(dict(zip(cols, inputs)), index=[0]).fillna(self.na_fill)
            return df[self.columns_cats]
        elif len(inputs) == len(self.columns):
            cols = self.columns_ranked_by_shap() if ranked_by_shap else self.columns
            df = pd.DataFrame(dict(zip(cols, inputs)), index=[0]).fillna(self.na_fill)
            return df[self.columns]
        else:
            raise ValueError(f"len inputs {len(inputs)} should be the same length as either "
                f"explainer.columns_cats ({len(self.columns_cats)}) or "
                f"explainer.columns ({len(self.columns)})!")


    def description(self, col):
        """returns the written out description of what feature col means

        Args:
          col(str): col to get description for

        Returns:
            str, description
        """
        if col in self.descriptions.keys():
            return self.descriptions[col]
        elif self.equivalent_col(col) in self.descriptions.keys():
            return self.descriptions[self.equivalent_col(col)]
        return ""

    def description_list(self, cols):
        """returns a list of descriptions of a list of cols

        Args:
          cols(list): cols to be converted to descriptions

        Returns:
            list of descriptions
        """
        return [self.description(col) for col in cols]

    def get_col(self, col):
        """return pd.Series with values of col

        For categorical feature reverse engineers the onehotencoding.

        Args:
          col: column tof values to be returned

        Returns:
          pd.Series with values of col

        """
        assert col in self.columns or col in self.onehot_cols, \
            f"{col} not in columns!"

        if col in self.X.columns:
            return self.X[col]
        elif col in self.onehot_cols:
            return pd.Series(retrieve_onehot_value(
                self.X, col, self.onehot_dict[col]), name=col)
        
    def get_col_value_plus_prediction(self, col, index=None, X_row=None, pos_label=None):
        """return value of col and prediction for either index or X_row

        Args:
          col: feature col
          index (str or int, optional): index row
          X_row (single row pd.DataFrame, optional): single row of features
        
        Returns:
          tupe(value of col, prediction for index)

        """
        
        assert (col in self.X.columns) or (col in self.onehot_cols),\
            f"{col} not in columns of dataset"
        if index is not None:
            assert index in self, f"index {index} not found"
            idx = self.get_int_idx(index)

            if col in self.X.columns:
                col_value = self.X[col].iloc[idx]
            elif col in self.onehot_cols:
                col_value = retrieve_onehot_value(self.X, col, self.onehot_dict[col])[idx]

            if self.is_classifier:
                if pos_label is None:
                    pos_label = self.pos_label
                prediction = self.pred_probas(pos_label)[idx]
                if self.model_output == 'probability':
                    prediction = 100*prediction
            elif self.is_regression:
                prediction = self.preds[idx]

            return col_value, prediction
        elif X_row is not None:
            assert X_row.shape[0] == 1, "X_Row should be single row dataframe!"
            
            if ((len(X_row.columns) == len(self.X_cats.columns)) and 
                (X_row.columns == self.X_cats.columns).all()):
                X_row = X_cats_to_X(X_row, self.onehot_dict, self.X.columns)    
            else:
                assert (X_row.columns == self.X.columns).all(), \
                    "X_row should have the same columns as self.X or self.X_cats!"
            
            if col in X_row.columns:
                col_value = X_row[col].item()
            elif col in self.onehot_cols:
                col_value = retrieve_onehot_value(X_row, col, self.onehot_dict[col]).item()

            if self.is_classifier:
                if pos_label is None:
                    pos_label = self.pos_label
                prediction = self.model.predict_proba(X_row)[0][pos_label]
                if self.model_output == 'probability':
                    prediction = 100*prediction
            elif self.is_regression:
                prediction = self.model.predict(X_row)[0]
            return col_value, prediction
        else:
            raise ValueError("You need to pass either index or X_row!")


    @property
    def permutation_importances(self):
        """Permutation importances """
        if not hasattr(self, '_perm_imps'):
            print("Calculating importances...", flush=True)
            self._perm_imps = cv_permutation_importances(
                                self.model, self.X, self.y, self.metric,
                                cv=self.permutation_cv,
                                n_jobs=self.n_jobs,
                                needs_proba=self.is_classifier)
        return make_callable(self._perm_imps)

    @property
    def permutation_importances_cats(self):
        """permutation importances with categoricals grouped"""
        if not hasattr(self, '_perm_imps_cats'):
            self._perm_imps_cats = cv_permutation_importances(
                                self.model, self.X, self.y, self.metric, 
                                onehot_dict=self.onehot_dict,
                                cv=self.permutation_cv,
                                n_jobs=self.n_jobs,
                                needs_proba=self.is_classifier)
        return make_callable(self._perm_imps_cats)

    @property
    def X_cats(self):
        """X with categorical variables grouped together"""
        if not hasattr(self, '_X_cats'):
            self._X_cats = merge_categorical_columns(self.X, self.onehot_dict)
        return self._X_cats

    @property
    def columns_cats(self):
        """columns of X with categorical features grouped"""
        if not hasattr(self, '_columns_cats'):
            self._columns_cats = self.X_cats.columns.tolist()
        return self._columns_cats

    @property
    def shap_base_value(self):
        """the intercept for the shap values.
        
        (i.e. 'what would the prediction be if we knew none of the features?')
        """
        if not hasattr(self, '_shap_base_value'):
            # CatBoost needs shap values calculated before expected value
            if not hasattr(self, "_shap_values"):
                _ = self.shap_values
            self._shap_base_value = self.shap_explainer.expected_value
            if isinstance(self._shap_base_value, np.ndarray):
                # shap library now returns an array instead of float
                self._shap_base_value = self._shap_base_value.item()
        return make_callable(self._shap_base_value)

    @property
    def shap_values(self):
        """SHAP values calculated using the shap library"""
        if not hasattr(self, '_shap_values'):
            print("Calculating shap values...", flush=True)
            self._shap_values = self.shap_explainer.shap_values(self.X)
        return make_callable(self._shap_values)
    
    @property
    def shap_values_cats(self):
        """SHAP values when categorical features have been grouped"""
        if not hasattr(self, '_shap_values_cats'):
            self._shap_values_cats = merge_categorical_shap_values(
                        self.X, self.shap_values, self.onehot_dict)
        return make_callable(self._shap_values_cats)

    @property
    def shap_interaction_values(self):
        """SHAP interaction values calculated using shap library"""
        assert self.shap != 'linear', \
            "Unfortunately shap.LinearExplainer does not provide " \
            "shap interaction values! So no interactions tab!"
        if not hasattr(self, '_shap_interaction_values'):
            print("Calculating shap interaction values...", flush=True)
            if self.shap == 'tree':
                print("Reminder: TreeShap computational complexity is O(TLD^2), "
                    "where T is the number of trees, L is the maximum number of"
                    " leaves in any tree and D the maximal depth of any tree. So "
                    "reducing these will speed up the calculation.", 
                    flush=True)
            self._shap_interaction_values = \
                self.shap_explainer.shap_interaction_values(self.X)
        return make_callable(self._shap_interaction_values)

    @property
    def shap_interaction_values_cats(self):
        """SHAP interaction values with categorical features grouped"""
        if not hasattr(self, '_shap_interaction_values_cats'):
            self._shap_interaction_values_cats = \
                merge_categorical_shap_interaction_values(
                    self.shap_interaction_values, self.X, self.X_cats, self.onehot_dict)
        return make_callable(self._shap_interaction_values_cats)

    @property
    def mean_abs_shap(self):
        """Mean absolute SHAP values per feature."""
        if not hasattr(self, '_mean_abs_shap'):
            self._mean_abs_shap = mean_absolute_shap_values(
                                self.columns, self.shap_values)
        return make_callable(self._mean_abs_shap)

    @property
    def mean_abs_shap_cats(self):
        """Mean absolute SHAP values with categoricals grouped."""
        if not hasattr(self, '_mean_abs_shap_cats'):
            self._mean_abs_shap_cats = mean_absolute_shap_values(
                                self.columns_cats, self.shap_values_cats)
        return make_callable(self._mean_abs_shap_cats)

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
        _ = (self.preds,  self.pred_percentiles,
                self.shap_base_value, self.shap_values,
                self.mean_abs_shap)
        if not self.y_missing:
            _ = self.permutation_importances
        if self.onehot_cols:
            _ = (self.mean_abs_shap_cats, self.X_cats,
                    self.shap_values_cats)
        if self.interactions_should_work and include_interactions:
            _ = self.shap_interaction_values
            if self.onehot_cols:
                _ = self.shap_interaction_values_cats

    def metrics(self, *args, **kwargs):
        """returns a dict of metrics.
        
        Implemented by either ClassifierExplainer or RegressionExplainer
        """
        return {}
    
    def mean_abs_shap_df(self, topx=None, cutoff=None, cats=False, pos_label=None):
        """sorted dataframe with mean_abs_shap
        
        returns a pd.DataFrame with the mean absolute shap values per features,
        sorted rom highest to lowest.

        Args:
          topx(int, optional, optional): Only return topx most importance features, defaults to None
          cutoff(float, optional, optional): Only return features with mean abs shap of at least cutoff, defaults to None
          cats(bool, optional, optional): group categorical variables, defaults to False
          pos_label:  (Default value = None)

        Returns:
          pd.DataFrame: shap_df

        """
        if cats:
            shap_df = self.mean_abs_shap_cats(pos_label)
        else:
            shap_df = self.mean_abs_shap(pos_label)

        if topx is None: topx = len(shap_df)
        if cutoff is None: cutoff = shap_df['MEAN_ABS_SHAP'].min()
        return (shap_df[shap_df['MEAN_ABS_SHAP'] >= cutoff]
                    .sort_values('MEAN_ABS_SHAP', ascending=False).head(topx))

    def shap_top_interactions(self, col, topx=None, cats=False, pos_label=None):
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
        if cats:
            if hasattr(self, '_shap_interaction_values'):
                col_idx = self.X_cats.columns.get_loc(col)
                top_interactions = self.X_cats.columns[
                    np.argsort(
                        -np.abs(self.shap_interaction_values_cats(
                            pos_label)[:, col_idx, :]).mean(0))].tolist()
            else:
                top_interactions = self.mean_abs_shap_cats(pos_label)\
                                        .Feature.values.tolist()
                top_interactions.insert(0, top_interactions.pop(
                    top_interactions.index(col))) #put col first

            if topx is None: topx = len(top_interactions)
            return top_interactions[:topx]
        else:
            if hasattr(self, '_shap_interaction_values'):
                col_idx = self.X.columns.get_loc(col)
                top_interactions = self.X.columns[np.argsort(-np.abs(
                            self.shap_interaction_values(
                                pos_label)[:, col_idx, :]).mean(0))].tolist()
            else:
                if hasattr(shap, "utils"):
                    interaction_idxs = shap.utils.approximate_interactions(
                        col, self.shap_values(pos_label), self.X)
                elif hasattr(shap, "common"):
                    # shap < 0.35 has approximate interactions in common
                    interaction_idxs = shap.common.approximate_interactions(
                        col, self.shap_values(pos_label), self.X)

                top_interactions = self.X.columns[interaction_idxs].tolist()
                #put col first
                top_interactions.insert(0, top_interactions.pop(-1)) 

            if topx is None: topx = len(top_interactions)
            return top_interactions[:topx]

    def shap_interaction_values_by_col(self, col, cats=False, pos_label=None):
        """returns the shap interaction values[np.array(N,N)] for feature col

        Args:
          col(str): features for which you'd like to get the interaction value
          cats(bool, optional, optional): group categorical, defaults to False
          pos_label:  (Default value = None)

        Returns:
          np.array(N,N): shap_interaction_values

        """
        if cats:
            return self.shap_interaction_values_cats(pos_label)[:,
                        self.X_cats.columns.get_loc(col), :]
        else:
            return self.shap_interaction_values(pos_label)[:,
                        self.X.columns.get_loc(col), :]

    def permutation_importances_df(self, topx=None, cutoff=None, cats=False, 
                                    pos_label=None):
        """dataframe with features ordered by permutation importance.
        
        For more about permutation importances.
        
        see https://explained.ai/rf-importance/index.html

        Args:
          topx(int, optional, optional): only return topx most important 
                features, defaults to None
          cutoff(float, optional, optional): only return features with importance 
                of at least cutoff, defaults to None
          cats(bool, optional, optional): Group categoricals, defaults to False
          pos_label:  (Default value = None)

        Returns:
          pd.DataFrame: importance_df

        """
        if cats:
            importance_df = self.permutation_importances_cats(pos_label)
        else:
            importance_df = self.permutation_importances(pos_label)

        if topx is None: topx = len(importance_df)
        if cutoff is None: cutoff = importance_df.Importance.min()
        return importance_df[importance_df.Importance >= cutoff].head(topx)

    def importances_df(self, kind="shap", topx=None, cutoff=None, cats=False, 
                        pos_label=None):
        """wrapper function for mean_abs_shap_df() and permutation_importance_df()

        Args:
          kind(str): 'shap' or 'permutations'  (Default value = "shap")
          topx: only display topx highest features (Default value = None)
          cutoff: only display features above cutoff (Default value = None)
          cats: Group categoricals (Default value = False)
          pos_label: Positive class (Default value = None)

        Returns:
          pd.DataFrame

        """
        assert kind=='shap' or kind=='permutation', \
                "kind should either be 'shap' or 'permutation'!"
        if kind=='permutation':
            return self.permutation_importances_df(topx, cutoff, cats, pos_label)
        elif kind=='shap':
            return self.mean_abs_shap_df(topx, cutoff, cats, pos_label)

    def contrib_df(self, index=None, X_row=None, cats=True, topx=None, cutoff=None, sort='abs',
                    pos_label=None):
        """shap value contributions to the prediction for index.
        
        Used as input for the plot_contributions() method.

        Args:
          index(int or str): index for which to calculate contributions
          X_row (pd.DataFrame, single row): single row of feature for which
                to calculate contrib_df. Can us this instead of index
          cats(bool, optional, optional): Group categoricals, defaults to True
          topx(int, optional, optional): Only return topx features, remainder 
                    called REST, defaults to None
          cutoff(float, optional, optional): only return features with at least 
                    cutoff contributions, defaults to None
          sort({'abs', 'high-to-low', 'low-to-high', 'importance'}, optional): sort by 
                    absolute shap value, or from high to low, low to high, or
                    ordered by the global shap importances.
                    Defaults to 'abs'.
          pos_label:  (Default value = None)

        Returns:
          pd.DataFrame: contrib_df

        """
        if pos_label is None:
            pos_label = self.pos_label

        if sort =='importance':
            if cutoff is None:
                cols = self.columns_ranked_by_shap(cats)
            else:
                cols = self.mean_abs_shap_df(cats=cats).query(f"MEAN_ABS_SHAP > {cutoff}").Feature.tolist()
            if topx is not None:
                cols = cols[:topx]
        else:
            cols =  None
        if X_row is not None:
            if ((len(X_row.columns) == len(self.X_cats.columns)) and 
                (X_row.columns == self.X_cats.columns).all()):
                if cats: 
                    X_row_cats = X_row
                    X_row = X_cats_to_X(X_row, self.onehot_dict, self.X.columns)    
            else:
                assert (X_row.columns == self.X.columns).all(), \
                    "X_row should have the same columns as self.X or self.X_cats!"
                X_row_cats = merge_categorical_columns(X_row, self.onehot_dict)

            shap_values = self.shap_explainer.shap_values(X_row)
            if self.is_classifier:
                if not isinstance(shap_values, list) and len(self.labels)==2:
                    shap_values = [-shap_values, shap_values]
                shap_values = shap_values[self.get_pos_label_index(pos_label)]

            if cats:
                shap_values = merge_categorical_shap_values(X_row, shap_values, self.onehot_dict)
                return get_contrib_df(self.shap_base_value(pos_label), shap_values[0], 
                            remove_cat_names(X_row_cats, self.onehot_dict), 
                            topx, cutoff, sort, cols)   
            else:
                return get_contrib_df(self.shap_base_value(pos_label), shap_values[0], 
                            X_row, topx, cutoff, sort, cols)  
        elif index is not None:
            idx = self.get_int_idx(index)
            if cats:
                return get_contrib_df(self.shap_base_value(pos_label), 
                                        self.shap_values_cats(pos_label)[idx],
                                        remove_cat_names(self.X_cats.iloc[[idx]], self.onehot_dict), 
                                        topx, cutoff, sort, cols)
            else:
                return get_contrib_df(self.shap_base_value(pos_label), 
                                        self.shap_values(pos_label)[idx],
                                        self.X.iloc[[idx]], topx, cutoff, sort, cols)
        else:
            raise ValueError("Either index or X_row should be passed!")

    def contrib_summary_df(self, index=None, X_row=None, cats=True, topx=None, cutoff=None, 
                            round=2, sort='abs', pos_label=None):
        """Takes a contrib_df, and formats it to a more human readable format

        Args:
          index: index to show contrib_summary_df for
          X_row (pd.DataFrame, single row): single row of feature for which
                to calculate contrib_df. Can us this instead of index
          cats: Group categoricals (Default value = True)
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
        idx = self.get_int_idx(index) # if passed str convert to int index
        return get_contrib_summary_df(
                    self.contrib_df(idx, X_row, cats, topx, cutoff, sort, pos_label), 
                model_output=self.model_output, round=round, units=self.units, na_fill=self.na_fill)

    def interactions_df(self, col, cats=False, topx=None, cutoff=None, 
                            pos_label=None):
        """dataframe of mean absolute shap interaction values for col

        Args:
          col: Feature to get interactions_df for
          cats: Group categoricals (Default value = False)
          topx: Only display topx most important features (Default value = None)
          cutoff: Only display features with mean abs shap of at least cutoff (Default value = None)
          pos_label: Positive class  (Default value = None)

        Returns:
          pd.DataFrame

        """
        importance_df = mean_absolute_shap_values(
            self.columns_cats if cats else self.columns, 
            self.shap_interaction_values_by_col(col, cats, pos_label))

        if topx is None: topx = len(importance_df)
        if cutoff is None: cutoff = importance_df.MEAN_ABS_SHAP.min()
        return importance_df[importance_df.MEAN_ABS_SHAP >= cutoff].head(topx)
    
    def formatted_contrib_df(self, index, round=None, lang='en', pos_label=None):
        """contrib_df formatted in a particular idiosyncratic way.
        
        Additional language option for output in Dutch (lang='nl')

        Args:
          index(str or int): index to return contrib_df for
          round(int, optional, optional): rounding of continuous features, defaults to 2
          lang(str, optional, optional): language to name the columns, defaults to 'en'
          pos_label:  (Default value = None)

        Returns:
          pd.DataFrame: formatted_contrib_df
        """
        cdf = self.contrib_df(index, cats=True, pos_label=pos_label).copy()
        cdf.reset_index(inplace=True)
        cdf.loc[cdf.col=='base_value', 'value'] = np.nan
        cdf['row_id'] = self.get_int_idx(index)
        cdf['name_id'] = index
        cdf['cat_value'] = np.where(cdf.col.isin(self.onehot_cols), cdf.value, np.nan)
        cdf['cont_value'] = np.where(cdf.col.isin(self.onehot_cols), np.nan, cdf.value)
        if round is not None:
            rounded_cont = np.round(cdf['cont_value'].values.astype(float), round)
            cdf['value'] = np.where(cdf.col.isin(self.onehot_cols), cdf.cat_value, rounded_cont)
        cdf['type'] = np.where(cdf.col.isin(self.onehot_cols), 'cat', 'cont')
        cdf['abs_contribution'] = np.abs(cdf.contribution)
        cdf = cdf[['row_id', 'name_id', 'contribution', 'abs_contribution',
                    'col', 'value', 'cat_value', 'cont_value', 'type', 'index']]
        if lang == 'nl':
            cdf.columns = ['row_id', 'name_id', 'SHAP', 'ABS_SHAP', 'Variabele', 'Waarde',
                            'Cat_Waarde', 'Cont_Waarde', 'Waarde_Type', 'Variabele_Volgorde']
            return cdf

        cdf.columns = ['row_id', 'name_id', 'SHAP', 'ABS_SHAP', 'Feature', 'Value',
                        'Cat_Value', 'Cont_Value', 'Value_Type', 'Feature_Order']
        return cdf

    def pdp_df(self, col, index=None, X_row=None, drop_na=True, sample=500, 
                    n_grid_points=10, pos_label=None, sort='freq'):
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
        assert col in self.X.columns or col in self.onehot_cols, \
            f"{col} not in columns of dataset"
        if col in self.onehot_cols:
            features = self.ordered_cats(col, n_grid_points, sort)
            if index is not None or X_row is not None:
                val, pred = self.get_col_value_plus_prediction(col, index, X_row)
                if val not in features:
                    features[-1] = val
            grid_values = None
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
                vals = np.delete(self.X[col].values, np.where(self.X[col].values==self.na_fill), axis=0)
                grid_values = get_grid_points(vals, n_grid_points=n_grid_points)
            else:
                grid_values = get_grid_points(self.X[col].values, n_grid_points=n_grid_points)
            if index is not None or X_row is not None:
                val, pred = self.get_col_value_plus_prediction(col, index, X_row)
                if val not in grid_values:
                    grid_values = np.append(grid_values, val).sort()

        if pos_label is None: 
            pos_label = self.pos_label

        if index is not None:
            index = self.get_index(index)
            if isinstance(features, str) and drop_na: # regular col, not onehotencoded
                sample_size=min(sample, len(self.X[(self.X[features] != self.na_fill)])-1)
                sampleX = pd.concat([
                    self.X[self.X.index==index],
                    self.X[(self.X.index != index) & (self.X[features] != self.na_fill)]\
                            .sample(sample_size)],
                    ignore_index=True, axis=0)
            else:
                sample_size = min(sample, len(self.X)-1)
                sampleX = pd.concat([
                    self.X[self.X.index==index],
                    self.X[(self.X.index!=index)].sample(sample_size)],
                    ignore_index=True, axis=0)
        elif X_row is not None:
            if ((len(X_row.columns) == len(self.X_cats.columns)) and 
                (X_row.columns == self.X_cats.columns).all()):
                X_row = X_cats_to_X(X_row, self.onehot_dict, self.X.columns)    
            else:
                assert (X_row.columns == self.X.columns).all(), \
                    "X_row should have the same columns as self.X or self.X_cats!"

            if isinstance(features, str) and drop_na: # regular col, not onehotencoded
                sample_size=min(sample, len(self.X[(self.X[features] != self.na_fill)])-1)
                sampleX = pd.concat([
                    X_row,
                    self.X[(self.X[features] != self.na_fill)]\
                            .sample(sample_size)],
                    ignore_index=True, axis=0)
            else:
                sample_size = min(sample, len(self.X)-1)
                sampleX = pd.concat([
                    X_row,
                    self.X.sample(sample_size)],
                    ignore_index=True, axis=0)
        else:
            if isinstance(features, str) and drop_na: # regular col, not onehotencoded
                sample_size=min(sample, len(self.X[(self.X[features] != self.na_fill)])-1)
                sampleX = self.X[(self.X[features] != self.na_fill)]\
                                .sample(sample_size)
            else:
                sampleX = self.X.sample(min(sample, len(self.X)))

        pdp_df = get_pdp_df(
                model=self.model, X_sample=sampleX,
                feature=features, n_grid_points=n_grid_points, 
                pos_label=pos_label, grid_values=grid_values)

        if all([str(c).startswith(col+"_") for c in pdp_df.columns]):
            pdp_df.columns = [str(c)[len(col)+1:] for c in pdp_df.columns]
        if self.is_classifier and self.model_output == 'probability':
            pdp_df = pdp_df.multiply(100)
        return pdp_df

    def get_dfs(self, cats=True, round=None, lang='en', pos_label=None):
        """return three summary dataframes for storing main results
        
        Returns three pd.DataFrames. The first with id, prediction, actual and
        feature values, the second with only id and shap values. The third
        is similar to contrib_df for every id.
        These can then be used to build your own custom dashboard on these data,
        for example using PowerBI.

        Args:
          cats(bool, optional, optional): group categorical variables, defaults to True
          round(int, optional, optional): how to round shap values (Default value = None)
          lang(str, optional, optional): language to format dfs in. Defaults to 'en', 'nl' also available
          pos_label:  (Default value = None)

        Returns:
          pd.DataFrame, pd.DataFrame, pd.DataFrame: cols_df, shap_df, contribs_df

        """
        if cats:
            cols_df = self.X_cats.copy()
            shap_df = pd.DataFrame(self.shap_values_cats(pos_label), columns = self.X_cats.columns)
        else:
            cols_df = self.X.copy()
            shap_df = pd.DataFrame(self.shap_values(pos_label), columns = self.X.columns)

        actual_str = 'Uitkomst' if lang == 'nl' else 'Actual'
        prediction_str = 'Voorspelling' if lang == 'nl' else 'Prediction'
        
        cols_df.insert(0, actual_str, self.y )
        if self.is_classifier:
            cols_df.insert(0, prediction_str, self.pred_probas)
        else:
            cols_df.insert(0, prediction_str, self.preds)
        cols_df.insert(0, 'name_id', self.idxs)
        cols_df.insert(0, 'row_id', range(len(self)))
 
        shap_df.insert(0, 'SHAP_base', np.repeat(self.shap_base_value, len(self)))
        shap_df.insert(0, 'name_id', self.idxs)
        shap_df.insert(0, 'row_id', range(len(self)))


        contribs_df = None
        for idx in range(len(self)):
            fcdf = self.formatted_contrib_df(idx, round=round, lang=lang)
            if contribs_df is None: contribs_df = fcdf
            else: contribs_df = pd.concat([contribs_df, fcdf])

        return cols_df, shap_df, contribs_df

    def to_sql(self, conn, schema, name, if_exists='replace',
                cats=True, round=None, lang='en', pos_label=None):
        """Writes three dataframes generated by .get_dfs() to a sql server.
        
        Tables will be called name_COLS and name_SHAP and name_CONTRBIB

        Args:
          conn(sqlalchemy.engine.Engine or sqlite3.Connection):     
                    database connecter acceptable for pd.to_sql
          schema(str): schema to write to
          name(str): name prefix of tables
          cats(bool, optional, optional): group categorical variables, defaults to True
          if_exists({'fail’, ‘replace’, ‘append’}, default ‘replace’, optional): 
                    How to behave if the table already exists. (Default value = 'replace')
          round(int, optional, optional): how to round shap values (Default value = None)
          lang(str, optional, optional): language to format dfs in. Defaults to 'en', 'nl' also available
          pos_label:  (Default value = None)

        Returns:

        """
        cols_df, shap_df, contribs_df = self.get_dfs(cats, round, lang, pos_label)
        cols_df.to_sql(con=conn, schema=schema, name=name+"_COLS",
                        if_exists=if_exists, index=False)
        shap_df.to_sql(con=conn, schema=schema, name=name+"_SHAP",
                        if_exists=if_exists, index=False)
        contribs_df.to_sql(con=conn, schema=schema, name=name+"_CONTRIB",
                        if_exists=if_exists, index=False)

    def plot_importances(self, kind='shap', topx=None, cats=False, round=3, pos_label=None):
        """plot barchart of importances in descending order.

        Args:
          type(str, optional): shap' for mean absolute shap values, 'permutation' for
                    permutation importances, defaults to 'shap'
          topx(int, optional, optional): Only return topx features, defaults to None
          cats(bool, optional, optional): Group categoricals defaults to False
          kind:  (Default value = 'shap')
          round:  (Default value = 3)
          pos_label:  (Default value = None)

        Returns:
          plotly.fig: fig

        """
        importances_df = self.importances_df(kind=kind, topx=topx, cats=cats, pos_label=pos_label)
        if kind=='shap':
            if self.target: 
                title = f"Average impact on predicted {self.target}<br>(mean absolute SHAP value)"
            else:
                title = 'Average impact on prediction<br>(mean absolute SHAP value)' 
                
            units = self.units
        else:
            title = f"Permutation Importances <br>(decrease in metric '{self.metric.__name__}'' with randomized feature)"
            units = ""
        if self.descriptions:
            descriptions = self.description_list(importances_df.Feature)
            return plotly_importances_plot(importances_df, descriptions, round=round, units=units, title=title)
        else:
            return plotly_importances_plot(importances_df, round=round, units=units, title=title)


    def plot_interactions(self, col, cats=False, topx=None, pos_label=None):
        """plot mean absolute shap interaction value for col.

        Args:
          col: column for which to generate shap interaction value
          cats(bool, optional, optional): Group categoricals defaults to False
          topx(int, optional, optional): Only return topx features, defaults to None
          pos_label:  (Default value = None)

        Returns:
          plotly.fig: fig

        """
        if col in self.onehot_cols:
            cats = True
        interactions_df = self.interactions_df(col, cats=cats, topx=topx, pos_label=pos_label)
        title = f"Average interaction shap values for {col}"
        return plotly_importances_plot(interactions_df, units=self.units, title=title)

    def plot_shap_contributions(self, index=None, X_row=None, cats=True, topx=None, cutoff=None, 
                        sort='abs', orientation='vertical', higher_is_better=True,
                        round=2, pos_label=None):
        """plot waterfall plot of shap value contributions to the model prediction for index.

        Args:
          index(int or str): index for which to display prediction
          X_row (pd.DataFrame single row): a single row of a features to plot
                shap contributions for. Can use this instead of index for
                what-if scenarios.
          cats(bool, optional, optional): Group categoricals, defaults to True
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
        assert orientation in ['vertical', 'horizontal']
        contrib_df = self.contrib_df(self.get_int_idx(index), X_row, cats, topx, cutoff, sort, pos_label)
        return plotly_contribution_plot(contrib_df, model_output=self.model_output, 
                    orientation=orientation, round=round, higher_is_better=higher_is_better,
                    target=self.target, units=self.units)

    def plot_shap_summary(self, index=None, topx=None, cats=False, pos_label=None):
        """Plot barchart of mean absolute shap value.
        
        Displays all individual shap value for each feature in a horizontal
        scatter chart in descending order by mean absolute shap value.

        Args:
          index (str or int): index to highlight
          topx(int, optional): Only display topx most important features, defaults to None
          cats(bool, optional): Group categoricals , defaults to False
          pos_label: positive class (Default value = None)

        Returns:
          plotly.Fig

        """
        if self.is_classifier:
            if pos_label is None:
                pos_label = self.pos_label
            pos_label_str = self.labels[self.get_pos_label_index(pos_label)]
            if self.model_output == 'probability':
                if self.target:
                    title = f"Impact of feature on predicted probability {self.target}={pos_label_str} <br> (SHAP values)"
                else:
                    title = f"Impact of Feature on Prediction probability <br> (SHAP values)"
            elif self.model_output == 'logodds':
                title = f"Impact of Feature on predicted logodds <br> (SHAP values)"
        elif self.is_regression:
            if self.target:
                title = f"Impact of Feature on Predicted {self.target} <br> (SHAP values)"
            else:
                title = f"Impact of Feature on Prediction<br> (SHAP values)"

        if cats:
            return plotly_shap_scatter_plot(
                                self.shap_values_cats(pos_label),
                                self.X_cats,
                                self.importances_df(kind='shap', topx=topx, cats=True, pos_label=pos_label)\
                                        ['Feature'].values.tolist(), 
                                idxs=self.idxs.values,
                                highlight_index=index,
                                title=title,
                                na_fill=self.na_fill,
                                index_name=self.index_name)
        else:
            return plotly_shap_scatter_plot(
                                self.shap_values(pos_label),
                                self.X,
                                self.importances_df(kind='shap', topx=topx, cats=False, pos_label=pos_label)\
                                        ['Feature'].values.tolist(), 
                                idxs=self.idxs.values,
                                highlight_index=index,
                                title=title,
                                na_fill=self.na_fill,
                                index_name=self.index_name)

    def plot_shap_interaction_summary(self, col, index=None, topx=None, cats=False, pos_label=None):
        """Plot barchart of mean absolute shap interaction values
        
        Displays all individual shap interaction values for each feature in a
        horizontal scatter chart in descending order by mean absolute shap value.

        Args:
          col(type]): feature for which to show interactions summary
          index (str or int): index to highlight
          topx(int, optional): only show topx most important features, defaults to None
          cats:  group categorical features (Default value = False)
          pos_label: positive class (Default value = None)

        Returns:
          fig
        """
        if col in self.onehot_cols:
            cats = True
        interact_cols = self.shap_top_interactions(col, cats=cats, pos_label=pos_label)
        if topx is None: topx = len(interact_cols)
        title = f"Shap interaction values for {col}"

        return plotly_shap_scatter_plot(
                self.shap_interaction_values_by_col(col, cats=cats, pos_label=pos_label),
                self.X_cats if cats else self.X, interact_cols[:topx], title=title, 
                idxs=self.idxs.values, highlight_index=index, na_fill=self.na_fill,
                index_name=self.index_name)

    def plot_shap_dependence(self, col, color_col=None, highlight_index=None, 
                                topx=None, sort='alphabet', pos_label=None):
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
          pos_label: positive class (Default value = None)

        Returns:

        """
        cats = self.check_cats(col, color_col)
        highlight_idx = self.get_int_idx(highlight_index)

        if cats:
            
            if col in self.onehot_cols or col in self.categorical_cols:
                return plotly_shap_violin_plot(
                            self.X_cats, 
                            self.shap_values_cats(pos_label), 
                            col, 
                            color_col, 
                            highlight_index=highlight_idx,
                            idxs=self.idxs.values,
                            index_name=self.index_name,
                            cats_order=self.ordered_cats(col, topx, sort))
            else:
                return plotly_dependence_plot(
                            self.X_cats, 
                            self.shap_values_cats(pos_label),
                            col, 
                            color_col, 
                            na_fill=self.na_fill, 
                            units=self.units, 
                            highlight_index=highlight_idx,
                            idxs=self.idxs.values,
                            index_name=self.index_name)
        else:
            if col in self.categorical_cols:
                return plotly_shap_violin_plot(
                                self.X_cats, 
                                self.shap_values_cats(pos_label), 
                                col, 
                                color_col, 
                                highlight_index=highlight_idx,
                                idxs=self.idxs.values,
                                index_name=self.index_name,
                                cats_order=self.ordered_cats(col, topx, sort))
            else:
                return plotly_dependence_plot(
                                self.X, 
                                self.shap_values(pos_label),
                                col, 
                                color_col, 
                                na_fill=self.na_fill, 
                                units=self.units, 
                                highlight_index=highlight_idx,
                                idxs=self.idxs.values,
                                index_name=self.index_name)

    def plot_shap_interaction(self, col, interact_col, highlight_index=None, 
                                topx=10, sort='alphabet', pos_label=None):
        """plots a dependence plot for shap interaction effects

        Args:
          col(str): feature for which to find interaction values
          interact_col(str): feature for which interaction value are displayed
          highlight_idx(int, optional, optional): idx that will be highlighted, defaults to None
          pos_label:  (Default value = None)

        Returns:
          plotly.Fig: Plotly Fig

        """
        cats = self.check_cats(col, interact_col)
        highlight_idx = self.get_int_idx(highlight_index)

        if cats and (interact_col in self.onehot_cols or interact_col in self.categorical_cols):
            return plotly_shap_violin_plot(
                self.X_cats, 
                self.shap_interaction_values_by_col(col, cats, pos_label=pos_label),
                interact_col, col, interaction=True, units=self.units, 
                highlight_index=highlight_idx, idxs=self.idxs.values,
                index_name=self.index_name, cats_order=self.ordered_cats(interact_col, topx, sort))
        else:
            return plotly_dependence_plot(self.X_cats if cats else self.X,
                self.shap_interaction_values_by_col(col, cats, pos_label=pos_label),
                interact_col, col, interaction=True, units=self.units,
                highlight_index=highlight_idx, idxs=self.idxs.values,
                index_name=self.index_name)

    def plot_pdp(self, col, index=None, X_row=None, drop_na=True, sample=100,
                    gridlines=100, gridpoints=10, sort='freq', round=2,
                    pos_label=None):
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
        pdp_df = self.pdp_df(col, index, X_row,
                        drop_na=drop_na, sample=sample, n_grid_points=gridpoints, 
                        pos_label=pos_label, sort=sort)
        units = "Predicted %" if self.model_output=='probability' else self.units
        if index is not None or X_row is not None:
            col_value, pred = self.get_col_value_plus_prediction(col, index=index, X_row=X_row, pos_label=pos_label)
            if (col in self.cat_cols 
                and col_value not in pdp_df.columns 
                and col_value[len(col)+1:] in pdp_df.columns):
                col_value = col_value[len(col)+1:]
            return plotly_pdp(pdp_df,
                            display_index=0, # the idx to be displayed is always set to the first row by self.pdp_df()
                            index_feature_value=col_value, 
                            index_prediction=pred,
                            feature_name=col,
                            num_grid_lines=min(gridlines, sample, len(self.X)),
                            round=round, target=self.target, units=units)
        else:
            return plotly_pdp(pdp_df, feature_name=col,
                        num_grid_lines=min(gridlines, sample, len(self.X)), 
                        round=round, target=self.target, units=units)


class ClassifierExplainer(BaseExplainer):
    """ """
    def __init__(self, model,  X, y=None,  permutation_metric=roc_auc_score, 
                    shap='guess', X_background=None, model_output="probability",
                    cats=None, idxs=None, index_name=None, target=None,
                    descriptions=None, n_jobs=None, permutation_cv=None, na_fill=-999,
                    labels=None, pos_label=1):
        """
        Explainer for classification models. Defines the shap values for
        each possible class in the classification.

        You assign the positive label class afterwards with e.g. explainer.pos_label=0

        In addition defines a number of plots specific to classification problems
        such as a precision plot, confusion matrix, roc auc curve and pr auc curve.

        Compared to BaseExplainer defines two additional parameters

        Args:
            labels(list): list of str labels for the different classes, 
                        defaults to e.g. ['0', '1'] for a binary classification
            pos_label: class that should be used as the positive class, 
                        defaults to 1
        """
        super().__init__(model, X, y, permutation_metric, 
                            shap, X_background, model_output, 
                            cats, idxs, index_name, target, descriptions, 
                            n_jobs, permutation_cv, na_fill)

        assert hasattr(model, "predict_proba"), \
                ("for ClassifierExplainer, model should be a scikit-learn "
                 "compatible *classifier* model that has a predict_proba(...) "
                 f"method, so not a {type(model)}!")

        self._params_dict = {**self._params_dict, **dict(
            labels=labels, pos_label=pos_label)}

        if self.categorical_cols and model_output == 'probability':
            print("Warning: Models that deal with categorical features directly "
                f"such as {self.model.__class__.__name__} are incompatible with model_output='probability'"
                " for now. So setting model_output='logodds'...", flush=True)
            self.model_output = 'logodds'
        if labels is not None:
            self.labels = labels
        elif hasattr(self.model, 'classes_'):
                self.labels =  [str(cls) for cls in self.model.classes_]
        else:
            self.labels = [str(i) for i in range(self.y.nunique())]
        self.pos_label = pos_label
        self.is_classifier = True
        if str(type(self.model)).endswith("RandomForestClassifier'>"):
            print(f"Detected RandomForestClassifier model: "
                    "Changing class type to RandomForestClassifierExplainer...", 
                    flush=True)
            self.__class__ = RandomForestClassifierExplainer 
        if str(type(self.model)).endswith("XGBClassifier'>"):
            print(f"Detected XGBClassifier model: "
                    "Changing class type to XGBClassifierExplainer...", 
                    flush=True)
            self.__class__ = XGBClassifierExplainer

        _ = self.shap_explainer

    @property
    def shap_explainer(self):
        """Initialize SHAP explainer. 
        
        Taking into account model type and model_output
        """
        if not hasattr(self, '_shap_explainer'):
            model_str = str(type(self.model)).replace("'", "").replace("<", "").replace(">", "").split(".")[-1]
            if self.shap == 'tree':
                if safe_is_instance(self.model, 
                    "XGBClassifier", "LGBMClassifier", "CatBoostClassifier", 
                    "GradientBoostingClassifier", "HistGradientBoostingClassifier"):
                    if self.model_output == "probability": 
                        if self.X_background is None:
                            print(
                                f"Note: model_output=='probability'. For {model_str} shap values normally get "
                                "calculated against X_background, but paramater X_background=None, "
                                "so using X instead")
                        print("Generating self.shap_explainer = shap.TreeExplainer(model, "
                             f"{'X_background' if self.X_background is not None else 'X'}"
                             ", model_output='probability', feature_perturbation='interventional')...")
                        print("Note: Shap interaction values will not be available. "
                              "If shap values in probability space are not necessary you can "
                              "pass model_output='logodds' to get shap values in logodds without the need for "
                              "a background dataset and also working shap interaction values...")
                        self._shap_explainer = shap.TreeExplainer(
                                                    self.model, 
                                                    self.X_background if self.X_background is not None else self.X,
                                                    model_output="probability",
                                                    feature_perturbation="interventional")
                        self.interactions_should_work = False
                    else:
                        self.model_output = "logodds"
                        print(f"Generating self.shap_explainer = shap.TreeExplainer(model{', X_background' if self.X_background is not None else ''})")
                        self._shap_explainer = shap.TreeExplainer(self.model, self.X_background)
                else:
                    if self.model_output == "probability":
                        print(f"Note: model_output=='probability', so assuming that raw shap output of {model_str} is in probability space...")
                    print(f"Generating self.shap_explainer = shap.TreeExplainer(model{', X_background' if self.X_background is not None else ''})")
                    self._shap_explainer = shap.TreeExplainer(self.model, self.X_background)


            elif self.shap=='linear':
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
                        "X_background, but paramater X_background=None, so using X instead...")
                print("Generating self.shap_explainer = shap.LinearExplainer(model, "
                             f"{'X_background' if self.X_background is not None else 'X'})...")
                
                self._shap_explainer = shap.LinearExplainer(self.model, 
                                            self.X_background if self.X_background is not None else self.X)
            elif self.shap=='deep':
                print("Generating self.shap_explainer = shap.DeepExplainer(model{', X_background' if self.X_background is not None else ''})")
                self._shap_explainer = shap.DeepExplainer(self.model, self.X_background)
            elif self.shap=='kernel': 
                if self.X_background is None:
                    print(
                        "Note: shap values for shap='kernel' normally get calculated against "
                        "X_background, but paramater X_background=None, so using X instead...")
                if self.model_output != "probability":
                    print(
                        "Note: for ClassifierExplainer shap='kernel' defaults to model_output='probability"
                    )
                    self.model_output = 'probability'
                print("Generating self.shap_explainer = shap.KernelExplainer(model, "
                             f"{'X_background' if self.X_background is not None else 'X'}"
                             ", link='identity')")
                self._shap_explainer = shap.KernelExplainer(self.model.predict_proba, 
                                            self.X_background if self.X_background is not None else self.X,
                                            link="identity")       
        return self._shap_explainer

    @property
    def pos_label(self):
        return self._pos_label

    @pos_label.setter
    def pos_label(self, label):
        if label is None or isinstance(label, int) and label >=0 and label <len(self.labels):
            self._pos_label = label
        elif isinstance(label, str) and label in self.labels:
            self._pos_label = self.labels.index(label)
        else:
            raise ValueError(f"'{label}' not in labels")

    @property
    def pos_label_str(self):
        """return str label of self.pos_label"""
        return self.labels[self.pos_label]

    def get_pos_label_index(self, pos_label):
        """return int index of pos_label_str"""
        if isinstance(pos_label, int):
            assert pos_label <= len(self.labels), \
                f"pos_label {pos_label} is larger than number of labels!"
            return pos_label
        elif isinstance(pos_label, str):
            assert pos_label in self.labels, \
                f"Unknown pos_label. {pos_label} not in self.labels!" 
            return self.labels.index(pos_label)
        raise ValueError("pos_label should either be int or str in self.labels!")

    def get_prop_for_label(self, prop:str, label):
        """return property for a specific pos_label

        Args:
          prop: property to get for a certain pos_label
          label: pos_label

        Returns:
            property
        """
        tmp = self.pos_label
        self.pos_label = label
        ret = getattr(self, prop)
        self.pos_label = tmp
        return ret

    @property
    def y_binary(self):
        """for multiclass problems returns one-vs-rest array of [1,0] pos_label"""
        if not hasattr(self, '_y_binaries'):
            if not self.y_missing:
                self._y_binaries = [np.where(self.y.values==i, 1, 0)
                            for i in range(self.y.nunique())]
            else:
                self._y_binaries = [self.y.values for i in range(len(self.labels))]
        return default_list(self._y_binaries, self.pos_label)

    @property
    def pred_probas_raw(self):
        """returns pred_probas with probability for each class"""
        if not hasattr(self, '_pred_probas'):
            print("Calculating prediction probabilities...", flush=True)
            assert hasattr(self.model, 'predict_proba'), \
                "model does not have a predict_proba method!"
            self._pred_probas =  self.model.predict_proba(self.X)
        return self._pred_probas

    @property
    def pred_percentiles_raw(self):
        """ """
        if not hasattr(self, '_pred_percentiles_raw'):
            print("Calculating pred_percentiles...", flush=True)
            self._pred_percentiles_raw = (pd.DataFrame(self.pred_probas_raw)
                                .rank(method='min')
                                .divide(len(self.pred_probas_raw))
                                .values)
        return self._pred_percentiles_raw

    @property
    def pred_probas(self):
        """returns pred_proba for pos_label class"""
        return default_2darray(self.pred_probas_raw, self.pos_label)

    @property
    def pred_percentiles(self):
        """returns ranks for pos_label class"""
        return default_2darray(self.pred_percentiles_raw, self.pos_label)

    @property
    def permutation_importances(self):
        """Permutation importances"""
        if not hasattr(self, '_perm_imps'):
            print("Calculating permutation importances (if slow, try setting n_jobs parameter)...", flush=True)
            self._perm_imps = [cv_permutation_importances(
                            self.model, self.X, self.y, self.metric,
                            cv=self.permutation_cv,
                            needs_proba=self.is_classifier,
                            pos_label=label) for label in range(len(self.labels))]
        return default_list(self._perm_imps, self.pos_label)

    @property
    def permutation_importances_cats(self):
        """permutation importances with categoricals grouped"""
        if not hasattr(self, '_perm_imps_cats'):
            print("Calculating categorical permutation importances (if slow, try setting n_jobs parameter)...", flush=True)
            if self.onehot_cols:
                self._perm_imps_cats = [cv_permutation_importances(
                                self.model, self.X, self.y, self.metric, 
                                onehot_dict=self.onehot_dict,
                                cv=self.permutation_cv,
                                needs_proba=self.is_classifier,
                                pos_label=label) for label in range(len(self.labels))]
            else:
                _ = self.permutation_importances
                self._perm_imps_cats = self._perm_imps
        return default_list(self._perm_imps_cats, self.pos_label)

    @property
    def shap_base_value(self):
        """SHAP base value: average outcome of population"""
        if not hasattr(self, '_shap_base_value'):
            _ = self.shap_values() # CatBoost needs to have shap values calculated before expected value for some reason
            self._shap_base_value = self.shap_explainer.expected_value
            if isinstance(self._shap_base_value, np.ndarray) and len(self._shap_base_value) == 1:
                self._shap_base_value = self._shap_base_value[0]
            if isinstance(self._shap_base_value, np.ndarray):
                self._shap_base_value = list(self._shap_base_value)
            if len(self.labels)==2 and isinstance(self._shap_base_value, (np.floating, float)):
                if self.model_output == 'probability':
                    self._shap_base_value = [1-self._shap_base_value, self._shap_base_value]
                else: # assume logodds
                    self._shap_base_value = [-self._shap_base_value, self._shap_base_value]
            assert len(self._shap_base_value)==len(self.labels),\
                f"len(shap_explainer.expected_value)={len(self._shap_base_value)}"\
                 + f"and len(labels)={len(self.labels)} do not match!"
            if self.model_output == 'probability':
                for shap_base_value in self._shap_base_value:
                    assert shap_base_value >= 0.0 and shap_base_value <= 1.0, \
                        (f"Shap base value does not look like a probability: {self._shap_base_value}. "
                         "Try setting model_output='logodds'.")
        return default_list(self._shap_base_value, self.pos_label)

    @property
    def shap_values(self):
        """SHAP Values"""
        if not hasattr(self, '_shap_values'):
            print("Calculating shap values...", flush=True)
            self._shap_values = self.shap_explainer.shap_values(self.X)
            
            if not isinstance(self._shap_values, list) and len(self.labels)==2:
                    self._shap_values = [-self._shap_values, self._shap_values]

            assert len(self._shap_values)==len(self.labels),\
                f"len(shap_values)={len(self._shap_values)}"\
                + f"and len(labels)={len(self.labels)} do not match!"
            if self.model_output == 'probability':
                for shap_values in self._shap_values:
                    assert np.all(shap_values >= -1.0) , \
                        (f"model_output=='probability but some shap values are < 1.0!"
                         "Try setting model_output='logodds'.")
                for shap_values in self._shap_values:
                    assert np.all(shap_values <= 1.0) , \
                        (f"model_output=='probability but some shap values are > 1.0!"
                         "Try setting model_output='logodds'.")
        return default_list(self._shap_values, self.pos_label)

    @property
    def shap_values_cats(self):
        """SHAP values with categoricals grouped together"""
        if not hasattr(self, '_shap_values_cats'):
            _ = self.shap_values
            self._shap_values_cats = [
                    merge_categorical_shap_values(
                        self.X, sv, self.onehot_dict) for sv in self._shap_values]
        return default_list(self._shap_values_cats, self.pos_label)


    @property
    def shap_interaction_values(self):
        """SHAP interaction values"""
        if not hasattr(self, '_shap_interaction_values'):
            _ = self.shap_values #make sure shap values have been calculated
            print("Calculating shap interaction values...", flush=True)
            if self.shap == 'tree':
                print("Reminder: TreeShap computational complexity is O(TLD^2), "
                    "where T is the number of trees, L is the maximum number of"
                    " leaves in any tree and D the maximal depth of any tree. So "
                    "reducing these will speed up the calculation.", 
                    flush=True)
            self._shap_interaction_values = self.shap_explainer.shap_interaction_values(self.X)
            
            if not isinstance(self._shap_interaction_values, list) and len(self.labels)==2:
                if self.model_output == "probability":
                    self._shap_interaction_values = [1-self._shap_interaction_values,
                                                        self._shap_interaction_values]
                else: # assume logodds so logodds of negative class is -logodds of positive class
                    self._shap_interaction_values = [-self._shap_interaction_values,
                                                        self._shap_interaction_values]

            self._shap_interaction_values = [
                normalize_shap_interaction_values(siv, self.shap_values)
                    for siv, sv in zip(self._shap_interaction_values, self._shap_values)]
        return default_list(self._shap_interaction_values, self.pos_label)

    @property
    def shap_interaction_values_cats(self):
        """SHAP interaction values with categoricals grouped together"""
        if not hasattr(self, '_shap_interaction_values_cats'):
            _ = self.shap_interaction_values
            self._shap_interaction_values_cats = [
                merge_categorical_shap_interaction_values(
                    siv, self.X, self.X_cats, self.onehot_dict) 
                        for siv in self._shap_interaction_values]
        return default_list(self._shap_interaction_values_cats, self.pos_label)

    @property
    def mean_abs_shap(self):
        """mean absolute SHAP values"""
        if not hasattr(self, '_mean_abs_shap'):
            _ = self.shap_values
            self._mean_abs_shap = [mean_absolute_shap_values(
                                self.columns, sv) for sv in self._shap_values]
        return default_list(self._mean_abs_shap, self.pos_label)

    @property
    def mean_abs_shap_cats(self):
        """mean absolute SHAP values with categoricals grouped together"""
        if not hasattr(self, '_mean_abs_shap_cats'):
            _ = self.shap_values_cats
            self._mean_abs_shap_cats = [
                mean_absolute_shap_values(self.columns_cats, sv) 
                    for sv in self._shap_values_cats]
        return default_list(self._mean_abs_shap_cats, self.pos_label)

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
        if pos_label is None:
            return pd.Series(self.pred_probas).nlargest(int((1-percentile)*len(self))).min()
        else:
            return pd.Series(self.pred_probas_raw[:, pos_label]).nlargest(int((1-percentile)*len(self))).min()

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
        if pos_label is None:
            return 1-(self.pred_probas < cutoff).mean()
        else:
            pos_label = self.get_pos_label_index(pos_label)
            return 1-np.mean(self.pred_probas_raw[:, pos_label] < cutoff)

    def metrics(self, cutoff=0.5, pos_label=None, **kwargs):
        """returns a dict with useful metrics for your classifier:
        
        accuracy, precision, recall, f1, roc auc, pr auc, log loss

        Args:
          cutoff(float): cutoff used to calculate metrics (Default value = 0.5)
          pos_label: positive class (Default value = None)

        Returns:
          dict

        """
        if self.y_missing:
            raise ValueError("No y was passed to explainer, so cannot calculate metrics!")

        if pos_label is None: pos_label = self.pos_label
        metrics_dict = {
            'accuracy' : accuracy_score(self.y_binary(pos_label), np.where(self.pred_probas(pos_label) > cutoff, 1, 0)),
            'precision' : precision_score(self.y_binary(pos_label), np.where(self.pred_probas(pos_label) > cutoff, 1, 0)),
            'recall' : recall_score(self.y_binary(pos_label), np.where(self.pred_probas(pos_label) > cutoff, 1, 0)),
            'f1' : f1_score(self.y_binary(pos_label), np.where(self.pred_probas(pos_label) > cutoff, 1, 0)),
            'roc_auc_score' : roc_auc_score(self.y_binary(pos_label), self.pred_probas(pos_label)),
            'pr_auc_score' : average_precision_score(self.y_binary(pos_label), self.pred_probas(pos_label)),
            'log_loss' : log_loss(self.y_binary(pos_label), self.pred_probas(pos_label))
        }
        return metrics_dict

    def metrics_descriptions(self, cutoff=0.5, pos_label=None):
        metrics_dict = self.metrics(cutoff, pos_label)
        metrics_descriptions_dict = {}
        for k, v in metrics_dict.items():
            if k == 'accuracy':
                metrics_descriptions_dict[k] = f"{round(100*v, 2)}% of predicted labels was predicted correctly."
            if k == 'precision':
                metrics_descriptions_dict[k] = f"{round(100*v, 2)}% of predicted positive labels was predicted correctly."
            if k == 'recall':
                metrics_descriptions_dict[k] = f"{round(100*v, 2)}% of positive labels was predicted correctly."
            if k == 'f1':
                metrics_descriptions_dict[k] = f"The weighted average of precision and recall is {round(v, 2)}"
            if k == 'roc_auc_score':
                metrics_descriptions_dict[k] = f"The probability that a random positive label has a higher score than a random negative label is {round(100*v, 2)}%"
            if k == 'pr_auc_score':
                metrics_descriptions_dict[k] = f"The average precision score calculated for each recall threshold is {round(v, 2)}. This ignores true negatives."
            if k == 'log_loss':
                metrics_descriptions_dict[k] = f"A measure of how far the predicted label is from the true label on average in log space {round(v, 2)}"
        return metrics_descriptions_dict


    def random_index(self, y_values=None, return_str=False,
                    pred_proba_min=None, pred_proba_max=None,
                    pred_percentile_min=None, pred_percentile_max=None, pos_label=None):
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
        if (y_values is None 
            and pred_proba_min is None and pred_proba_max is None
            and pred_percentile_min is None and pred_percentile_max is None):
            potential_idxs = self.idxs.values
        else:
            if pred_proba_min is None: pred_proba_min = self.pred_probas(pos_label).min()
            if pred_proba_max is None: pred_proba_max = self.pred_probas(pos_label).max()
            if pred_percentile_min is None: pred_percentile_min = 0.0
            if pred_percentile_max is None: pred_percentile_max = 1.0
            
            if not self.y_missing:
                if y_values is None: y_values = self.y.unique().tolist()
                if not isinstance(y_values, list): y_values = [y_values]
                y_values = [y if isinstance(y, int) else self.labels.index(y) for y in y_values]

                potential_idxs = self.idxs[(self.y.isin(y_values)) &
                                (self.pred_probas(pos_label) >= pred_proba_min) &
                                (self.pred_probas(pos_label) <= pred_proba_max) &
                                (self.pred_percentiles(pos_label) > pred_percentile_min) &
                                (self.pred_percentiles(pos_label) <= pred_percentile_max)].values
            
            else:
                potential_idxs = self.idxs[
                                (self.pred_probas(pos_label) >= pred_proba_min) &
                                (self.pred_probas(pos_label) <= pred_proba_max) &
                                (self.pred_percentiles(pos_label) > pred_percentile_min) &
                                (self.pred_percentiles(pos_label) <= pred_percentile_max)].values

        if len(potential_idxs) > 0:
            idx = np.random.choice(potential_idxs)
        else:
            return None
        if return_str:
            return idx
        return self.idxs.get_loc(idx)

    def prediction_result_df(self, index=None, X_row=None, add_star=True, logodds=False, round=3):
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
            int_idx = self.get_int_idx(index)
            pred_probas = self.pred_probas_raw[int_idx, :]
        elif X_row is not None:
            if X_row.columns.tolist()==self.X_cats.columns.tolist():
                X_row = X_cats_to_X(X_row, self.onehot_dict, self.X.columns)  
            pred_probas = self.model.predict_proba(X_row)[0, :]

        preds_df =  pd.DataFrame(dict(
            label=self.labels, 
            probability=pred_probas))
        if logodds:
            preds_df.loc[:, "logodds"] = \
                preds_df.probability.apply(lambda p: np.log(p / (1-p)))
        if index is not None and not self.y_missing and not np.isnan(self.y[int_idx]):
            preds_df.iloc[self.y[int_idx], 0] = f"{preds_df.iloc[self.y[int_idx], 0]}*"
        return preds_df.round(round)

    def precision_df(self, bin_size=None, quantiles=None, multiclass=False, 
                        round=3, pos_label=None):
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
            raise ValueError("No y was passed to explainer, so cannot calculate precision_df!")
        assert self.pred_probas is not None

        if pos_label is None: pos_label = self.pos_label

        if bin_size is None and quantiles is None:
            bin_size=0.1 # defaults to bin_size=0.1
        if multiclass:
            return get_precision_df(self.pred_probas_raw, self.y,
                                bin_size, quantiles, 
                                round=round, pos_label=pos_label)
        else:
            return get_precision_df(self.pred_probas(pos_label), self.y_binary(pos_label), 
                                    bin_size, quantiles, round=round)

    def lift_curve_df(self, pos_label=None):
        """returns a pd.DataFrame with data needed to build a lift curve

        Args:
          pos_label:  (Default value = None)

        Returns:

        """
        if pos_label is None: pos_label = self.pos_label
        return get_lift_curve_df(self.pred_probas(pos_label), self.y, pos_label)

    def prediction_result_markdown(self, index, include_percentile=True, round=2, pos_label=None):
        """markdown of result of prediction for index

        Args:
          index(int or str): the index of the row for which to generate the prediction
          include_percentile(bool, optional, optional): include the rank 
                    percentile of the prediction, defaults to True
          round(int, optional, optional): rounding to apply to results, defaults to 2
          pos_label:  (Default value = None)
          **kwargs: 

        Returns:
          str: markdown string

        """
        int_idx = self.get_int_idx(index)
        if pos_label is None: pos_label = self.pos_label
        
        def display_probas(pred_probas_raw, labels, model_output='probability', round=2):
            assert (len(pred_probas_raw.shape)==1 and len(pred_probas_raw) ==len(labels))
            def log_odds(p, round=2):
                return np.round(np.log(p / (1-p)), round)
            for i in range(len(labels)):
                proba_str = f"{np.round(100*pred_probas_raw[i], round)}%"
                logodds_str = f"(logodds={log_odds(pred_probas_raw[i], round)})"
                yield f"* {labels[i]}: {proba_str} {logodds_str if model_output=='logodds' else ''}\n"

        model_prediction = "###### Prediction:\n\n"
        if (isinstance(self.y[0], int) or 
            isinstance(self.y[0], np.int64)):
            model_prediction += f"Observed {self.target}: {self.labels[self.y[int_idx]]}\n\n"
        
        model_prediction += "Prediction probabilities per label:\n\n" 
        for pred in display_probas(
                self.pred_probas_raw[int_idx], 
                self.labels, self.model_output, round):
            model_prediction += pred
        
        if include_percentile:
            percentile = np.round(100*(1-self.pred_percentiles(pos_label)[int_idx]))
            model_prediction += f'\nIn top {percentile}% percentile probability {self.labels[pos_label]}'      
        return model_prediction


    def plot_precision(self, bin_size=None, quantiles=None, cutoff=None, multiclass=False, pos_label=None):
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
        if pos_label is None: pos_label = self.pos_label
        if bin_size is None and quantiles is None:
            bin_size=0.1 # defaults to bin_size=0.1
        precision_df = self.precision_df(
                bin_size=bin_size, quantiles=quantiles, multiclass=multiclass, pos_label=pos_label)
        return plotly_precision_plot(precision_df,
                    cutoff=cutoff, labels=self.labels, pos_label=pos_label)

    def plot_cumulative_precision(self, percentile=None, pos_label=None):
        """plot cumulative precision
        
        returns a cumulative precision plot, which is a slightly different
        representation of a lift curve.

        Args:
          pos_label: positive label to display, defaults to self.pos_label

        Returns:
          plotly fig

        """
        if pos_label is None: pos_label = self.pos_label
        return plotly_cumulative_precision_plot(
                    self.lift_curve_df(pos_label=pos_label), labels=self.labels, 
                    percentile=percentile, pos_label=pos_label)

    def plot_confusion_matrix(self, cutoff=0.5, normalized=False, binary=False, pos_label=None):
        """plot of a confusion matrix.

        Args:
          cutoff(float, optional, optional): cutoff of positive class to 
                    calculate confusion matrix for, defaults to 0.5
          normalized(bool, optional, optional): display percentages instead 
                    of counts , defaults to False
          binary(bool, optional, optional): if multiclass display one-vs-rest 
                    instead, defaults to False
          pos_label: positive label to display, defaults to self.pos_label

        Returns:
          plotly fig

        """
        if self.y_missing:
            raise ValueError("No y was passed to explainer, so cannot plot confusion matrix!")
        if pos_label is None: pos_label = self.pos_label
        pos_label_str = self.labels[pos_label]

        if binary:
            if len(self.labels)==2:
                def order_binary_labels(labels, pos_label):
                    pos_index = labels.index(pos_label)
                    return [labels[1-pos_index], labels[pos_index]]
                labels = order_binary_labels(self.labels, pos_label_str)
            else:
                labels = ['Not ' + pos_label_str, pos_label_str]

            return plotly_confusion_matrix(
                    self.y_binary(pos_label), np.where(self.pred_probas(pos_label) > cutoff, 1, 0),
                    percentage=normalized, labels=labels)
        else:
            return plotly_confusion_matrix(
                self.y, self.pred_probas_raw.argmax(axis=1),
                percentage=normalized, labels=self.labels)

    def plot_lift_curve(self, cutoff=None, percentage=False, add_wizard=True, 
                        round=2, pos_label=None):
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
        return plotly_lift_curve(self.lift_curve_df(pos_label), cutoff=cutoff, 
                percentage=percentage, add_wizard=add_wizard, round=round)

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
        return plotly_classification_plot(self.pred_probas(pos_label), self.y, self.labels, cutoff, percentage=percentage)

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
        return plotly_roc_auc_curve(self.y_binary(pos_label), self.pred_probas(pos_label), cutoff=cutoff)

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
        return plotly_pr_auc_curve(self.y_binary(pos_label), self.pred_probas(pos_label), cutoff=cutoff)

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
        _ = self.pred_probas
        super().calculate_properties(include_interactions=include_interactions)


class RegressionExplainer(BaseExplainer):
    """ """
    def __init__(self, model,  X, y=None, permutation_metric=r2_score, 
                    shap="guess", X_background=None, model_output="raw",
                    cats=None, idxs=None, index_name=None, target=None,
                    descriptions=None, n_jobs=None, permutation_cv=None, 
                    na_fill=-999, units=""):
        """Explainer for regression models.

        In addition to BaseExplainer defines a number of plots specific to 
        regression problems such as a predicted vs actual and residual plots.

        Combared to BaseExplainerBunch defines two additional parameters.

        Args:
          units(str): units to display for regression quantity
        """
        super().__init__(model, X, y, permutation_metric, 
                            shap, X_background, model_output,
                            cats, idxs, index_name, target, descriptions,
                            n_jobs, permutation_cv, na_fill)

        self._params_dict = {**self._params_dict, **dict(units=units)}
        self.units = units
        self.is_regression = True

        if str(type(self.model)).endswith("RandomForestRegressor'>"):
            print(f"Changing class type to RandomForestRegressionExplainer...", flush=True)
            self.__class__ = RandomForestRegressionExplainer 
        if str(type(self.model)).endswith("XGBRegressor'>"):
            print(f"Changing class type to XGBRegressionExplainer...", flush=True)
            self.__class__ = XGBRegressionExplainer

        _ = self.shap_explainer
    
    @property
    def residuals(self):
        """residuals: y-preds"""
        if not hasattr(self, '_residuals'):
            print("Calculating residuals...")
            self._residuals =  self.y-self.preds
        return self._residuals

    @property
    def abs_residuals(self):
        """absolute residuals"""
        if not hasattr(self, '_abs_residuals'):
            print("Calculating absolute residuals...")
            self._abs_residuals =  np.abs(self.residuals)
        return self._abs_residuals

    def random_index(self, y_min=None, y_max=None, pred_min=None, pred_max=None, 
                        residuals_min=None, residuals_max=None,
                        abs_residuals_min=None, abs_residuals_max=None,
                        return_str=False, **kwargs):
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
                                (self.preds >= pred_min) & 
                                (self.preds <= pred_max)].values
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

            potential_idxs = self.idxs[(self.y >= y_min) & 
                                    (self.y <= y_max) & 
                                    (self.preds >= pred_min) & 
                                    (self.preds <= pred_max) &
                                    (self.residuals >= residuals_min) & 
                                    (self.residuals <= residuals_max) &
                                    (self.abs_residuals >= abs_residuals_min) & 
                                    (self.abs_residuals <= abs_residuals_max)].values

        if len(potential_idxs) > 0:
            idx = np.random.choice(potential_idxs)
        else:
            return None
        if return_str:
            return idx
        return self.idxs.get_loc(idx)

    def prediction_result_markdown(self, index, include_percentile=True, round=2):
        """markdown of prediction result

        Args:
          index: row index to be predicted
          include_percentile (bool): include line about prediciton percentile
          round:  (Default value = 2)

        Returns:
          str: markdown summary of prediction for index 

        """
        int_idx = self.get_int_idx(index)
        model_prediction = "###### Prediction:\n"
        model_prediction += f"Predicted {self.target}: {np.round(self.preds[int_idx], round)} {self.units}\n\n"
        if not self.y_missing:
            model_prediction += f"Observed {self.target}: {np.round(self.y[int_idx], round)} {self.units}\n\n"
            model_prediction += f"Residual: {np.round(self.residuals[int_idx], round)} {self.units}\n\n"
        if include_percentile:
            percentile = np.round(100*(1-self.pred_percentiles[int_idx]))
            model_prediction += f"\nIn top {percentile}% percentile predicted {self.target}"
        return model_prediction

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
            int_idx = self.get_int_idx(index)
            preds_df = pd.DataFrame(columns = ["", self.target])
            preds_df = preds_df.append(
                pd.Series(("Predicted", str(np.round(self.preds[int_idx], round)) + f" {self.units}"), 
                        index=preds_df.columns), ignore_index=True)
            if not self.y_missing:
                preds_df = preds_df.append(
                    pd.Series(("Observed", str(np.round(self.y[int_idx], round)) + f" {self.units}"), 
                        index=preds_df.columns), ignore_index=True)
                preds_df = preds_df.append(
                    pd.Series(("Residual", str(np.round(self.residuals[int_idx], round)) + f" {self.units}"), 
                        index=preds_df.columns), ignore_index=True)

        elif X_row is not None:
            if X_row.columns.tolist()==self.X_cats.columns.tolist():
                X_row = X_cats_to_X(X_row, self.onehot_dict, self.X.columns) 
            assert np.all(X_row.columns==self.X.columns), \
                ("The column names of X_row should match X! Instead X_row.columns"
                 f"={X_row.columns.tolist()}...")
            prediction = self.model.predict(X_row)[0]
            preds_df = pd.DataFrame(columns = ["", self.target])
            preds_df = preds_df.append(
                pd.Series(("Predicted", str(np.round(prediction, round)) + f" {self.units}"), 
                        index=preds_df.columns), ignore_index=True)
        return preds_df

    def metrics(self):
        """dict of performance metrics: rmse, mae and R^2"""

        if self.y_missing:
            raise ValueError("No y was passed to explainer, so cannot calculate metrics!")
        metrics_dict = {
            'root_mean_squared_error' : np.sqrt(mean_squared_error(self.y, self.preds)),
            'mean_absolute_error' : mean_absolute_error(self.y, self.preds),
            'R-squared' : r2_score(self.y, self.preds),
        }
        return metrics_dict

    def metrics_descriptions(self):
        metrics_dict = self.metrics()
        metrics_descriptions_dict = {}
        for k, v in metrics_dict.items():
            if k == 'root_mean_squared_error':
                metrics_descriptions_dict[k] = f"A measure of how close predicted value fits true values, where large deviations are punished more heavily. So the lower this number the better the model."
            if k == 'mean_absolute_error':
                metrics_descriptions_dict[k] = f"On average predictions deviate {round(v, 2)} {self.units} off the observed value of {self.target} (can be both above or below)"
            if k == 'R-squared':
                metrics_descriptions_dict[k] = f"{round(100*v, 2)}% of all variation in {self.target} was explained by the model."
        return metrics_descriptions_dict

    def plot_predicted_vs_actual(self, round=2, logs=False, log_x=False, log_y=False, **kwargs):
        """plot with predicted value on x-axis and actual value on y axis.

        Args:
          round(int, optional): rounding to apply to outcome, defaults to 2
          logs (bool, optional): log both x and y axis, defaults to False
          log_y (bool, optional): only log x axis. Defaults to False.
          log_x (bool, optional): only log y axis. Defaults to False.
          **kwargs: 

        Returns:
          Plotly fig

        """
        if self.y_missing:
            raise ValueError("No y was passed to explainer, so cannot plot predicted vs actual!")
        return plotly_predicted_vs_actual(self.y, self.preds, 
                target=self.target, units=self.units, idxs=self.idxs.values, 
                logs=logs, log_x=log_x, log_y=log_y, round=round, 
                index_name=self.index_name)
    
    def plot_residuals(self, vs_actual=False, round=2, residuals='difference'):
        """plot of residuals. x-axis is the predicted outcome by default

        Args:
          vs_actual(bool, optional): use actual value for x-axis,   
                    defaults to False
          round(int, optional): rounding to perform on values, defaults to 2
          residuals (str, {'difference', 'ratio', 'log-ratio'} optional): 
                    How to calcualte residuals. Defaults to 'difference'.
        Returns:
          Plotly fig

        """
        if self.y_missing:
            raise ValueError("No y was passed to explainer, so cannot plot residuals!")
        return plotly_plot_residuals(self.y, self.preds, idxs=self.idxs.values,
                                     vs_actual=vs_actual, target=self.target, 
                                     units=self.units, residuals=residuals, 
                                     round=round, index_name=self.index_name)
    
    def plot_residuals_vs_feature(self, col, residuals='difference', round=2, 
                dropna=True, points=True, winsor=0):
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

        Returns:
          plotly fig
        """
        if self.y_missing:
            raise ValueError("No y was passed to explainer, so cannot plot residuals!")
        assert col in self.columns or col in self.columns_cats, \
            f'{col} not in columns or columns_cats!'
        col_vals = self.X_cats[col] if self.check_cats(col) else self.X[col]
        na_mask = col_vals != self.na_fill if dropna else np.array([True]*len(col_vals))
        return plotly_residuals_vs_col(
            self.y[na_mask], self.preds[na_mask], col_vals[na_mask], 
            residuals=residuals, idxs=self.idxs.values[na_mask], points=points, 
            round=round, winsor=winsor, index_name=self.index_name)

    def plot_y_vs_feature(self, col, residuals='difference', round=2, 
                dropna=True, points=True, winsor=0):
        """Plot y vs individual features

        Args:
          col(str): Plot against feature col
          round(int, optional): rounding to perform on residuals, defaults to 2
          dropna(bool, optional): drop missing values from plot, defaults to True.
          points (bool, optional): display point cloud next to violin plot. 
                    Defaults to True.
          winsor (int, 0-50, optional): percentage of outliers to winsor out of 
                    the y-axis. Defaults to 0.

        Returns:
          plotly fig
        """
        if self.y_missing:
            raise ValueError("No y was passed to explainer, so cannot plot y vs feature!")
        assert col in self.columns or col in self.columns_cats, \
            f'{col} not in columns or columns_cats!'
        col_vals = self.X_cats[col] if self.check_cats(col) else self.X[col]
        na_mask = col_vals != self.na_fill if dropna else np.array([True]*len(col_vals))
        return plotly_actual_vs_col(self.y[na_mask], self.preds[na_mask], col_vals[na_mask], 
                idxs=self.idxs.values[na_mask], points=points, round=round, winsor=winsor,
                units=self.units, target=self.target, index_name=self.index_name)

    def plot_preds_vs_feature(self, col, residuals='difference', round=2, 
                dropna=True, points=True, winsor=0):
        """Plot y vs individual features

        Args:
          col(str): Plot against feature col
          round(int, optional): rounding to perform on residuals, defaults to 2
          dropna(bool, optional): drop missing values from plot, defaults to True.
          points (bool, optional): display point cloud next to violin plot. 
                    Defaults to True.
          winsor (int, 0-50, optional): percentage of outliers to winsor out of 
                    the y-axis. Defaults to 0.

        Returns:
          plotly fig
        """
        assert col in self.columns or col in self.columns_cats, \
            f'{col} not in columns or columns_cats!'
        col_vals = self.X_cats[col] if self.check_cats(col) else self.X[col]
        na_mask = col_vals != self.na_fill if dropna else np.array([True]*len(col_vals))
        return plotly_preds_vs_col(self.y[na_mask], self.preds[na_mask], col_vals[na_mask], 
                idxs=self.idxs.values[na_mask], points=points, round=round, winsor=winsor,
                units=self.units, target=self.target, index_name=self.index_name)


class RandomForestExplainer(BaseExplainer):
    """RandomForestBunch allows for the analysis of individual DecisionTrees that
    make up the RandomForest.

    """
    
    @property
    def is_tree_explainer(self):
        """this is either a RandomForestExplainer or XGBExplainer"""
        return True

    @property
    def no_of_trees(self):
        """The number of trees in the RandomForest model"""
        return len(self.model.estimators_)
        
    @property
    def graphviz_available(self):
        """ """
        if not hasattr(self, '_graphviz_available'):
            try:
                import graphviz.backend as be
                cmd = ["dot", "-V"]
                stdout, stderr = be.run(cmd, capture_output=True, check=True, quiet=True)
            except:
                print("""
                WARNING: you don't seem to have graphviz in your path (cannot run 'dot -V'), 
                so no dtreeviz visualisation of decision trees will be shown on the shadow trees tab.

                See https://github.com/parrt/dtreeviz for info on how to properly install graphviz 
                for dtreeviz. 
                """)
                self._graphviz_available = False
            else:
                self._graphviz_available = True
        return self._graphviz_available

    @property
    def decision_trees(self):
        """a list of ShadowDecTree objects"""
        if not hasattr(self, '_decision_trees'):
            print("Calculating ShadowDecTree for each individual decision tree...", flush=True)
            assert hasattr(self.model, 'estimators_'), \
                """self.model does not have an estimators_ attribute, so probably not
                actually a sklearn RandomForest?"""
                
            self._decision_trees = [
                ShadowDecTree.get_shadow_tree(decision_tree,
                                        self.X,
                                        self.y,
                                        feature_names=self.X.columns.tolist(),
                                        target_name='target',
                                        class_names = self.labels if self.is_classifier else None)
                            for decision_tree in self.model.estimators_]
        return self._decision_trees

    def decisiontree_df(self, tree_idx, index, pos_label=None):
        """dataframe with all decision nodes of a particular decision tree

        Args:
          tree_idx: the n'th tree in the random forest
          index: row index
          round:  (Default value = 2)
          pos_label:  positive class (Default value = None)

        Returns:
          dataframe with summary of the decision tree path

        """
        assert tree_idx >= 0 and tree_idx < len(self.decision_trees), \
            f"tree index {tree_idx} outside 0 and number of trees ({len(self.decision_trees)}) range"
        idx = self.get_int_idx(index)
        assert idx >= 0 and idx < len(self.X), \
            f"=index {idx} outside 0 and size of X ({len(self.X)}) range"
        
        if self.is_classifier:
            if pos_label is None: pos_label = self.pos_label
            return get_decisiontree_df(self.decision_trees[tree_idx], self.X.iloc[idx],
                    pos_label=pos_label)
        else:
            return get_decisiontree_df(self.decision_trees[tree_idx], self.X.iloc[idx])

    def decisiontree_summary_df(self, tree_idx, index, round=2, pos_label=None):
        """formats decisiontree_df in a slightly more human readable format.

        Args:
          tree_idx: the n'th tree in the random forest
          index: row index
          round:  (Default value = 2)
          pos_label:  positive class (Default value = None)

        Returns:
          dataframe with summary of the decision tree path

        """
        idx=self.get_int_idx(index)
        return get_decisiontree_summary_df(self.decisiontree_df(tree_idx, idx, pos_label=pos_label),
                    classifier=self.is_classifier, round=round, units=self.units)

    def decision_path_file(self, tree_idx, index, show_just_path=False):
        """get a dtreeviz visualization of a particular tree in the random forest.

        Args:
          tree_idx: the n'th tree in the random forest
          index: row index
          show_just_path (bool, optional): show only the path not rest of the 
                    tree. Defaults to False. 

        Returns:
          the path where the .svg file is stored.

        """
        if not self.graphviz_available:
            print("No graphviz 'dot' executable available!") 
            return None

        idx = self.get_int_idx(index)
        viz = dtreeviz(self.decision_trees[tree_idx], 
                        X=self.X.iloc[idx, :], 
                        fancy=False,
                        show_node_labels = False,
                        show_just_path=show_just_path) 
        return viz.save_svg()

    def decision_path(self, tree_idx, index, show_just_path=False):
        """get a dtreeviz visualization of a particular tree in the random forest.

        Args:
          tree_idx: the n'th tree in the random forest
          index: row index
          show_just_path (bool, optional): show only the path not rest of the 
                    tree. Defaults to False. 

        Returns:
          a IPython display SVG object for e.g. jupyter notebook.

        """
        if not self.graphviz_available:
            print("No graphviz 'dot' executable available!") 
            return None

        from IPython.display import SVG
        svg_file = self.decision_path_file(tree_idx, index, show_just_path)
        return SVG(open(svg_file,'rb').read())

    def decision_path_encoded(self, tree_idx, index, show_just_path=False):
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
        
        svg_file = self.decision_path_file(tree_idx, index, show_just_path)
        encoded = base64.b64encode(open(svg_file,'rb').read()) 
        svg_encoded = 'data:image/svg+xml;base64,{}'.format(encoded.decode()) 
        return svg_encoded


    def plot_trees(self, index, highlight_tree=None, round=2, 
                higher_is_better=True, pos_label=None):
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
        idx=self.get_int_idx(index)
        assert idx is not None, 'invalid index'
        
        if self.is_classifier:
            if pos_label is None: pos_label = self.pos_label
            if not np.isnan(self.y[idx]):
                y = 100*self.y_binary(pos_label)[idx] 
            else:
                y = None

            return plotly_rf_trees(self.model, self.X.iloc[[idx]], y,
                        highlight_tree=highlight_tree, round=round, 
                        pos_label=pos_label, target=self.target)
        else:
            y = self.y[idx]
            return plotly_rf_trees(self.model, self.X.iloc[[idx]], y,
                        highlight_tree=highlight_tree, round=round, 
                        target=self.target, units=self.units)

    def calculate_properties(self, include_interactions=True):
        """

        Args:
          include_interactions:  If False do not calculate shap interaction value
            (Default value = True)

        Returns:

        """
        _ = self.decision_trees
        super().calculate_properties(include_interactions=include_interactions)


class XGBExplainer(BaseExplainer):
    """XGBExplainer allows for the analysis of individual DecisionTrees that
    make up the xgboost model.
    """

    @property
    def is_tree_explainer(self):
        """this is either a RandomForestExplainer or XGBExplainer"""
        return True

    @property
    def model_dump_list(self):
        if not hasattr(self, "_model_dump_list"):
            print("Generating model dump...", flush=True)
            self._model_dump_list = self.model.get_booster().get_dump()
        return self._model_dump_list

    @property
    def no_of_trees(self):
        """The number of trees in the RandomForest model"""
        if self.is_classifier and len(self.labels) > 2:
            return int(len(self.model_dump_list) / len(self.labels))
        return len(self.model_dump_list)
        
    @property
    def graphviz_available(self):
        """ """
        if not hasattr(self, '_graphviz_available'):
            try:
                import graphviz.backend as be
                cmd = ["dot", "-V"]
                stdout, stderr = be.run(cmd, capture_output=True, check=True, quiet=True)
            except:
                print("""
                WARNING: you don't seem to have graphviz in your path (cannot run 'dot -V'), 
                so no dtreeviz visualisation of decision trees will be shown on the shadow trees tab.

                See https://github.com/parrt/dtreeviz for info on how to properly install graphviz 
                for dtreeviz. 
                """)
                self._graphviz_available = False
            else:
                self._graphviz_available = True
        return self._graphviz_available

    @property
    def decision_trees(self):
        """a list of ShadowDecTree objects"""
        if not hasattr(self, '_decision_trees'):
            print("Calculating ShadowDecTree for each individual decision tree...", flush=True)
                
            self._decision_trees = [
                ShadowDecTree.get_shadow_tree(self.model.get_booster(),
                                        self.X,
                                        self.y,
                                        feature_names=self.X.columns.tolist(),
                                        target_name='target',
                                        class_names = self.labels if self.is_classifier else None,
                                        tree_index=i)
                            for i in range(len(self.model_dump_list))]
        return self._decision_trees

    def decisiontree_df(self, tree_idx, index, pos_label=None):
        """dataframe with all decision nodes of a particular decision tree

        Args:
          tree_idx: the n'th tree in the random forest
          index: row index
          round:  (Default value = 2)
          pos_label:  positive class (Default value = None)

        Returns:
          dataframe with summary of the decision tree path

        """
        assert tree_idx >= 0 and tree_idx < self.no_of_trees, \
            f"tree index {tree_idx} outside 0 and number of trees ({len(self.decision_trees)}) range"
        idx = self.get_int_idx(index)
        assert idx >= 0 and idx < len(self.X), \
            f"=index {idx} outside 0 and size of X ({len(self.X)}) range"
        
        if self.is_classifier:
            if pos_label is None: 
                pos_label = self.pos_label
            if len(self.labels) > 2:
                tree_idx = tree_idx * len(self.labels) + pos_label
        return get_xgboost_path_df(self.model_dump_list[tree_idx], self.X.iloc[idx])


    def decisiontree_summary_df(self, tree_idx, index, round=2, pos_label=None):
        """formats decisiontree_df in a slightly more human readable format.

        Args:
          tree_idx: the n'th tree in the random forest
          index: row index
          round:  (Default value = 2)
          pos_label:  positive class (Default value = None)

        Returns:
          dataframe with summary of the decision tree path

        """
        idx = self.get_int_idx(index)
        return get_xgboost_path_summary_df(self.decisiontree_df(tree_idx, idx, pos_label=pos_label))


    def decision_path_file(self, tree_idx, index, show_just_path=False, pos_label=None):
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

        idx = self.get_int_idx(index)
        if self.is_classifier:
            if pos_label is None: 
                pos_label = self.pos_label
            if len(self.labels) > 2:
                tree_idx = tree_idx * len(self.labels) + pos_label

        viz = dtreeviz(self.decision_trees[tree_idx], 
                        X=self.X.iloc[idx], 
                        fancy=False,
                        show_node_labels = False,
                        show_just_path=show_just_path) 
        return viz.save_svg()

    def decision_path(self, tree_idx, index, show_just_path=False, pos_label=None):
        """get a dtreeviz visualization of a particular tree in the random forest.

        Args:
          tree_idx: the n'th tree in the random forest
          index: row index
          show_just_path (bool, optional): show only the path not rest of the 
                    tree. Defaults to False. 

        Returns:
          a IPython display SVG object for e.g. jupyter notebook.

        """
        if not self.graphviz_available:
            print("No graphviz 'dot' executable available!") 
            return None

        from IPython.display import SVG
        svg_file = self.decision_path_file(tree_idx, index, show_just_path, pos_label)
        return SVG(open(svg_file,'rb').read())

    def decision_path_encoded(self, tree_idx, index, show_just_path=False, pos_label=None):
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
        
        svg_file = self.decision_path_file(tree_idx, index, show_just_path, pos_label)
        encoded = base64.b64encode(open(svg_file,'rb').read()) 
        svg_encoded = 'data:image/svg+xml;base64,{}'.format(encoded.decode()) 
        return svg_encoded


    def plot_trees(self, index, highlight_tree=None, round=2, 
                higher_is_better=True, pos_label=None):
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
        idx=self.get_int_idx(index)
        assert idx is not None, 'invalid index'
        if self.is_classifier:
            if pos_label is None: 
                pos_label = self.pos_label
            y = self.y_binary(pos_label)[idx]
            xgboost_preds_df = get_xgboost_preds_df(
                self.model, self.X.iloc[[idx]], pos_label=pos_label)
            return plotly_xgboost_trees(xgboost_preds_df, 
                            y=y, 
                            highlight_tree=highlight_tree, 
                            target=self.target, 
                            higher_is_better=higher_is_better)
        else:
            y = self.y[idx]
            xgboost_preds_df = get_xgboost_preds_df(
                self.model, self.X.iloc[[idx]])
            return plotly_xgboost_trees(xgboost_preds_df, 
                            y=y, highlight_tree=highlight_tree,
                            target=self.target, units=self.units,
                            higher_is_better=higher_is_better)

    def calculate_properties(self, include_interactions=True):
        """

        Args:
          include_interactions:  If False do not calculate shap interaction value
            (Default value = True)

        Returns:

        """
        _ = self.decision_trees, self.model_dump_list
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


class ClassifierBunch:
    """ """
    def __init__(self, *args, **kwargs):
        raise ValueError("ClassifierBunch has been deprecated, use ClassifierExplainer instead...")

class RegressionBunch:
    """ """
    def __init__(self, *args, **kwargs):
        raise ValueError("RegressionBunch has been deprecated, use RegressionrExplainer instead...")

class RandomForestExplainerBunch:
    """ """
    def __init__(self, *args, **kwargs):
        raise ValueError("RandomForestExplainerBunch has been deprecated, use RandomForestExplainer instead...")

class RandomForestClassifierBunch:
    """ """
    def __init__(self, *args, **kwargs):
        raise ValueError("RandomForestClassifierBunch has been deprecated, use RandomForestClassifierExplainer instead...")

class RandomForestRegressionBunch:
    """ """
    def __init__(self, *args, **kwargs):
        raise ValueError("RandomForestRegressionBunch has been deprecated, use RandomForestRegressionExplainer instead...")




