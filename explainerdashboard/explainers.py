from abc import ABC, abstractmethod
import warnings
import base64

import pandas as pd
from pdpbox import pdp
import shap
from dtreeviz.trees import *

from sklearn.metrics import r2_score, roc_auc_score

from .explainer_methods import *
from .explainer_plots import *


class BaseExplainerBunch(ABC):
    """Abstract Base Class. Defines the basic functionality of an ExplainerBunch
    But does not yet have a defined shap_explainer.
    """
    def __init__(self, model, X, y=None, metric=r2_score,
                    cats=None, idxs=None, permutation_cv=None, na_fill=-999):
        """init

        :param model: a model with a scikit-learn compatible .fit and .predict method
        :type model: [type]
        :param X: a pd.DataFrame with your model features
        :type X: pd.DataFrame
        :param y: Dependent variable of your model, defaults to None
        :type y: pd.Series, optional
        :param metric: is a scikit-learn compatible metric function (or string), 
            defaults to r2_score
        :type metric: metric function, optional
        :param cats: list of variables that have been onehotencoded. Allows to 
            group categorical variables together in plots, defaults to None
        :type cats: list, optional
        :param idxs: list of row identifiers. Can be names, id's, etc. Get 
            converted to str, defaults to None
        :type idxs: list, optional
        :param permutation_cv: If given permutation importances get calculated 
            using cross validation, defaults to None
        :type permutation_cv: int, optional
        :param na_fill: The filler used for missing values, defaults to -999
        :type na_fill: int, optional
        """
        self.model  = model
        self.X = X.reset_index(drop=True)
        if y is not None:
            self.y = pd.Series(y).reset_index(drop=True)
        else:
            self.y = pd.Series(np.full(len(X), np.nan))
        self.metric = metric

        self.cats = cats
        if idxs is not None:
            if isinstance(idxs, list):
                self.idxs = [str(i) for i in idxs]
            else:
                self.idxs = list(idxs.astype(str))
        elif idxs=="use_index":
            self.idxs = list(X.index.astype(str))
        else:
            self.idxs = [str(i) for i in range(len(X))]

        self.permutation_cv = permutation_cv
        self.na_fill=na_fill
        self.columns = self.X.columns.tolist()
        self.is_classifier = False
        self.is_regression = False

    @classmethod
    def from_ModelBunch(cls, model_bunch, raw_data, metric,
                        index_column=None, permutation_cv=None, na_fill=-999,
                        *args, **kwargs):
        """create an ExplainerBunch from a ModelBunch. A ModelBunch is a class
        containing a model, data transformer, target name and used columns, with
        both a .transform() and a .predict() method.

        :param model_bunch: ModelBunch
        :type model_bunch: [type]
        :param raw_data: raw input data that will be transformed
        :type raw_data: pd.DataFrame
        :param metric: metric with which to evaluate permutation importances
        :type metric: metric function or str
        :param index_column: column that will be used for idxs, defaults to None
        :type index_column: str, optional
        :param permutation_cv: [description], defaults to None
        :type permutation_cv: [type], optional
        :param na_fill: [description], defaults to -999
        :type na_fill: int, optional
        :return: ExplainerBunch instance
        """
        assert not index_column or index_column in raw_data.columns, \
            f"{index_column} not in raw_data!"

        X, y = model_bunch.transform(raw_data)
        cats = raw_data.select_dtypes(include='object').columns.tolist()

        if index_column is not None:
            idxs = raw_data[index_column].astype(str).values.tolist()
        else:
            idxs = None
        return cls(model_bunch.model, X, y, metric,
                    cats, idxs, permutation_cv, na_fill,
                    *args, **kwargs)

    def __len__(self):
        return len(self.X)

    def random_index(self, y_min=None, y_max=None, pred_min=None, pred_max=None, return_str=False):
        """
        Return a random index from dataset.
        if y_values is given select an index for which y in y_values
        if return_str return str index from self.idxs
        """
        if y_min is None:
            y_min = self.y.min()
        if y_max is None:
            y_max = self.y.max()
        if pred_min is None:
            pred_min = self.preds.min()
        if pred_max is None:
            pred_max = self.preds.max()

        potential_idxs = self.y[(self.y>=y_min) & (self.y <= y_max) & 
                                (self.preds>=pred_min) & (self.preds <= pred_max)].index

        if len(potential_idxs) > 0:
            idx = np.random.choice(potential_idxs)
        else:
            return None
        if return_str:
            assert self.idxs is not None, \
                "no self.idxs property found..."
            return self.idxs[idx]
        return idx

    def __contains__(self, index):
        if self.get_int_idx(index) is not None:
            return True
        return False

    def get_int_idx(self, index):
        """
        Always returns an int index.
        If index is already int, simply return directly
        if index is str, lookup corresponding int index and return
        if index not found, return None
        """
        if isinstance(index, int):
            if index >= 0 and index < len(self):
                return index
        elif isinstance(index, str):
            if self.idxs is not None and index in self.idxs:
                return self.idxs.index(index)
        return None

    @property
    def preds(self):
        """model predictions"""
        if not hasattr(self, '_preds'):
            print("Calculating predictions...")
            self._preds = self.model.predict(self.X)
            
        return self._preds

    @property
    def ranks(self):
        if not hasattr(self, '_ranks'):
            print("Calculating ranks...")
            self._ranks = (pd.Series(self.preds)
                                .rank(method='min')
                                .divide(len(self.preds))
                                .values)
        return self._ranks

    def columns_ranked(self, cats=False):
        if cats:
            return self.mean_abs_shap_cats.Feature.tolist()
        else:
            return self.mean_abs_shap.Feature.tolist()

    def inverse_cats(self, col):
        """if col in self.columns, return equivalent col in self.columns_cats,
           if col in self.columns_cats, return equivalent in self.columns
        11"""
        if col in self.columns_cats:
            new_col = get_feature_dict(self.columns, self.cats)[col][0]
        else:
            new_col = [k for k, v in get_feature_dict(self.columns, self.cats).items() if col in v][0]
        return new_col

    def get_col(self, col):
        assert col in self.columns or col in self.cats, \
            f"{col} not in columns!"

        if col in self.X.columns:
            return self.X[col]
        elif col in self.cats:
            return retrieve_onehot_value(self.X, col)
        
    def get_col_value_plus_prediction(self, index, col):
        """return value of col and prediction for index"""
        assert index in self, f"index {index} not found"
        assert (col in self.X.columns) or (col in self.cats),\
            f"{col} not in columns of dataset"

        idx = self.get_int_idx(index)

        if col in self.X.columns:
            col_value = self.X[col].iloc[idx]
        elif col in self.cats:
            col_value = retrieve_onehot_value(self.X, col).iloc[idx]

        try:
            prediction = self.pred_probas[idx]
        except:
            prediction = self.preds[idx]

        return col_value, prediction

    @property
    def permutation_importances(self):
        """return the permatuation importances of the model features"""
        if not hasattr(self, '_perm_imps'):
            print("Calculating importances...")
            self._perm_imps = cv_permutation_importances(
                            self.model, self.X, self.y, self.metric,
                            cv=self.permutation_cv,
                            needs_proba=self.is_classifier)
        return self._perm_imps

    @property
    def permutation_importances_cats(self):
        """permutation importances with categoricals grouped"""
        if not hasattr(self, '_perm_imps_cats'):
            self._perm_imps_cats = cv_permutation_importances(
                            self.model, self.X, self.y, self.metric, self.cats,
                            cv=self.permutation_cv,
                            needs_proba=self.is_classifier)
        return self._perm_imps_cats

    @property
    def X_cats(self):
        """return model features DataFrame with onehot encoded categorical features
        reverse encoded to original state"""
        if not hasattr(self, '_X_cats'):
            self._X_cats = merge_categorical_columns(self.X, self.cats)
        return self._X_cats

    @property
    def columns_cats(self):
        """columns features with onehot encoded categorical features
        reverse encoded to original state"""
        if not hasattr(self, '_columns_cats'):
            self._columns_cats = self.X_cats.columns.tolist()
        return self._columns_cats

    @property
    @abstractmethod
    def shap_explainer(self):
        """this property will be supplied by the inheriting classes individually
        e.g. using KernelExplainer, TreeExplainer, DeepExplainer, etc"""
        raise NotImplementedError()

    @property
    def shap_base_value(self):
        """the intercept for the shap values. (i.e. 'what would the prediction be
        if we knew none of the features?')"""
        if not hasattr(self, '_shap_base_value'):
            self._shap_base_value = self.shap_explainer.expected_value
        return self._shap_base_value

    @property
    def shap_values(self):
        """SHAP values calculated using the shap library"""
        if not hasattr(self, '_shap_values'):
            print("Calculating shap values...")
            self._shap_values = self.shap_explainer.shap_values(self.X)
        return self._shap_values

    @property
    def shap_values_cats(self):
        """SHAP values when categorical features have been grouped"""
        if not hasattr(self, '_shap_values_cats'):
            print("Calculating shap values...")
            self._shap_values_cats = merge_categorical_shap_values(
                    self.X, self.shap_values, self.cats)
        return self._shap_values_cats

    @property
    def shap_interaction_values(self):
        """SHAP interaction values calculated using shap library"""
        if not hasattr(self, '_shap_interaction_values'):
            print("Calculating shap interaction values...")
            self._shap_interaction_values = \
                self.shap_explainer.shap_interaction_values(self.X)
        return self._shap_interaction_values

    @property
    def shap_interaction_values_cats(self):
        """SHAP interaction values with categorical features grouped"""
        if not hasattr(self, '_shap_interaction_values_cats'):
            print("Calculating shap interaction values...")
            self._shap_interaction_values_cats = \
                merge_categorical_shap_interaction_values(
                    self.X, self.X_cats, self.shap_interaction_values)
        return self._shap_interaction_values_cats

    @property
    def mean_abs_shap(self):
        """Mean absolute SHAP values per feature. Gives indication of overall
        importance of feature for predictions of model."""
        if not hasattr(self, '_mean_abs_shap'):
            self._mean_abs_shap = mean_absolute_shap_values(
                                self.columns, self.shap_values)
        return self._mean_abs_shap

    @property
    def mean_abs_shap_cats(self):
        """Mean absolute SHAP values per feature with categorical features grouped.
        Gives indication of overall importance of feature for predictions of model."""
        if not hasattr(self, '_mean_abs_shap_cats'):
            self._mean_abs_shap_cats = mean_absolute_shap_values(
                                self.columns, self.shap_values, self.cats)
        return self._mean_abs_shap_cats

    def calculate_properties(self, include_interactions=True):
        """Explicitely calculates all lazily calculated properties. Can be useful
        to call before saving ExplainerBunch to disk so that no need
        to calculate these properties when for example starting a Dashboard.

        :param include_interactions: shap interaction values can take a long time
        to compute for larger datasets with more features. Therefore you can choose
        not to calculate these, defaults to True
        :type include_interactions: bool, optional
        """
        _ = (self.preds, self.permutation_importances,
                self.shap_base_value, self.shap_values,
                self.shap_interaction_values, self.mean_abs_shap)
        if self.cats is not None:
            _ = (self.mean_abs_shap_cats, self.X_cats,
                    self.shap_values_cats)
        if include_interactions:
            _ = self.shap_interaction_values
            if self.cats is not None:
                _ = self.shap_interaction_values_cats

    def mean_abs_shap_df(self, topx=None, cutoff=None, cats=False):
        """returns a pd.DataFrame with the mean absolute shap values per features,
        sorted rom highest to lowest.

        :param topx: Only return topx most importance features, defaults to None
        :type topx: int, optional
        :param cutoff: Only return features with mean abs shap of at least cutoff, defaults to None
        :type cutoff: float, optional
        :param cats: group categorical variables, defaults to False
        :type cats: bool, optional
        :return:shap_df
        :rtype pd.DataFrame

        """
        shap_df = self.mean_abs_shap_cats if cats else self.mean_abs_shap

        if topx is None: topx = len(shap_df)
        if cutoff is None: cutoff = shap_df['MEAN_ABS_SHAP'].min()
        return shap_df[shap_df['MEAN_ABS_SHAP'] >= cutoff].head(topx)

    def shap_top_interactions(self, col, topx=None, cats=False):
        """returns the features that interact with feature col in descending order.

        :param col: feature for which you want to get the interactions
        :type col: str
        :param topx: Only return topx features, defaults to None
        :type topx: int, optional
        :param cats: Group categorical features, defaults to False
        :type cats: bool, optional
        :return: top_interactions
        :rtype: list

        """
        if cats:
            if hasattr(self, '_shap_interaction_values'):
                col_idx = self.X_cats.columns.get_loc(col)
                top_interactions = self.X_cats.columns[np.argsort(-np.abs(
                        self.shap_interaction_values_cats[:, col_idx, :]).mean(0))].tolist()
            else:
                top_interactions = self.mean_abs_shap_cats.Feature.values.tolist()
                top_interactions.insert(0, top_interactions.pop(
                    top_interactions.index(col))) #put col first

            if topx is None: topx = len(top_interactions)
            return top_interactions[:topx]
        else:
            if hasattr(self, '_shap_interaction_values'):
                col_idx = self.X.columns.get_loc(col)
                top_interactions = self.X.columns[np.argsort(-np.abs(
                            self.shap_interaction_values[:, col_idx, :]).mean(0))].tolist()
            else:
                interaction_idxs = shap.common.approximate_interactions(
                    col, self.shap_values, self.X)
                top_interactions = self.X.columns[interaction_idxs].tolist()
                top_interactions.insert(0, top_interactions.pop(-1)) #put col first

            if topx is None: topx = len(top_interactions)
            return top_interactions[:topx]

    def shap_interaction_values_by_col(self, col, cats=False):
        """
        returns the shap interaction values[np.array(N,N)] for feature col

        :param col: features for which you'd like to get the interaction value
        :type col: str
        :param cats: group categorical, defaults to False
        :type cats: bool, optional
        :return: shap_interaction_values
        :rtype: np.array(N,N)
        """
        if cats:
            return self.shap_interaction_values_cats[:,
                        self.X_cats.columns.get_loc(col), :]
        else:
            return self.shap_interaction_values[:,
                        self.X.columns.get_loc(col), :]

    def permutation_importances_df(self, topx=None, cutoff=None, cats=False):
        """Returns pd.DataFrame with features ordered by permutation importance.
        For more about permutation importances see https://explained.ai/rf-importance/index.html

        :param topx: only return topx most important features, defaults to None
        :type topx: int, optional
        :param cutoff: only return features with importance of at least cutoff, defaults to None
        :type cutoff: float, optional
        :param cats: Group categoricals, defaults to False
        :type cats: bool, optional
        :return: importance_df
        :rtype: pd.DataFrame
        """
        importance_df = self.permutation_importances_cats.reset_index() if cats \
                                else self.permutation_importances.reset_index()

        if topx is None: topx = len(importance_df)
        if cutoff is None: cutoff = importance_df.Importance.min()
        return importance_df[importance_df.Importance > cutoff].head(topx)

    def importances_df(self, type="permutation", topx=None, cutoff=None, cats=False):
        """wrapper function for mean_abs_shap_df() and permutation_importance_df()"""
        if type=='permutation':
            return self.permutation_importances_df(topx, cutoff, cats)
        elif type=='shap':
            return self.mean_abs_shap_df(topx, cutoff, cats)

    def contrib_df(self, index, cats=True, topx=None, cutoff=None):
        """returns a contrib_df pd.DataFrame with the shap value contributions
        to the prediction for index. Used as input for the plot_contributions()
        method.

        :param index: index for which to calculate contributions
        :type index: int or str
        :param cats: Group categoricals, defaults to True
        :type cats: bool, optional
        :param topx: Only return topx features, remainder called REST, defaults to None
        :type topx: int, optional
        :param cutoff: only return features with at least cutoff contributions, defaults to None
        :type cutoff: float, optional
        :return: contrib_df
        :rtype: pd.DataFrame
        """
        idx = self.get_int_idx(index)
        if cats:
            return get_contrib_df(self.shap_base_value, self.shap_values_cats[idx],
                                    self.X_cats.iloc[[idx]], topx, cutoff)
        else:
            return get_contrib_df(self.shap_base_value, self.shap_values[idx],
                                    self.X.iloc[[idx]], topx, cutoff)

    def contrib_summary_df(self, index, cats=True,
                            topx=None, cutoff=None, round=round):
        """Takes a contrib_df, and formats it to a more human readable format"""
        idx = self.get_int_idx(index) # if passed str convert to int index
        return get_contrib_summary_df(self.contrib_df(idx, cats, topx, cutoff),
                                        classification=self.is_classifier,
                                        round=round)

    def interactions_df(self, col, cats=False, topx=None, cutoff=None):
        importance_df = mean_absolute_shap_values(
                            self.columns_cats if cats else self.columns, 
                            self.shap_interaction_values_by_col(col, cats))

        if topx is None: topx = len(importance_df)
        if cutoff is None: cutoff = importance_df.MEAN_ABS_SHAP.min()
        return importance_df[importance_df.MEAN_ABS_SHAP > cutoff].head(topx)
    
    def formatted_contrib_df(self, index, round=None, lang='en'):
        """Out PowerBI specialist wanted this the contrib_df in a certain format in order
        to conventiently build powerbi dashboards from the output of get_dfs.

        Additional language option for output in Dutch (lang='nl')


        :param index: index to return contrib_df for
        :type index: str or int
        :param round: rounding of continuous features, defaults to 2
        :type round: int, optional
        :param lang: language to name the columns, defaults to 'en'
        :type lang: str, optional
        :return: formatted_contrib_df
        :rtype: pd.DataFrame
        """
        cdf = self.contrib_df(index, cats=True).copy()
        cdf.reset_index(inplace=True)
        cdf.loc[cdf.col=='base_value', 'value'] = np.nan
        cdf['row_id'] = self.get_int_idx(index)
        cdf['name_id'] = self.idxs[self.get_int_idx(index)]
        cdf['cat_value'] = np.where(cdf.col.isin(self.cats), cdf.value, np.nan)
        cdf['cont_value'] = np.where(cdf.col.isin(self.cats), np.nan, cdf.value)
        if round is not None:
            rounded_cont = np.round(cdf['cont_value'].values.astype(float), round)
            cdf['value'] = np.where(cdf.col.isin(self.cats), cdf.cat_value, rounded_cont)
        cdf['type'] = np.where(cdf.col.isin(self.cats), 'cat', 'cont')
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

    def get_pdp_result(self, col, index=None, drop_na=True,
                        sample=500, num_grid_points=20):
        """Uses the PDPBox to calculate partial dependences for feature col.

        :param col: Feature to calculate partial dependences for
        :type col: str
        :param index: Index of row to put at iloc[0], defaults to None
        :type index: int or str, optional
        :param drop_na: drop rows where col equals na_fill, defaults to True
        :type drop_na: bool, optional
        :param sample: Number of rows to sample for plot, defaults to 500
        :type sample: int, optional
        :param num_grid_points: Number of grid points to calculate, defaults to 20
        :type num_grid_points: int, optional
        :return: pdp_result
        :rtype: PDPBox.pdp_result
        """
        assert col in self.X.columns or col in self.cats, \
            f"{col} not in columns of dataset"

        features = get_feature_dict(self.X.columns, self.cats)[col]

        if index is not None:
            idx = self.get_int_idx(index)
            if len(features)==1 and drop_na: # regular col, not onehotencoded
                sample_size=min(sample, len(self.X[(self.X[features[0]] != self.na_fill)])-1)
                sampleX = pd.concat([
                    self.X[self.X.index==idx],
                    self.X[(self.X.index!=idx) & (self.X[features[0]] != self.na_fill)]\
                            .sample(sample_size)],
                    ignore_index=True, axis=0)
            else:
                sample_size = min(sample, len(self.X)-1)
                sampleX = pd.concat([
                    self.X[self.X.index==idx],
                    self.X[(self.X.index!=idx)].sample(sample_size)],
                    ignore_index=True, axis=0)
        else:
            if len(features)==1 and drop_na: # regular col, not onehotencoded
                sample_size=min(sample, len(self.X[(self.X[features[0]] != self.na_fill)])-1)
                sampleX = self.X[(self.X[features[0]] != self.na_fill)]\
                                .sample(sample_size)
            else:
                sampleX = self.X.sample(min(sample, len(self.X)))

        # if only a single value (i.e. not onehot encoded, take that value
        # instead of list):
        if len(features)==1: features=features[0]
        pdp_result = pdp.pdp_isolate(
                model=self.model, dataset=sampleX,
                model_features=self.X.columns,
                num_grid_points=num_grid_points, feature=features)
        if isinstance(features, list):
            # strip 'col_' from the grid points
            pdp_result.feature_grids = \
                pd.Series(pdp_result.feature_grids).str.split(col+'_').str[1].values
        return pdp_result

    def get_dfs(self, cats=True, round=None, lang='en'):
        """returns two pd.DataFrames. The first with id, prediction, actual and
        feature values, and one with only id and shap values.
        These can then be used to build your own custom dashboard on these data,
        for example using PowerBI.

        :param cats: group categorical variables, defaults to True
        :type cats: bool, optional
        :return: cols_df, shap_df
        :rtype: pd.DataFrame, pd.DataFrame
        """
        if cats:
            cols_df = self.X_cats.copy()
            shap_df = pd.DataFrame(self.shap_values_cats, columns = self.X_cats.columns)
        else:
            cols_df = self.X.copy()
            shap_df = pd.DataFrame(self.shap_values, columns = self.X.columns)

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
                cats=True, round=None, lang='en'):
        """Writes two dataframes generated by .get_dfs() to a sql server.
        Tables will be called name_COLS and name_SHAP

        :param conn: database connecter acceptable for pd.to_sql
        :type conn: sqlalchemy.engine.Engine or sqlite3.Connection
        :param schema: schema to write to
        :type schema: str
        :param name: name prefix of tables
        :type name: str
        :param cats: group categorical variables, defaults to True
        :type cats: bool, optional
        :param if_exists: How to behave if the table already exists.
        :type if_exists: {‘fail’, ‘replace’, ‘append’}, default ‘fail’
        """
        cols_df, shap_df, contribs_df = self.get_dfs(cats, round, lang)
        cols_df.to_sql(con=conn, schema=schema, name=name+"_COLS",
                        if_exists=if_exists, index=False)
        shap_df.to_sql(con=conn, schema=schema, name=name+"_SHAP",
                        if_exists=if_exists, index=False)
        contribs_df.to_sql(con=conn, schema=schema, name=name+"_CONTRIB",
                        if_exists=if_exists, index=False)

    def plot_importances(self, type='shap', topx=None, cats=False):
        """return Plotly fig with barchart of importances in descending order.

        :param type: 'shap' for mean absolute shap values, 'permutation' for
        permutation importances, defaults to 'shap'
        :type type: str, optional
        :param topx: Only return topx features, defaults to None
        :type topx: int, optional
        :param cats: Group categoricals defaults to False
        :type cats: bool, optional
        :return: fig
        :rtype: plotly.fig
        """
        importances_df = self.importances_df(type=type, topx=topx, cats=cats)
        return plotly_importances_plot(importances_df)

    def plot_interactions(self, col, cats=False, topx=None):
        interactions_df = self.interactions_df(col, cats=cats, topx=topx)
        return plotly_importances_plot(interactions_df)

    def plot_shap_contributions(self, index, cats=True,
                                    topx=None, cutoff=None, round=2):
        """reutn Plotly fig with waterfall plot of shap value contributions
        to the model prediction for index.

        :param index: index for which to display prediction
        :type index: int or str
        :param cats: Group categoricals, defaults to True
        :type cats: bool, optional
        :param topx: Only display topx features, defaults to None
        :type topx: int, optional
        :param cutoff: Only display features with at least cutoff contribution, defaults to None
        :type cutoff: float, optional
        :param round: round contributions to round precision, defaults to 2
        :type round: int, optional
        :return: fig
        :rtype: plotly.Fig
        """
        contrib_df = self.contrib_df(self.get_int_idx(index), cats, topx, cutoff)
        return plotly_contribution_plot(contrib_df,
                    classification=self.is_classifier, round=round)

    def plot_shap_summary(self, topx=None, cats=False):
        """Displays all individual shap value for each feature in a horizontal
        scatter chart in descending order by mean absolute shap value.

        :param topx: Only display topx most important features , defaults to 10
        :type topx: int, optional
        :param cats: Group categoricals , defaults to False
        :type cats: bool, optional
        :return: fig
        :rtype: plotly.Fig
        """
        if cats:
            return plotly_shap_scatter_plot(
                                self.shap_values_cats,
                                self.X_cats,
                                self.importances_df(type='shap', topx=topx, cats=True)\
                                        ['Feature'].values.tolist())
        else:
            return plotly_shap_scatter_plot(
                                self.shap_values,
                                self.X,
                                self.importances_df(type='shap', topx=topx)\
                                        ['Feature'].values.tolist())

    def plot_shap_interaction_summary(self, col, topx=None, cats=False):
        """Displays all individual shap interaction values for each feature in a
        horizontal scatter chart in descending order by mean absolute shap value.

        :param col: feature for which
        :type col: [type]
        :param topx: [description], defaults to 10
        :type topx: int, optional
        :param cats: [description], defaults to False
        :type cats: bool, optional
        :return: [description]
        :rtype: [type]
        """
        interact_cols = self.shap_top_interactions(col, cats=cats)
        if topx is None: topx = len(interact_cols)
        if cats:
            return plotly_shap_scatter_plot(
                self.shap_interaction_values_by_col(col, cats=cats),
                self.X_cats, interact_cols[:topx])
        else:
            return plotly_shap_scatter_plot(
                self.shap_interaction_values_by_col(col),
                self.X, interact_cols[:topx])

    def plot_shap_dependence(self, col, color_col=None, highlight_idx=None, cats=False):
        """
        Plots a shap dependence plot:
            - on the x axis the possible values of the feature `col`
            - on the y axis the associated individual shap values

        :param color_col: if color_col provided then shap values colored (blue-red) according to feature color_col
        :param highlight_idx: individual observation to be highlighed in the plot.
        :param cats: group categorical variables
        """
        if cats:
            if col in self.cats:
                return plotly_shap_violin_plot(self.X_cats, self.shap_values_cats, col, color_col)
            else:
                return plotly_dependence_plot(self.X_cats, self.shap_values_cats,
                                                col, color_col,
                                                highlight_idx=highlight_idx,
                                                na_fill=self.na_fill)
        else:
            return plotly_dependence_plot(self.X, self.shap_values,
                                            col, color_col,
                                            highlight_idx=highlight_idx,
                                            na_fill=self.na_fill)

    def plot_shap_interaction_dependence(self, col, interact_col,
                                            highlight_idx=None, cats=False):
        """plots a dependence plot for shap interaction effects

        :param col: feature for which to find interaction values
        :type col: str
        :param interact_col: feature for which interaction value are displayed
        :type interact_col: str
        :param highlight_idx: idx that will be highlighted, defaults to None
        :type highlight_idx: int, optional
        :param cats: group categorical features, defaults to False
        :type cats: bool, optional
        :return: Plotly Fig
        :rtype: plotly.Fig
        """
        if cats and interact_col in self.cats:
            return plotly_shap_violin_plot(
                self.X_cats, 
                self.shap_interaction_values_by_col(col, cats),
                interact_col, col, interaction=True)
        else:
            return plotly_dependence_plot(self.X_cats if cats else self.X,
                self.shap_interaction_values_by_col(col, cats),
                interact_col, col, highlight_idx=highlight_idx,
                interaction=True)

    def plot_pdp(self, col, index=None, drop_na=True, sample=100,
                    num_grid_lines=100, num_grid_points=10):
        """returns plotly fig for a partial dependence plot showing ice lines
        for num_grid_lines rows, average pdp based on sample of sample.
        If index is given, display pdp for this specific index.

        :param col: feature to display pdp graph for
        :type col: str
        :param index: index to highlight in pdp graph, defaults to None
        :type index: int or str, optional
        :param drop_na: if true drop samples with value equal to na_fill, defaults to True
        :type drop_na: bool, optional
        :param sample: sample size on which the average pdp will be calculated, defaults to 100
        :type sample: int, optional
        :param num_grid_lines: number of ice lines to display, defaults to 100
        :type num_grid_lines: int, optional
        :param num_grid_points: number of points on the x axis to calculate the pdp for, defaults to 10
        :type num_grid_points: int, optional
        :return: fig
        :rtype: plotly.Fig
        """
        pdp_result = self.get_pdp_result(col, index,
                            drop_na=drop_na, sample=sample,
                            num_grid_points=num_grid_points)

        if index is not None:
            try:
                col_value, pred = self.get_col_value_plus_prediction(index, col)
                return plotly_pdp(pdp_result,
                                display_index=0, # the idx to be displayed is always set to the first row by self.get_pdp_result()
                                index_feature_value=col_value, index_prediction=pred,
                                feature_name=col,
                                num_grid_lines=min(num_grid_lines, sample, len(self.X)))
            except:
                return plotly_pdp(pdp_result, feature_name=col,
                        num_grid_lines=min(num_grid_lines, sample, len(self.X)))
        else:
            return plotly_pdp(pdp_result, feature_name=col,
                        num_grid_lines=min(num_grid_lines, sample, len(self.X)))


class TreeExplainerBunch(BaseExplainerBunch):
    """Defines an explainer bunch for tree based models (random forests, xgboost,etc).

    Generates a self.shap_explainer based on shap.TreeExplainer
    """
    @property
    def shap_explainer(self):
        if not hasattr(self, '_shap_explainer'):
            print("Generating shap TreeExplainer...")
            if str(type(self.model))[-15:-2]=='XGBClassifier':
                warnings.warn("Warning: shap values for XGBoost models get calcaulated"
                    "against the background data X, not against the data the"
                    "model was trained on.")
                self._shap_explainer = shap.TreeExplainer(
                                            self.model, self.X,
                                            model_output="probability",
                                            feature_dependence= "independent")
            else:
                self._shap_explainer = shap.TreeExplainer(self.model)
        return self._shap_explainer



class LinearExplainerBunch(BaseExplainerBunch):
    """Defines an explainer bunch for linear models.

    Generates a self.shap_explainer based on shap.LinearExplainer
    """
    @property
    def shap_explainer(self):
        if not hasattr(self, '_shap_explainer'):
            print("Generating shap LinearExplainer...")
            self._shap_explainer = shap.LinearExplainer(self.model)
        return self._shap_explainer

class DeepExplainerBunch(BaseExplainerBunch):
    """Defines an explainer bunch for Deep Learning neural nets.

    Generates a self.shap_explainer based on shap.DeepExplainer
    """
    @property
    def shap_explainer(self):
        if not hasattr(self, '_shap_explainer'):
            print("Generating shap DeepExplainer...")
            self._shap_explainer = shap.DeepExplainer(self.model)
        return self._shap_explainer


class KernelExplainerBunch(BaseExplainerBunch):
    """Defines an explainer bunch for any type of model.

    Generates a self.shap_explainer based on shap.KernelExplainer
    """
    @property
    def shap_explainer(self):
        if not hasattr(self, '_shap_explainer'):
            print("Generating shap TreeExplainer...")
            self._shap_explainer = shap.KernelExplainer(self.model)
        return self._shap_explainer


class RandomForestExplainerBunch(TreeExplainerBunch):
    """
    RandomForestBunch allows for the analysis of individual DecisionTrees that
    make up the RandomForest.
    """
    
    @property
    def graphviz_available(self):
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
    def shadow_trees(self):
        if not hasattr(self, '_shadow_trees'):
            print("Generating shadow trees...")
            self._shadow_trees = get_shadow_trees(self.model, self.X, self.y)
        return self._shadow_trees

    def shadowtree_df(self, tree_idx, index):
        """returns a pd.DataFrame with all decision nodes of a particular
                        tree (indexed by tree_idx) for a particular observation
                        (indexed by index)"""
        assert tree_idx >= 0 and tree_idx < len(self.shadow_trees), \
            f"tree index {tree_idx} outside 0 and number of trees ({len(self.shadow_trees)}) range"
        idx=self.get_int_idx(index)
        assert idx >= 0 and idx < len(self.X), \
            f"=index {idx} outside 0 and size of X ({len(self.X)}) range"
        if self.is_classifier:
            return get_shadowtree_df(self.shadow_trees[tree_idx], self.X.iloc[idx],
                    pos_label=self.pos_label)
        else:
            return get_shadowtree_df(self.shadow_trees[tree_idx], self.X.iloc[idx])

    def shadowtree_df_summary(self, tree_idx, index, round=2):
        """formats shadowtree_df in a slightly more human readable format."""
        idx=self.get_int_idx(index)
        return shadowtree_df_summary(self.shadowtree_df(tree_idx, idx),
                    classifier=self.is_classifier, round=round)

    def decision_path_file(self, tree_idx, index):
        if not self.graphviz_available:
            print("No graphviz 'dot' executable available!") 
            return None

        idx = self.get_int_idx(index)

        if self.is_regression:
            viz = dtreeviz(self.model.estimators_[tree_idx],
               self.X, self.y, 
               target_name='Target',
               #orientation ='LR',  # left-right orientation
               feature_names=self.columns,
               X=self.X.iloc[idx, :],)
        elif self.is_classifier:
            viz = dtreeviz(self.model.estimators_[tree_idx],
               self.X, self.y, 
               target_name='Target',
               #orientation ='LR',  # left-right orientation
               feature_names=self.columns,
               class_names=self.labels,
               X=self.X.iloc[idx, :]) 
        return viz.save_svg()

    def decision_path(self, tree_idx, index):
        if not self.graphviz_available:
            print("No graphviz 'dot' executable available!") 
            return None

        from IPython.display import SVG
        svg_file = self.decision_path_file(tree_idx, index)
        return SVG(open(svg_file,'rb').read())

    def decision_path_encoded(self, tree_idx, index):
        if not self.graphviz_available: 
            print("No graphviz 'dot' executable available!")
            return None

        svg_file = self.decision_path_file(tree_idx, index)
        encoded = base64.b64encode(open(svg_file,'rb').read()) 
        #encoded = base64.b64encode(viz.svg()) 
        svg_encoded = 'data:image/svg+xml;base64,{}'.format(encoded.decode()) 
        return svg_encoded


    def plot_trees(self, index, round=2):
        """returns a plotly barchart with the values of the predictions
                of each individual tree for observation idx"""
        #print('explainer call')
        idx=self.get_int_idx(index)
        assert idx is not None, 'invalid index'
        if self.is_classifier:
            return plotly_tree_predictions(self.model, self.X.iloc[[idx]],
                        round=round, pos_label=self.pos_label)
        else:
            return plotly_tree_predictions(self.model, self.X.iloc[[idx]], round)

    def calculate_properties(self, include_interactions=True):
        _ = self.shadow_trees
        super().calculate_properties(include_interactions)


class ClassifierBunch(BaseExplainerBunch):
    """ExplainerBunch for classification models. Defines the shap values for
    each possible class in the classifications.

    You assign the positive label class afterwards with e.g. .pos_label=0

    In addition defines a number of plots specific to classification problems
    such as a precision plot, confusion matrix, roc auc curve and pr auc curve.
    """
    def __init__(self, model,  X, y=None, metric=roc_auc_score,
                    cats=None, idxs=None, permutation_cv=None, na_fill=-999,
                    labels=None, pos_label=1):
        """Combared to BaseExplainerBunch defines two additional parameters:
        :param labels: list of str labels for the different classes, defaults to e.g. ['0', '1'] for a binary classification
        :type labels: list of str, optional
        :param pos_label: class that should be used as the positive class, defaults to 1
        :type pos_label: int or str (if str, needs to be in labels), optional
        """
        super().__init__(model, X, y, metric, cats, idxs, permutation_cv, na_fill)

        if labels is not None:
            self.labels = labels
        elif hasattr(self.model, 'classes_'):
                self.labels = self.model.classes_
        else:
            self.labels = [str(i) for i in range(self.y.nunique())]
        self.pos_label = pos_label
        self.is_classifier = True

    @property
    def pos_label(self):
        return self._pos_label

    @pos_label.setter
    def pos_label(self, label):
        if isinstance(label, int) and label >=0 and label <len(self.labels):
            self._pos_label = label
        elif isinstance(label, str) and label in self.labels:
            self._pos_label = self.labels.index(label)
        else:
            raise ValueError(f"'{label}' not in labels")

    @property
    def pos_label_str(self):
        return self.labels[self.pos_label]

    @property
    def y_binary(self):
        return np.where(self.y.values==self.pos_label, 1, 0)

    def get_prop_for_label(self, prop:str, label):
        tmp = self.pos_label
        self.pos_label = label
        ret = getattr(self, prop)
        self.pos_label = tmp
        return ret

    @property
    def pred_probas(self):
        """returns pred_proba for pos_label class"""
        return self.pred_probas_raw[:, self.pos_label]

    @property
    def ranks(self):
        """returns ranks for pos_label class"""
        return self.ranks_raw[:, self.pos_label]



    @property
    def pred_probas_raw(self):
        """returns pred_probas with probability for each class"""
        if not hasattr(self, '_pred_probas'):
            print("Calculating prediction probabilities...")
            assert hasattr(self.model, 'predict_proba'), \
                "model does not have a predict_proba method!"
            self._pred_probas =  self.model.predict_proba(self.X)
        return self._pred_probas

    @property
    def ranks_raw(self):
        if not hasattr(self, '_ranks_raw'):
            print("Calculating ranks...")
            self._ranks_raw = (pd.DataFrame(self.pred_probas_raw)
                                .rank(method='min')
                                .divide(len(self.pred_probas_raw))
                                .values)
        return self._ranks_raw

    @property
    def permutation_importances(self):
        """return the permatuation importances of the model features"""
        if not hasattr(self, '_perm_imps'):
            print("Calculating importances...")
            self._perm_imps = [cv_permutation_importances(
                            self.model, self.X, self.y, self.metric,
                            cv=self.permutation_cv,
                            needs_proba=self.is_classifier,
                            pos_label=label) for label in range(len(self.labels))]
        return self._perm_imps[self.pos_label]

    @property
    def permutation_importances_cats(self):
        """permutation importances with categoricals grouped"""
        if not hasattr(self, '_perm_imps_cats'):
            self._perm_imps_cats = [cv_permutation_importances(
                            self.model, self.X, self.y, self.metric, self.cats,
                            cv=self.permutation_cv,
                            needs_proba=self.is_classifier,
                            pos_label=label) for label in range(len(self.labels))]
        return self._perm_imps_cats[self.pos_label]

    @property
    def shap_base_value(self):
        if not hasattr(self, '_shap_base_value'):
            self._shap_base_value = self.shap_explainer.expected_value
            if isinstance(self._shap_base_value, np.ndarray):
                self._shap_base_value = list(self._shap_base_value)
            if len(self.labels)==2 and isinstance(self._shap_base_value, (np.floating, float)):
                self._shap_base_value = [1-self._shap_base_value, self._shap_base_value]
            assert len(self._shap_base_value)==len(self.labels),\
                f"len(shap_explainer.expected_value)={len(self._shap_base_value)}"\
                 + "and len(labels)={len(self.labels)} do not match!"
        return self._shap_base_value[self.pos_label]

    @property
    def shap_values(self):
        if not hasattr(self, '_shap_values'):
            print("Calculating shap values...")
            self._shap_values = self.shap_explainer.shap_values(self.X)
            if not isinstance(self._shap_values, list) and len(self.labels)==2:
                self._shap_values = [1-self._shap_values, self._shap_values]

            assert len(self._shap_values)==len(self.labels),\
                f"len(shap_values)={len(self._shap_base_value)}"\
                 + f"and len(labels)={len(self.labels)} do not match!"
        return self._shap_values[self.pos_label]

    @property
    def shap_values_cats(self):
        if not hasattr(self, '_shap_values_cats'):
            _ = self.shap_values
            self._shap_values_cats = [
                merge_categorical_shap_values(
                    self.X, sv, self.cats) for sv in self._shap_values]
        return self._shap_values_cats[self.pos_label]

    @property
    def shap_interaction_values(self):
        if not hasattr(self, '_shap_interaction_values'):
            print("Calculating shap interaction values...")
            _ = self.shap_values #make sure shap values have been calculated
            self._shap_interaction_values = self.shap_explainer.shap_interaction_values(self.X)
            if not isinstance(self._shap_interaction_values, list) and len(self.labels)==2:
                self._shap_interaction_values = [1-self._shap_interaction_values,
                                                    self._shap_interaction_values]
            self._shap_interaction_values = [
                normalize_shap_interaction_values(siv, self.shap_values)
                    for siv, sv in zip(self._shap_interaction_values, self._shap_values)]
        return self._shap_interaction_values[self.pos_label]

    @property
    def shap_interaction_values_cats(self):
        if not hasattr(self, '_shap_interaction_values_cats'):
            _ = self.shap_interaction_values
            self._shap_interaction_values_cats = [
                merge_categorical_shap_interaction_values(
                    self.X, self.X_cats, siv) for siv in self._shap_interaction_values]
        return self._shap_interaction_values_cats[self.pos_label]

    @property
    def mean_abs_shap(self):
        if not hasattr(self, '_mean_abs_shap'):
            _ = self.shap_values
            self._mean_abs_shap = [mean_absolute_shap_values(
                                self.columns, sv) for sv in self._shap_values]
        return self._mean_abs_shap[self.pos_label]

    @property
    def mean_abs_shap_cats(self):
        if not hasattr(self, '_mean_abs_shap_cats'):
            _ = self.shap_values
            self._mean_abs_shap_cats = [mean_absolute_shap_values(
                                self.columns, sv, self.cats) for sv in self._shap_values]
        return self._mean_abs_shap_cats[self.pos_label]

    def cutoff_fraction(self, fraction, pos_label=None):
        if pos_label is None:
            return pd.Series(self.pred_probas).nlargest(int((1-fraction)*len(self))).min()
        else:
            return pd.Series(self.pred_probas_raw[:, pos_label]).nlargest(int((1-fraction)*len(self))).min()


    def get_pdp_result(self, col, index=None, drop_na=True,
                        sample=1000, num_grid_points=20):
        pdp_result = super(ClassifierBunch, self).get_pdp_result(
                                col, index, drop_na, sample, num_grid_points)
        if len(self.labels)==2:
            # for binary classifer PDPBox only gives pdp for the positive class.
            # instead of a list of pdps for every class
            # so we simply inverse when predicting the negative class
            if self.pos_label==0:
                pdp_result.pdp = 1 - pdp_result.pdp
                pdp_result.ice_lines = 1-pdp_result.ice_lines
            return pdp_result
        else:
             return pdp_result[self.pos_label]

    def random_index(self, y_values=None, return_str=False,
                    pred_proba_min=None, pred_proba_max=None,
                    rank_min=None, rank_max=None):
        """
        Return a random index from dataset.
        if y_values is given select an index for which y in y_values
        if return_str return str index from self.idxs

        if pred_proba_min(max) is given, return an index with at least a predicted
        probabiity of positive class of pred_proba_min(max)

        if rank_min(max) is given, return an index with at least a predicted
        rank of probabiity of positive class of rank_min(max)
        """
        if (y_values is None 
            and pred_proba_min is None and pred_proba_max is None
            and rank_min is None and rank_max is None):
            potential_idxs = self.y.index
        else:
            if y_values is None: y_values = self.y.unique().tolist()
            if not isinstance(y_values, list): y_values = [y_values]
            if pred_proba_min is None: pred_proba_min = self.pred_probas.min()
            if pred_proba_max is None: pred_proba_max = self.pred_probas.max()
            if rank_min is None: rank_min = 0.0
            if rank_max is None: rank_max = 1.0
            
            potential_idxs = self.y[(self.y.isin(y_values)) &
                            (self.pred_probas >= pred_proba_min) &
                            (self.pred_probas <= pred_proba_max) &
                            (self.ranks > rank_min) &
                            (self.ranks <= rank_max)].index
        if not potential_idxs.empty:
            idx = np.random.choice(potential_idxs)
        else:
            return None
        if return_str:
            assert self.idxs is not None, \
                "no self.idxs property found..."
            return self.idxs[idx]
        return idx

    def precision_df(self, bin_size=None, quantiles=None, multiclass=False):
        """returns a pd.DataFrame with predicted probabilities and actually
        observed number of positive cases (i.e. precision)

        :param bin_size: group predictions in bins of size bin_size, defaults to 0.1
        :type bin_size: float, optional
        :param quantiles: group predictions in evenly sized quantiles of size quantiles, defaults to None
        :type quantiles: int, optional
        :param multiclass: whether to calculate precision for every class
        :type multiclass: bool, optional
        :return: precision_df
        :rtype: pd.DataFrame
        """
        assert self.pred_probas is not None
        if bin_size is None and quantiles is None:
            bin_size=0.1 # defaults to bin_size=0.1
        if multiclass:
            return get_precision_df(self.pred_probas_raw, self.y,
                                bin_size, quantiles, pos_label=self.pos_label)
        else:
            return get_precision_df(self.pred_probas, self.y_binary, bin_size, quantiles)

    def lift_curve_df(self):
        return get_lift_curve_df(self.pred_probas, self.y, self.pos_label)

    def plot_precision(self, bin_size=None, quantiles=None, cutoff=0.5, multiclass=False):
        """plots predicted probability on the x-axis
        binned by bin_size, and observed precision (fraction of actual positive
        cases) on the y-axis"""

        if bin_size is None and quantiles is None:
            bin_size=0.1 # defaults to bin_size=0.1
        precision_df = self.precision_df(
                bin_size=bin_size, quantiles=quantiles, multiclass=multiclass)
        return plotly_precision_plot(precision_df,
                    cutoff=cutoff, labels=self.labels, pos_label=self.pos_label)

    def plot_cumulative_precision(self):
        return plotly_cumulative_precision_plot(
                    self.lift_curve_df(), self.labels, self.pos_label)

    def plot_confusion_matrix(self, cutoff=0.5, normalized=False):
        """plots a standard 2d confusion
        matrix, depending on model cutoff. If normalized display percentage
        otherwise counts."""
        if len(self.labels)==2:
            def order_binary_labels(labels, pos_label):
                pos_index = labels.index(pos_label)
                return [labels[1-pos_index], labels[pos_index]]
            labels = order_binary_labels(self.labels, self.pos_label_str)
        else:
            labels = ['Not ' + self.pos_label_str, self.pos_label_str]
        return plotly_confusion_matrix(
                self.y_binary, self.pred_probas,
                cutoff=cutoff, normalized=normalized,
                labels=labels)

    def plot_lift_curve(self, cutoff=None, percentage=False, round=2):
        return plotly_lift_curve(self.lift_curve_df(), cutoff, percentage, round)

    def plot_cumulative_precision(self):
        return plotly_cumulative_precision_plot(self.lift_curve_df(), 
                labels=self.labels, pos_label=self.pos_label)

    def plot_classification(self, cutoff=0.5, percentage=True):
        return plotly_classification_plot(self.pred_probas, self.y, self.labels, cutoff, percentage=percentage)

    def plot_roc_auc(self, cutoff=0.5):
        """plots ROC_AUC curve. The TPR and FPR of a particular
            cutoff is displayed in crosshairs."""
        return plotly_roc_auc_curve(self.y_binary, self.pred_probas, cutoff=cutoff)

    def plot_pr_auc(self, cutoff=0.5):
        """plots PR_AUC curve. the precision and recall of particular
            cutoff is displayed in crosshairs."""
        return plotly_pr_auc_curve(self.y_binary, self.pred_probas, cutoff=cutoff)

    def calculate_properties(self, include_interactions=True):
        _ = self.pred_probas
        super().calculate_properties(include_interactions)


class RegressionBunch(BaseExplainerBunch):
    """
     ExplainerBunch for regression models.

    In addition defines a number of plots specific to regression problems
    such as a predicted vs actual and residual plots.
    """
    def __init__(self, model,  X, y=None, metric=roc_auc_score,
                    cats=None, idxs=None, permutation_cv=None, na_fill=-999,
                    units=""):
        """Combared to BaseExplainerBunch defines two additional parameters:
        :param units: units to display for regression quantity
        :type units: str, optional

        """
        super().__init__(model, X, y, metric, cats, idxs, permutation_cv, na_fill)
        self.units = units
        self.is_regression = True
    
    @property
    def residuals(self):
        if not hasattr(self, '_residuals'):
            print("Calculating residuals...")
            self._residuals =  self.preds-self.y
        return self._residuals

    def plot_predicted_vs_actual(self, round=2, logs=False):
        return plotly_predicted_vs_actual(self.y, self.preds, units=self.units, round=round, logs=logs)
    
    def plot_residuals(self, vs_actual=False, round=2, ratio=False):
        return plotly_plot_residuals(self.y, self.preds, 
                                     vs_actual=vs_actual, units=self.units, round=round, ratio=ratio)
    
    def plot_residuals_vs_feature(self, col, ratio=False, round=2, dropna=True):
        assert col in self.columns, \
            f'{col} not in columns!'
        na_mask = self.X[col] != self.na_fill if dropna else np.array([True]*len(self.X))
        return plotly_residuals_vs_col(self.y[na_mask], self.preds[na_mask], self.X[col][na_mask], 
                                       ratio=ratio, units=self.units, round=round)
    


class TreeClassifierBunch(TreeExplainerBunch, ClassifierBunch):
    """TreeModelClassifierBunch inherits from both TreeModelBunch and
    ClassifierBunch.
    """


class DeepClassifierBunch(DeepExplainerBunch, ClassifierBunch):
    """DeepClassifierBunch inherits from both DeepExplainerBunch and
    ClassifierBunch.
    """


class LinearClassifierBunch(LinearExplainerBunch, ClassifierBunch):
    """LinearClassifierBunch inherits from both LinearExplainerBunch and
    ClassifierBunch.
    """


class KernelClassifierBunch(KernelExplainerBunch, ClassifierBunch):
    """KernelClassifierBunch inherits from both KernelExplainerBunch and
    ClassifierBunch.
    """


class RandomForestClassifierBunch(RandomForestExplainerBunch, ClassifierBunch):
    """RandomForestClassifierBunch inherits from both RandomForestBunch and
    ClassifierBunch.
    """


class TreeRegressionBunch(TreeExplainerBunch, RegressionBunch):
    """TreeRegressionBunch inherits from both TreeExplainertBunch and
    RegressionBunch.
    """


class LinearRegressionBunch(LinearExplainerBunch, RegressionBunch):
    """LinearRegressionBunch inherits from both LinearExplainerBunch and
    RegressionBunch.
    """



class DeepRegressionBunch(DeepExplainerBunch, RegressionBunch):
    """DeepRegressionBunch inherits from both DeepExplainerBunch and
    RegressionBunch.
    """



class KernelRegressionBunch(KernelExplainerBunch, RegressionBunch):
    """KernelRegressionBunch inherits from both KernelExplainerBunch and
    RegressionBunch.
    """


class RandomForestRegressionBunch(RandomForestExplainerBunch, RegressionBunch):
    """RandomForestClassifierBunch inherits from both RandomForestBunch and
    RegressionBunch.
    """



