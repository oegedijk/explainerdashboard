from abc import ABC, abstractmethod
import warnings

from pdpbox import pdp

from .explainer_methods import *
from .explainer_plots import *


class BaseExplainerBunch(ABC):
    def __init__(self, model,  X, y=None, metric=r2_score, 
                    cats=None, idxs=None, permutation_cv=None, na_fill=-999):
        self.model  = model
        self.X = X.reset_index(drop=True)
        if y is not None:
            self.y = y.reset_index(drop=True) 
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
        
    @classmethod
    def from_ModelBunch(cls, model_bunch, raw_data, metric,
                        index_column=None, permutation_cv=None, na_fill=-999,
                        *args, **kwargs):
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

    def __getitem__(self, index):
        idx = self.get_int_idx(index)
        if self.pred_probas is not None:
            return (self.X.iloc[idx], self.y[idx], 
                    self.pred_probas[idx], self.shap_values[idx])
        else:
            return (self.X.iloc[idx], self.y[idx], 
                    self.preds[idx], self.shap_values[idx])

    def random_index(self, y_values=None, return_str=False):
        """
        Return a random index from dataset.
        if y_values is given select an index for which y in y_values
        if return_str return str index from self.idxs
        """
        if y_values is None: y_values = self.y.unique().tolist()
        
        if y_values is None:
            potential_idxs = self.y.index
        else:
            if not isinstance(y_values, list): y_values = [y_values]
            potential_idxs = self.y[(self.y.isin(y_values))].index
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
        if not hasattr(self, '_preds'):
            print("Calculating predictions...")
            self._preds = self.model.predict(self.X)
        return self._preds
    
    @property
    def pred_probas(self):
        if not hasattr(self, '_pred_probas'):
            print("Calculating prediction probabilities...")
            assert hasattr(self.model, 'predict_proba'), \
                "model does not have a predict_proba method!"
            self._pred_probas =  self.model.predict_proba(self.X)
            if len(self._pred_probas.shape) == 2 and self._pred_probas.shape[1]==2:
                # if binary classifier, take prediction of positive class. 
                self._pred_probas = self.pred_probas[:,1]
        return self._pred_probas 
    
    def get_col_value_plus_prediction(self, index, col):
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
        if not hasattr(self, '_perm_imps'):
            print("Calculating importances...")
            if self.permutation_cv is None:
                self._perm_imps = permutation_importances(
                            self.model, self.X, self.y, self.metric)
            else:
                self._perm_imps = cv_permutation_importances(
                            self.model, self.X, self.y, self.metric,
                            cv=self.permutation_cv)
        return self._perm_imps
    
    @property
    def permutation_importances_cats(self):
        """permutation importances with categoricals grouped""" 
        if not hasattr(self, '_perm_imps_cats'):
            if self.permutation_cv is None:
                self._perm_imps_cats = permutation_importances(
                            self.model, self.X, self.y, self.metric, self.cats)
            else:
                self._perm_imps_cats = cv_permutation_importances(
                            self.model, self.X, self.y, self.metric, seld.cats,
                            cv=self.permutation_cv)
        return self._perm_imps_cats

    @property 
    def X_cats(self):
        if not hasattr(self, '_X_cats'):
            self._X_cats, self._shap_values_cats = \
                merge_categorical_shap_values(self.X, self.shap_values, self.cats)
        return self._X_cats
    
    @property 
    def columns_cats(self):
        if not hasattr(self, '_columns_cats'):
            self._columns_cats = self.X_cats.columns.tolist()
        return self._columns_cats

    @property    
    @abstractmethod
    def shap_explainer(self):
        #this property will be supplied by the inheriting classes individually
        # e.g. using KernelExplainer, TreeExplainer, DeepExplainer, etc
        raise NotImplementedError()

    @property 
    def shap_base_value(self):
        if not hasattr(self, '_shap_base_value'):
            self._shap_base_value = self.shap_explainer.expected_value
        return self._shap_base_value
    
    @property 
    def shap_values(self):
        if not hasattr(self, '_shap_values'):
            print("Calculating shap values...")
            self._shap_values = self.shap_explainer.shap_values(self.X)
        return self._shap_values

    @property 
    def shap_interaction_values(self):
        if not hasattr(self, '_shap_interaction_values'):
            print("Calculating shap interaction values...")
            self._shap_interaction_values = \
                self.shap_explainer.shap_interaction_values(self.X)
        return self._shap_interaction_values

    @property 
    def shap_values_cats(self):
        if not hasattr(self, '_shap_values_cats'):
            self._X_cats, self._shap_values_cats = \
                merge_categorical_shap_values(self.X, self.shap_values, self.cats)
        return self._shap_values_cats
    
    @property 
    def shap_interaction_values_cats(self):
        if not hasattr(self, '_shap_interaction_values_cats'):
            print("Calculating categorical shap interaction values...")
            self._shap_interaction_values_cats = \
                merge_categorical_shap_interaction_values(
                    self.X, self.X_cats, self.shap_interaction_values)
        return self._shap_interaction_values_cats
    
    @property
    def mean_abs_shap(self):
        if not hasattr(self, '_mean_abs_shap'):
            self._mean_abs_shap = mean_absolute_shap_values(
                                self.columns, self.shap_values)
        return self._mean_abs_shap
    
    @property 
    def mean_abs_shap_cats(self):
        if not hasattr(self, '_mean_abs_shap_cats'):
            self._mean_abs_shap_cats = mean_absolute_shap_values(
                                self.columns, self.shap_values, self.cats)
        return self._mean_abs_shap_cats

    
    def calculate_properties(self, include_interactions=True):
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
        shap_df = self.mean_abs_shap_cats if cats else self.mean_abs_shap
        
        if topx is None: topx = len(shap_df)
        if cutoff is None: cutoff = shap_df['MEAN_ABS_SHAP'].min()
        return shap_df[shap_df['MEAN_ABS_SHAP'] > cutoff].head(topx)
    
    def shap_top_interactions(self, col_name, topx=None, cats=False):
        if cats:
            if hasattr(self, '_shap_interaction_values'):
                col_idx = self.X_cats.columns.get_loc(col_name)
                top_interactions = self.X_cats.columns[np.argsort(-np.abs(
                        self.shap_interaction_values_cats[:, col_idx, :]).mean(0))].tolist()
            else:
                interaction_idxs = shap.common.approximate_interactions(
                    col_name, self.shap_values_cats, self.X_cats)
                top_interactions = self.X_cats.columns[interaction_idxs].tolist()
                top_interactions.insert(0, top_interactions.pop(-1)) #put col_name first

            if topx is None: topx = len(top_interactions)
            return top_interactions[:topx]
        else:
            if hasattr(self, '_shap_interaction_values'):
                col_idx = self.X.columns.get_loc(col_name)
                top_interactions = self.X.columns[np.argsort(-np.abs(
                            self.shap_interaction_values[:, col_idx, :]).mean(0))].tolist()
            else:
                interaction_idxs = shap.common.approximate_interactions(
                    col_name, self.shap_values, self.X)
                top_interactions = self.X.columns[interaction_idxs].tolist()
                top_interactions.insert(0, top_interactions.pop(-1)) #put col_name first

            if topx is None: topx = len(top_interactions)
            return top_interactions[:topx]
       
    def shap_interaction_values_by_col(self, col_name, cats=False):
        """
        returns the shap interaction values for feature col_name
        """
        if cats:
            return self.shap_interaction_values_cats[:, 
                        self.X_cats.columns.get_loc(col_name), :]
        else:
            return self.shap_interaction_values[:, 
                        self.X.columns.get_loc(col_name), :]
    
    def permutation_importances_df(self, topx=None, cutoff=None, cats=False):
        importance_df = self.permutation_importances_cats.reset_index() if cats \
                                else self.permutation_importances.reset_index()
    
        if topx is None: topx = len(importance_df)
        if cutoff is None: cutoff = importance_df.Importance.min()
        return importance_df[importance_df.Importance > cutoff].head(topx)

    def importances_df(self, type="permutation", topx=None, cutoff=None, cats=False):
        """wrapper function for mean_abs_shap_df() and perumation_importance_df()"""
        if type=='permutation':
            return self.permutation_importances_df(topx, cutoff, cats)
        elif type=='shap':
            return self.mean_abs_shap_df(topx, cutoff, cats)
        
    def contrib_df(self, idx, cats=True, topx=None, cutoff=None):
        """
        Return a contrib_df DataFrame that lists the contribution of each input
        variable for the RandomForrestClassifier predictor rf_model. 
        """  
        idx = self.get_int_idx(idx)  
        if cats:
            return get_contrib_df(self.shap_base_value, self.shap_values_cats[idx], 
                                    self.X_cats.iloc[[idx]], topx, cutoff)
        else:
            return get_contrib_df(self.shap_base_value, self.shap_values[idx], 
                                    self.X.iloc[[idx]], topx, cutoff)

    def contrib_summary_df(self, idx, cats=True, topx=None, cutoff=None):
        idx=self.get_int_idx(idx) # if passed str convert to int index
        return get_contrib_summary_df(self.contrib_df(idx, cats, topx, cutoff))

    def get_pdp_result(self, col, index=None, drop_na=True, 
                        sample=1000, num_grid_points=20):
        """
        Returns a pdpbox result for feature feature_name.

        idx: If idx provided, row idx will be put at position 0 of the results set.
        sample: Only sample random rows will be used from X
        num_grid_points: the X axis will be divided into sampled at this number
                        of quantiles

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
        return pdp_result

    def plot_importances(self, type='shap', topx=10, cats=False):
        """
        displays barchart of mean absolute shap values by feature in descending order
        
        if cats==True then add up shap values of onehotencoded columns.
        """
        importances_df = self.importances_df(type=type, topx=topx, cats=cats)
        return plotly_importances_plot(importances_df)  
    
    def plot_shap_contributions(self, index, cats=True, topx=None, cutoff=None):
        """
        Returns waterfall chart of shap values in descending absolute value
        for a particular prediction of idx.
        """ 
        idx=self.get_int_idx(index)
        contrib_df = self.contrib_df(idx, cats, topx, cutoff)
        return plotly_contribution_plot(contrib_df)

    def plot_shap_summary(self, topx=10, cats=False):
        """
        Displays all individual shap values for each feature in a horizontal 
        scatter chart in descending order by mean absolute shap value. 
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

    def plot_shap_interaction_summary(self, col, topx=10, cats=False):
        """
        Displays all individual shap interaction values for each feature in a 
        horizontal scatter chart in descending order by mean absolute shap value. 
        """
        interact_cols = self.shap_top_interactions(col, cats=cats)
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

        color_col: if color_col provided then shap values colored (blue-red) according to
        feature color_col

        highlight_idx: individual observation to be highlighed in the plot. 

        cats: if True then onehotencoded categorical variables get all
                displayed on the x axis
        """
        if cats:
            return plotly_dependence_plot(self.X_cats, self.shap_values_cats, 
                                            col, color_col, 
                                            highlight_idx=highlight_idx)
        else:
            return plotly_dependence_plot(self.X, self.shap_values, 
                                            col, color_col, 
                                            highlight_idx=highlight_idx)

    def plot_shap_interaction_dependence(self, col, interact_col, highlight_idx=None, cats=False):
        return plotly_dependence_plot(self.X_cats if cats else self.X, 
                self.shap_interaction_values_by_col(col, cats), 
                interact_col, col, highlight_idx=highlight_idx)

    def plot_pdp(self, col, index=None, drop_na=True, sample=1000, num_grid_lines=100, num_grid_points=20):
        pdp_result = self.get_pdp_result(col, index, 
                            drop_na=True, sample=sample, 
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

    def is_classifier(self):
        return False


class TreeModelBunch(BaseExplainerBunch):
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


class ClassifierBunch(BaseExplainerBunch):
    def __init__(self, model,  X, y=None, metric=roc_auc_score, 
                    cats=None, idxs=None, permutation_cv=None, na_fill=-999, 
                    labels=['0', '1']):
        super().__init__(model, X, y, metric, cats, idxs, permutation_cv, na_fill)
        self.labels = labels or ['0', '1']

    @property 
    def shap_base_value(self):
        if not hasattr(self, '_shap_base_value'):
            try:
                self._shap_base_value = self.shap_explainer.expected_value[1]
            except:
                self._shap_base_value = self.shap_explainer.expected_value
        return self._shap_base_value
    
    @property 
    def shap_values(self):
        if not hasattr(self, '_shap_values'):
            print("Calculating shap values...")
            try:
                self._shap_values = self.shap_explainer.shap_values(self.X)[1]
                assert self._shap_values.shape == (len(self.X), len(self.X.columns))
            except:
                self._shap_values = self.shap_explainer.shap_values(self.X)
        return self._shap_values

    @property 
    def shap_interaction_values(self):
        if not hasattr(self, '_shap_interaction_values'):
            print("Calculating shap interaction values...")
            self._shap_interaction_values = normalize_shap_interaction_values(
                self.shap_explainer.shap_interaction_values(self.X)[1],
                self.shap_values)
        return self._shap_interaction_values

    def random_index(self, y_values=None, return_str=False, 
                    pred_proba_min=None, pred_proba_max=None):
        """
        Return a random index from dataset.
        if y_values is given select an index for which y in y_values
        if return_str return str index from self.idxs
        """
        if y_values is None and pred_proba_min is None and pred_proba_max is None:
            potential_idxs = self.y.index
        else:
            if y_values is None: y_values = self.y.unique().tolist()
            if not isinstance(y_values, list): y_values = [y_values]
            if pred_proba_min is None: pred_proba_min = self.pred_probas.min()
            if pred_proba_max is None: pred_proba_max = self.pred_probas.max()
            potential_idxs = self.y[(self.y.isin(y_values)) & 
                            (self.pred_probas >= pred_proba_min) & 
                            (self.pred_probas <= pred_proba_max)].index
        if not potential_idxs.empty:
            idx = np.random.choice(potential_idxs)
        else: 
            return None
        if return_str:
            assert self.idxs is not None, \
                "no self.idxs property found..."
            return self.idxs[idx]
        return idx

    def precision_df(self, bin_size=None, quantiles=None):
        assert self.pred_probas is not None
        if bin_size is None and quantiles is None:
            bin_size=0.1 # defaults to bin_size=0.1
        return get_precision_df(self.pred_probas, self.y.values, bin_size, quantiles)

    def plot_precision(self, bin_size=None, quantiles=None, cutoff=0.5):
        """plots predicted probability on the x-axis
        binned by bin_size, and observed precision (fraction of actual positive
        cases) on the y-axis"""

        if bin_size is None and quantiles is None:
            bin_size=0.1 # defaults to bin_size=0.1
        precision_df = self.precision_df(bin_size=bin_size, quantiles=quantiles)
        return plotly_precision_plot(precision_df, cutoff=cutoff)

    def plot_confusion_matrix(self, cutoff=0.5, normalized=False):
        """plots a standard 2d confusion 
        matrix, depending on model cutoff. If normalized display percentage
        otherwise counts."""
        return plotly_confusion_matrix(
                self.y, self.pred_probas,
                cutoff=cutoff, normalized=normalized, 
                labels=self.labels)

    def plot_roc_auc(self, cutoff=0.5):
        """plots ROC_AUC curve. The TPR and FPR of a particular
            cutoff is displayed in crosshairs."""
        return plotly_roc_auc_curve(self.y, self.pred_probas, cutoff=cutoff)

    def plot_pr_auc(self, cutoff=0.5):
        """plots PR_AUC curve. the precision and recall of particular
            cutoff is displayed in crosshairs."""
        return plotly_pr_auc_curve(self.y, self.pred_probas, cutoff=cutoff)

    def is_classifier(self):
        return True

    def calculate_properties(self, include_interactions=True):
        super().calculate_properties(include_interactions)


class RandomForestBunch(TreeModelBunch):
    """
    Shadow Trees are generated by the method ShadowDecTree() in de dtreeviz 
    package, that make it easy to navigate the nodes of an individual sklearn
    style DecisionTree. 

    These are used to output the path an individual predictions takes through
    a particular DecisionTree. 
    """
    @property
    def shadow_trees(self):
        if not hasattr(self, '_shadow_trees'):
            print("Generating shadow trees...")
            self._shadow_trees = get_shadow_trees(self.model, self.X, self.y)
        return self._shadow_trees

    def shadowtree_df(self, tree_idx, idx):
        """returns a pd.DataFrame with all decision nodes of a particular
                        tree (indexed by tree_idx) for a particular observation 
                        (indexed by idx)"""
        assert tree_idx >= 0 and tree_idx < len(self.shadow_trees), \
            f"tree index {tree_idx} outside 0 and number of trees ({len(self.shadow_trees)}) range"
        idx=self.get_int_idx(idx)
        assert idx >= 0 and idx < len(self.X), \
            f"=index {idx} outside 0 and size of X ({len(self.X)}) range"

        return get_shadowtree_df(self.shadow_trees[tree_idx], self.X.iloc[idx])

    def shadowtree_df_summary(self, tree_idx, idx):
        """formats shadowtree_df in a slightly more human 
                readable format."""
        idx=self.get_int_idx(idx)
        return shadowtree_df_summary(self.shadowtree_df(tree_idx, idx))

    def plot_trees(self, idx):
        """returns a plotly barchart with the values of the predictions
                of each individual tree for observation idx"""
        idx=self.get_int_idx(idx)
        return plotly_tree_predictions(self.model, self.X.iloc[[idx]])

    def calculate_properties(self, include_interactions=True):
        _ = (self.shadow_trees)
        super().calculate_properties(include_interactions)

class TreeModelClassifierBunch(ClassifierBunch, TreeModelBunch):
    def calculate_properties(self, include_interactions=True):
        super().calculate_properties(include_interactions)


class RandomForestClassifierBunch(ClassifierBunch, RandomForestBunch):
    def calculate_properties(self, include_interactions=True):
        super().calculate_properties(include_interactions)


    


    