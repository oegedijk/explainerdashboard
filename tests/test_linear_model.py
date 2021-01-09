import unittest

import pandas as pd
import numpy as np

from sklearn.metrics import r2_score, roc_auc_score

import shap
import plotly.graph_objects as go

from sklearn.linear_model import LinearRegression, LogisticRegression

from explainerdashboard.explainers import RegressionExplainer, ClassifierExplainer
from explainerdashboard.datasets import titanic_fare, titanic_survive, titanic_names


class LinearRegressionTests(unittest.TestCase):
    def setUp(self):
        X_train, y_train, X_test, y_test = titanic_fare()
        self.test_len = len(X_test)

        train_names, test_names = titanic_names()
        _, self.names = titanic_names()


        model = LinearRegression()
        model.fit(X_train, y_train)
        self.explainer = RegressionExplainer(model, X_test, y_test, r2_score, 
                                        shap='linear', 
                                        cats=[{'Gender': ['Sex_female', 'Sex_male', 'Sex_nan']}, 
                                                'Deck', 'Embarked'],
                                        idxs=test_names, units="$")

    def test_explainer_len(self):
        self.assertEqual(len(self.explainer), self.test_len)

    def test_int_idx(self):
        self.assertEqual(self.explainer.get_int_idx(self.names[0]), 0)

    def test_random_index(self):
        self.assertIsInstance(self.explainer.random_index(), int)
        self.assertIsInstance(self.explainer.random_index(return_str=True), str)

    def test_preds(self):
        self.assertIsInstance(self.explainer.preds, np.ndarray)

    def test_pred_percentiles(self):
        self.assertIsInstance(self.explainer.pred_percentiles, np.ndarray)

    def test_permutation_importances(self):
        self.assertIsInstance(self.explainer.permutation_importances, pd.DataFrame)
        self.assertIsInstance(self.explainer.permutation_importances_cats, pd.DataFrame)

    def test_metrics(self):
        self.assertIsInstance(self.explainer.metrics(), dict)
        self.assertIsInstance(self.explainer.metrics_descriptions(), dict)

    def test_mean_abs_shap_df(self):
        self.assertIsInstance(self.explainer.mean_abs_shap_df(), pd.DataFrame)

    def test_top_interactions(self):
        self.assertIsInstance(self.explainer.shap_top_interactions("Age"), list)
        self.assertIsInstance(self.explainer.shap_top_interactions("Age", topx=4), list)
        self.assertIsInstance(self.explainer.shap_top_interactions("Age", cats=True), list)
        self.assertIsInstance(self.explainer.shap_top_interactions("Gender", cats=True), list)

    def test_contrib_df(self):
        self.assertIsInstance(self.explainer.contrib_df(0), pd.DataFrame)
        self.assertIsInstance(self.explainer.contrib_df(0, cats=False), pd.DataFrame)
        self.assertIsInstance(self.explainer.contrib_df(0, topx=3), pd.DataFrame)

    def test_shap_base_value(self):
        self.assertIsInstance(self.explainer.shap_base_value, (np.floating, float))

    def test_shap_values_shape(self):
        self.assertTrue(self.explainer.shap_values.shape == (len(self.explainer), len(self.explainer.columns)))

    def test_shap_values(self):
        self.assertIsInstance(self.explainer.shap_values, np.ndarray)
        self.assertIsInstance(self.explainer.shap_values_cats, np.ndarray)

    def test_mean_abs_shap(self):
        self.assertIsInstance(self.explainer.mean_abs_shap, pd.DataFrame)
        self.assertIsInstance(self.explainer.mean_abs_shap_cats, pd.DataFrame)

    def test_calculate_properties(self):
        self.explainer.calculate_properties(include_interactions=False)

    def test_pdp_df(self):
        self.assertIsInstance(self.explainer.pdp_df("Age"), pd.DataFrame)
        self.assertIsInstance(self.explainer.pdp_df("Gender"), pd.DataFrame)
        self.assertIsInstance(self.explainer.pdp_df("Deck"), pd.DataFrame)
        self.assertIsInstance(self.explainer.pdp_df("Age", index=0), pd.DataFrame)
        self.assertIsInstance(self.explainer.pdp_df("Gender", index=0), pd.DataFrame)

    def test_get_dfs(self):
        cols_df, shap_df, contribs_df = self.explainer.get_dfs()
        self.assertIsInstance(cols_df, pd.DataFrame)
        self.assertIsInstance(shap_df, pd.DataFrame)
        self.assertIsInstance(contribs_df, pd.DataFrame)



class LogisticRegressionTests(unittest.TestCase):
    def setUp(self):
        X_train, y_train, X_test, y_test = titanic_survive()
        train_names, test_names = titanic_names()

        model = LogisticRegression()
        model.fit(X_train, y_train)

        self.explainer = ClassifierExplainer(
                            model, X_test, y_test, roc_auc_score, 
                            shap='linear',
                            cats=['Sex', 'Deck', 'Embarked'],
                            labels=['Not survived', 'Survived'],
                            idxs=test_names)

    def test_preds(self):
        self.assertIsInstance(self.explainer.preds, np.ndarray)

    def test_pred_percentiles(self):
        self.assertIsInstance(self.explainer.pred_percentiles, np.ndarray)

    def test_columns_ranked_by_shap(self):
        self.assertIsInstance(self.explainer.columns_ranked_by_shap(), list)
        self.assertIsInstance(self.explainer.columns_ranked_by_shap(cats=True), list)

    def test_permutation_importances(self):
        self.assertIsInstance(self.explainer.permutation_importances, pd.DataFrame)
        self.assertIsInstance(self.explainer.permutation_importances_cats, pd.DataFrame)

    def test_metrics(self):
        self.assertIsInstance(self.explainer.metrics(), dict)
        self.assertIsInstance(self.explainer.metrics_descriptions(), dict)

    def test_mean_abs_shap_df(self):
        self.assertIsInstance(self.explainer.mean_abs_shap_df(), pd.DataFrame)

    def test_contrib_df(self):
        self.assertIsInstance(self.explainer.contrib_df(0), pd.DataFrame)
        self.assertIsInstance(self.explainer.contrib_df(0, cats=False), pd.DataFrame)
        self.assertIsInstance(self.explainer.contrib_df(0, topx=3), pd.DataFrame)

    def test_shap_base_value(self):
        self.assertIsInstance(self.explainer.shap_base_value, (np.floating, float))

    def test_shap_values_shape(self):
        self.assertTrue(self.explainer.shap_values.shape == (len(self.explainer), len(self.explainer.columns)))

    def test_shap_values(self):
        self.assertIsInstance(self.explainer.shap_values, np.ndarray)
        self.assertIsInstance(self.explainer.shap_values_cats, np.ndarray)

    def test_mean_abs_shap(self):
        self.assertIsInstance(self.explainer.mean_abs_shap, pd.DataFrame)
        self.assertIsInstance(self.explainer.mean_abs_shap_cats, pd.DataFrame)

    def test_calculate_properties(self):
        self.explainer.calculate_properties(include_interactions=False)

    def test_pdp_df(self):
        self.assertIsInstance(self.explainer.pdp_df("Age"), pd.DataFrame)
        self.assertIsInstance(self.explainer.pdp_df("Sex"), pd.DataFrame)
        self.assertIsInstance(self.explainer.pdp_df("Deck"), pd.DataFrame)
        self.assertIsInstance(self.explainer.pdp_df("Age", index=0), pd.DataFrame)
        self.assertIsInstance(self.explainer.pdp_df("Sex", index=0), pd.DataFrame)

    def test_pos_label(self):
        self.explainer.pos_label = 1
        self.explainer.pos_label = "Not survived"
        self.assertIsInstance(self.explainer.pos_label, int)
        self.assertIsInstance(self.explainer.pos_label_str, str)
        self.assertEqual(self.explainer.pos_label, 0)
        self.assertEqual(self.explainer.pos_label_str, "Not survived")

    def test_get_prop_for_label(self):
        self.explainer.pos_label = 1
        tmp = self.explainer.pred_percentiles
        self.explainer.pos_label = 0
        self.assertTrue(np.alltrue(self.explainer.get_prop_for_label("pred_percentiles", 1)==tmp))

    def test_pred_probas(self):
        self.assertIsInstance(self.explainer.pred_probas, np.ndarray)

    def test_metrics(self):
        self.assertIsInstance(self.explainer.metrics(), dict)
        self.assertIsInstance(self.explainer.metrics(cutoff=0.9), dict)

    def test_precision_df(self):
        self.assertIsInstance(self.explainer.precision_df(), pd.DataFrame)
        self.assertIsInstance(self.explainer.precision_df(multiclass=True), pd.DataFrame)
        self.assertIsInstance(self.explainer.precision_df(quantiles=4), pd.DataFrame)

    def test_lift_curve_df(self):
        self.assertIsInstance(self.explainer.lift_curve_df(), pd.DataFrame)

    def test_prediction_result_markdown(self):
        self.assertIsInstance(self.explainer.prediction_result_markdown(0), str)


class LogisticRegressionKernelTests(unittest.TestCase):
    def setUp(self):
        X_train, y_train, X_test, y_test = titanic_survive()
        train_names, test_names = titanic_names()

        model = LogisticRegression()
        model.fit(X_train, y_train)

        self.explainer = ClassifierExplainer(
                            model, X_test, y_test, 
                            shap='kernel', model_output='probability', 
                            X_background=shap.sample(X_train, 5),
                            cats=[{'Gender': ['Sex_female', 'Sex_male', 'Sex_nan']}, 
                                                'Deck', 'Embarked'],
                            labels=['Not survived', 'Survived'],
                            idxs=test_names)

    def test_shap_values(self):
        self.assertIsInstance(self.explainer.shap_base_value, (np.floating, float))
        self.assertTrue(self.explainer.shap_values.shape == (len(self.explainer), len(self.explainer.columns)))
        self.assertIsInstance(self.explainer.shap_values, np.ndarray)
        self.assertIsInstance(self.explainer.shap_values_cats, np.ndarray)

    

