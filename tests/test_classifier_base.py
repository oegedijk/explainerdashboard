import unittest

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

import pdpbox
import plotly.graph_objects as go

from explainerdashboard.explainers import ClassifierExplainer
from explainerdashboard.datasets import titanic_survive, titanic_names


class ClassifierBaseExplainerTests(unittest.TestCase):
    def setUp(self):
        X_train, y_train, X_test, y_test = titanic_survive()
        train_names, test_names = titanic_names()

        model = RandomForestClassifier(n_estimators=5, max_depth=2)
        model.fit(X_train, y_train)

        self.explainer = ClassifierExplainer(
                            model, X_test, y_test, roc_auc_score, 
                            shap='tree',
                            cats=['Sex', 'Cabin', 'Embarked'],
                            labels=['Not survived', 'Survived'],
                            idxs=test_names)

    def test_explainer_len(self):
        self.assertEqual(len(self.explainer), len(titanic_survive()[2]))

    def test_int_idx(self):
        self.assertEqual(self.explainer.get_int_idx(titanic_names()[1][0]), 0)

    def test_random_index(self):
        self.assertIsInstance(self.explainer.random_index(), int)
        self.assertIsInstance(self.explainer.random_index(return_str=True), str)

    def test_preds(self):
        self.assertIsInstance(self.explainer.preds, np.ndarray)

    def test_pred_percentiles(self):
        self.assertIsInstance(self.explainer.pred_percentiles, np.ndarray)

    def test_columns_ranked_by_shap(self):
        self.assertIsInstance(self.explainer.columns_ranked_by_shap(), list)
        self.assertIsInstance(self.explainer.columns_ranked_by_shap(cats=True), list)

    def test_equivalent_col(self):
        self.assertEqual(self.explainer.equivalent_col("Sex_female"), "Sex")
        self.assertEqual(self.explainer.equivalent_col("Sex"), "Sex_female")
        self.assertIsNone(self.explainer.equivalent_col("random"))

    def test_get_col(self):
        self.assertIsInstance(self.explainer.get_col("Sex"), pd.Series)
        self.assertEqual(self.explainer.get_col("Sex").dtype, "object")

        self.assertIsInstance(self.explainer.get_col("Age"), pd.Series)
        self.assertEqual(self.explainer.get_col("Age").dtype, np.float)

    def test_permutation_importances(self):
        self.assertIsInstance(self.explainer.permutation_importances, pd.DataFrame)
        self.assertIsInstance(self.explainer.permutation_importances_cats, pd.DataFrame)
        
    def test_X_cats(self):
        self.assertIsInstance(self.explainer.X_cats, pd.DataFrame)

    def test_columns_cats(self):
        self.assertIsInstance(self.explainer.columns_cats, list)

    def test_metrics(self):
        self.assertIsInstance(self.explainer.metrics(), dict)
        self.assertIsInstance(self.explainer.metrics_markdown(), str)

    def test_mean_abs_shap_df(self):
        self.assertIsInstance(self.explainer.mean_abs_shap_df(), pd.DataFrame)

    def test_top_interactions(self):
        self.assertIsInstance(self.explainer.shap_top_interactions("Age"), list)
        self.assertIsInstance(self.explainer.shap_top_interactions("Age", topx=4), list)
        self.assertIsInstance(self.explainer.shap_top_interactions("Age", cats=True), list)
        self.assertIsInstance(self.explainer.shap_top_interactions("Sex", cats=True), list)

    def test_permutation_importances_df(self):
        self.assertIsInstance(self.explainer.permutation_importances_df(), pd.DataFrame)
        self.assertIsInstance(self.explainer.permutation_importances_df(topx=3), pd.DataFrame)
        self.assertIsInstance(self.explainer.permutation_importances_df(cats=True), pd.DataFrame)
        self.assertIsInstance(self.explainer.permutation_importances_df(cutoff=0.01), pd.DataFrame)

    def test_contrib_df(self):
        self.assertIsInstance(self.explainer.contrib_df(0), pd.DataFrame)
        self.assertIsInstance(self.explainer.contrib_df(0, cats=False), pd.DataFrame)
        self.assertIsInstance(self.explainer.contrib_df(0, topx=3), pd.DataFrame)

    def test_contrib_summary_df(self):
        self.assertIsInstance(self.explainer.contrib_summary_df(0), pd.DataFrame)
        self.assertIsInstance(self.explainer.contrib_summary_df(0, cats=False), pd.DataFrame)
        self.assertIsInstance(self.explainer.contrib_summary_df(0, topx=3), pd.DataFrame)
        self.assertIsInstance(self.explainer.contrib_summary_df(0, round=3), pd.DataFrame)

    def test_shap_base_value(self):
        self.assertIsInstance(self.explainer.shap_base_value, (np.floating, float))

    def test_shap_values_shape(self):
        self.assertTrue(self.explainer.shap_values.shape == (len(self.explainer), len(self.explainer.columns)))

    def test_shap_values(self):
        self.assertIsInstance(self.explainer.shap_values, np.ndarray)
        self.assertIsInstance(self.explainer.shap_values_cats, np.ndarray)

    def test_shap_interaction_values(self):
        self.assertIsInstance(self.explainer.shap_interaction_values, np.ndarray)
        self.assertIsInstance(self.explainer.shap_interaction_values_cats, np.ndarray)

    def test_mean_abs_shap(self):
        self.assertIsInstance(self.explainer.mean_abs_shap, pd.DataFrame)
        self.assertIsInstance(self.explainer.mean_abs_shap_cats, pd.DataFrame)

    def test_calculate_properties(self):
        self.explainer.calculate_properties()

    def test_shap_interaction_values_by_col(self):
        self.assertIsInstance(self.explainer.shap_interaction_values_by_col("Age"), np.ndarray)
        self.assertEquals(self.explainer.shap_interaction_values_by_col("Age").shape, 
                        self.explainer.shap_values.shape)
        self.assertEquals(self.explainer.shap_interaction_values_by_col("Age", cats=True).shape, 
                        self.explainer.shap_values_cats.shape)

    def test_pdp_result(self):
        self.assertIsInstance(self.explainer.get_pdp_result("Age"), pdpbox.pdp.PDPIsolate)
        self.assertIsInstance(self.explainer.get_pdp_result("Sex"), pdpbox.pdp.PDPIsolate)
        self.assertIsInstance(self.explainer.get_pdp_result("Age", index=0), pdpbox.pdp.PDPIsolate)
        self.assertIsInstance(self.explainer.get_pdp_result("Sex", index=0), pdpbox.pdp.PDPIsolate)

    def test_get_dfs(self):
        cols_df, shap_df, contribs_df = self.explainer.get_dfs()
        self.assertIsInstance(cols_df, pd.DataFrame)
        self.assertIsInstance(shap_df, pd.DataFrame)
        self.assertIsInstance(contribs_df, pd.DataFrame)

    def test_plot_importances(self):
        fig = self.explainer.plot_importances()
        self.assertIsInstance(fig, go.Figure)

        fig = self.explainer.plot_importances(kind='permutation')
        self.assertIsInstance(fig, go.Figure)

        fig = self.explainer.plot_importances(topx=3)
        self.assertIsInstance(fig, go.Figure)

        fig = self.explainer.plot_importances(cats=True)
        self.assertIsInstance(fig, go.Figure)

    def test_plot_interactions(self):
        fig = self.explainer.plot_interactions("Age")
        self.assertIsInstance(fig, go.Figure)

        fig = self.explainer.plot_interactions("Sex_female")
        self.assertIsInstance(fig, go.Figure)

        fig = self.explainer.plot_interactions("Age")
        self.assertIsInstance(fig, go.Figure)

        fig = self.explainer.plot_interactions("Sex")
        self.assertIsInstance(fig, go.Figure)

    def test_plot_shap_contributions(self):
        fig = self.explainer.plot_shap_contributions(0)
        self.assertIsInstance(fig, go.Figure)

        fig = self.explainer.plot_shap_contributions(0, cats=False)
        self.assertIsInstance(fig, go.Figure)

        fig = self.explainer.plot_shap_contributions(0, topx=3)
        self.assertIsInstance(fig, go.Figure)

        fig = self.explainer.plot_shap_contributions(0, cutoff=0.05)
        self.assertIsInstance(fig, go.Figure)

    def test_plot_shap_summary(self):
        fig = self.explainer.plot_shap_summary()
        self.assertIsInstance(fig, go.Figure)

        fig = self.explainer.plot_shap_summary(topx=3)
        self.assertIsInstance(fig, go.Figure)

        fig = self.explainer.plot_shap_summary(cats=True)
        self.assertIsInstance(fig, go.Figure)

    def test_plot_shap_interaction_summary(self):
        fig = self.explainer.plot_shap_interaction_summary("Age")
        self.assertIsInstance(fig, go.Figure)

        fig = self.explainer.plot_shap_interaction_summary("Age", topx=3)
        self.assertIsInstance(fig, go.Figure)

        fig = self.explainer.plot_shap_interaction_summary("Age")
        self.assertIsInstance(fig, go.Figure)

        fig = self.explainer.plot_shap_interaction_summary("Sex_female", topx=3)
        self.assertIsInstance(fig, go.Figure)

        fig = self.explainer.plot_shap_interaction_summary("Sex")
        self.assertIsInstance(fig, go.Figure)

    def test_plot_shap_dependence(self):
        fig = self.explainer.plot_shap_dependence("Age")
        self.assertIsInstance(fig, go.Figure)

        fig = self.explainer.plot_shap_dependence("Sex_female")
        self.assertIsInstance(fig, go.Figure)

        fig = self.explainer.plot_shap_dependence("Age", "Sex")
        self.assertIsInstance(fig, go.Figure)

        fig = self.explainer.plot_shap_dependence("Sex_female", "Age")
        self.assertIsInstance(fig, go.Figure)

        fig = self.explainer.plot_shap_dependence("Age", highlight_idx=0)
        self.assertIsInstance(fig, go.Figure)

        fig = self.explainer.plot_shap_dependence("Sex", highlight_idx=0)
        self.assertIsInstance(fig, go.Figure)

    def test_plot_shap_interaction(self):
        fig = self.explainer.plot_shap_dependence("Age", "Sex_female")
        self.assertIsInstance(fig, go.Figure)

        fig = self.explainer.plot_shap_dependence("Sex_female", "Age")
        self.assertIsInstance(fig, go.Figure)

        fig = self.explainer.plot_shap_dependence("Sex", "Age")
        self.assertIsInstance(fig, go.Figure)

        fig = self.explainer.plot_shap_dependence("Age", "Sex")
        self.assertIsInstance(fig, go.Figure)

        fig = self.explainer.plot_shap_dependence("Age", "Sex_female", highlight_idx=0)
        self.assertIsInstance(fig, go.Figure)

    def test_plot_pdp(self):
        fig = self.explainer.plot_pdp("Age")
        self.assertIsInstance(fig, go.Figure)

        fig = self.explainer.plot_pdp("Sex")
        self.assertIsInstance(fig, go.Figure)

        fig = self.explainer.plot_pdp("Sex", index=0)
        self.assertIsInstance(fig, go.Figure)

        fig = self.explainer.plot_pdp("Age", index=0)
        self.assertIsInstance(fig, go.Figure)




