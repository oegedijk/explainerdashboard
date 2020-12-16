import unittest

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

import pdpbox
import plotly.graph_objects as go

from explainerdashboard.explainers import RegressionExplainer
from explainerdashboard.datasets import titanic_fare, titanic_names


class RegressionBaseExplainerTests(unittest.TestCase):
    def setUp(self):
        X_train, y_train, X_test, y_test = titanic_fare()
        self.test_len = len(X_test)

        train_names, test_names = titanic_names()
        _, self.names = titanic_names()

        model = RandomForestRegressor(n_estimators=5, max_depth=2)
        model.fit(X_train, y_train)

        self.explainer = RegressionExplainer(
                            model, X_test, y_test, r2_score,
                            cats=[{'Gender': ['Sex_female', 'Sex_male', 'Sex_nan']}, 
                                                'Deck', 'Embarked'],
                            idxs=test_names, target='Fare', units='$')

    def test_explainer_len(self):
        self.assertEqual(len(self.explainer), self.test_len)

    def test_int_idx(self):
        self.assertEqual(self.explainer.get_int_idx(self.names[0]), 0)

    def test_random_index(self):
        self.assertIsInstance(self.explainer.random_index(), int)
        self.assertIsInstance(self.explainer.random_index(return_str=True), str)

    def test_row_from_input(self):
        input_row = self.explainer.get_row_from_input(
            self.explainer.X.iloc[[0]].values.tolist())
        self.assertIsInstance(input_row, pd.DataFrame)

        input_row = self.explainer.get_row_from_input(
            self.explainer.X_cats.iloc[[0]].values.tolist())
        self.assertIsInstance(input_row, pd.DataFrame)

        input_row = self.explainer.get_row_from_input(
            self.explainer.X_cats
            [self.explainer.columns_ranked_by_shap(cats=True)]
            .iloc[[0]].values.tolist(), ranked_by_shap=True)
        self.assertIsInstance(input_row, pd.DataFrame)
        
        input_row = self.explainer.get_row_from_input(
            self.explainer.X
            [self.explainer.columns_ranked_by_shap(cats=False)]
            .iloc[[0]].values.tolist(), ranked_by_shap=True)
        self.assertIsInstance(input_row, pd.DataFrame)

    def test_prediction_result_df(self):
        df = self.explainer.prediction_result_df(0)
        self.assertIsInstance(df, pd.DataFrame)

    def test_preds(self):
        self.assertIsInstance(self.explainer.preds, np.ndarray)

    def test_pred_percentiles(self):
        self.assertIsInstance(self.explainer.pred_percentiles, np.ndarray)

    def test_columns_ranked_by_shap(self):
        self.assertIsInstance(self.explainer.columns_ranked_by_shap(), list)
        self.assertIsInstance(self.explainer.columns_ranked_by_shap(cats=True), list)

    def test_equivalent_col(self):
        self.assertEqual(self.explainer.equivalent_col("Sex_female"), "Gender")
        self.assertEqual(self.explainer.equivalent_col("Gender"), "Sex_female")
        self.assertIsNone(self.explainer.equivalent_col("random"))

    def test_get_col(self):
        self.assertIsInstance(self.explainer.get_col("Gender"), pd.Series)
        self.assertEqual(self.explainer.get_col("Gender").dtype, "object")

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
        self.assertIsInstance(self.explainer.metrics_descriptions(), dict)

    def test_mean_abs_shap_df(self):
        self.assertIsInstance(self.explainer.mean_abs_shap_df(), pd.DataFrame)

    def test_top_interactions(self):
        self.assertIsInstance(self.explainer.shap_top_interactions("Age"), list)
        self.assertIsInstance(self.explainer.shap_top_interactions("Age", topx=4), list)
        self.assertIsInstance(self.explainer.shap_top_interactions("Age", cats=True), list)
        self.assertIsInstance(self.explainer.shap_top_interactions("Gender", cats=True), list)

    def test_permutation_importances_df(self):
        self.assertIsInstance(self.explainer.permutation_importances_df(), pd.DataFrame)
        self.assertIsInstance(self.explainer.permutation_importances_df(topx=3), pd.DataFrame)
        self.assertIsInstance(self.explainer.permutation_importances_df(cats=True), pd.DataFrame)
        self.assertIsInstance(self.explainer.permutation_importances_df(cutoff=0.01), pd.DataFrame)

    def test_contrib_df(self):
        self.assertIsInstance(self.explainer.contrib_df(0), pd.DataFrame)
        self.assertIsInstance(self.explainer.contrib_df(0, cats=False), pd.DataFrame)
        self.assertIsInstance(self.explainer.contrib_df(0, topx=3), pd.DataFrame)
        self.assertIsInstance(self.explainer.contrib_df(0, sort='high-to-low'), pd.DataFrame)
        self.assertIsInstance(self.explainer.contrib_df(0, sort='low-to-high'), pd.DataFrame)
        self.assertIsInstance(self.explainer.contrib_df(0, sort='importance'), pd.DataFrame)
        self.assertIsInstance(self.explainer.contrib_df(X_row=self.explainer.X.iloc[[0]]), pd.DataFrame)
        self.assertIsInstance(self.explainer.contrib_df(X_row=self.explainer.X_cats.iloc[[0]]), pd.DataFrame)


    def test_contrib_summary_df(self):
        self.assertIsInstance(self.explainer.contrib_summary_df(0), pd.DataFrame)
        self.assertIsInstance(self.explainer.contrib_summary_df(0, cats=False), pd.DataFrame)
        self.assertIsInstance(self.explainer.contrib_summary_df(0, topx=3), pd.DataFrame)
        self.assertIsInstance(self.explainer.contrib_summary_df(0, round=3), pd.DataFrame)
        self.assertIsInstance(self.explainer.contrib_summary_df(0, sort='high-to-low'), pd.DataFrame)
        self.assertIsInstance(self.explainer.contrib_summary_df(0, sort='low-to-high'), pd.DataFrame)
        self.assertIsInstance(self.explainer.contrib_summary_df(0, sort='importance'), pd.DataFrame)
        self.assertIsInstance(self.explainer.contrib_summary_df(X_row=self.explainer.X.iloc[[0]]), pd.DataFrame)
        self.assertIsInstance(self.explainer.contrib_summary_df(X_row=self.explainer.X_cats.iloc[[0]]), pd.DataFrame)


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
        self.assertEqual(self.explainer.shap_interaction_values_by_col("Age").shape,
                        self.explainer.shap_values.shape)
        self.assertEqual(self.explainer.shap_interaction_values_by_col("Age", cats=True).shape,
                        self.explainer.shap_values_cats.shape)

    def test_pdp_result(self):
        self.assertIsInstance(self.explainer.get_pdp_result("Age"), pdpbox.pdp.PDPIsolate)
        self.assertIsInstance(self.explainer.get_pdp_result("Gender"), pdpbox.pdp.PDPIsolate)
        self.assertIsInstance(self.explainer.get_pdp_result("Age", index=0), pdpbox.pdp.PDPIsolate)
        self.assertIsInstance(self.explainer.get_pdp_result("Gender", index=0), pdpbox.pdp.PDPIsolate)
        self.assertIsInstance(self.explainer.get_pdp_result("Age", X_row=self.explainer.X.iloc[[0]]), pdpbox.pdp.PDPIsolate)
        self.assertIsInstance(self.explainer.get_pdp_result("Gender", X_row=self.explainer.X_cats.iloc[[0]]), pdpbox.pdp.PDPIsolate)

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

        fig = self.explainer.plot_interactions("Gender")
        self.assertIsInstance(fig, go.Figure)

    def test_plot_shap_interactions(self):
        fig = self.explainer.plot_shap_contributions(0)
        self.assertIsInstance(fig, go.Figure)

        fig = self.explainer.plot_shap_contributions(0, cats=False)
        self.assertIsInstance(fig, go.Figure)

        fig = self.explainer.plot_shap_contributions(0, topx=3)
        self.assertIsInstance(fig, go.Figure)

        fig = self.explainer.plot_shap_contributions(0, cutoff=0.05)
        self.assertIsInstance(fig, go.Figure)

        fig = self.explainer.plot_shap_contributions(0, sort='high-to-low')
        self.assertIsInstance(fig, go.Figure)

        fig = self.explainer.plot_shap_contributions(0, sort='low-to-high')
        self.assertIsInstance(fig, go.Figure)

        fig = self.explainer.plot_shap_contributions(0, sort='importance')
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

        fig = self.explainer.plot_shap_interaction_summary("Age", cats=True)
        self.assertIsInstance(fig, go.Figure)

        fig = self.explainer.plot_shap_interaction_summary("Sex_female", topx=3)
        self.assertIsInstance(fig, go.Figure)

        fig = self.explainer.plot_shap_interaction_summary("Gender", cats=True)
        self.assertIsInstance(fig, go.Figure)

    def test_plot_shap_dependence(self):
        fig = self.explainer.plot_shap_dependence("Age")
        self.assertIsInstance(fig, go.Figure)

        fig = self.explainer.plot_shap_dependence("Gender")
        self.assertIsInstance(fig, go.Figure)

        fig = self.explainer.plot_shap_dependence("Age", "Gender")
        self.assertIsInstance(fig, go.Figure)

        fig = self.explainer.plot_shap_dependence("Sex_female", "Age")
        self.assertIsInstance(fig, go.Figure)

        fig = self.explainer.plot_shap_dependence("Age", highlight_index=0)
        self.assertIsInstance(fig, go.Figure)

        fig = self.explainer.plot_shap_dependence("Gender", highlight_index=0)
        self.assertIsInstance(fig, go.Figure)

    def test_plot_shap_contributions(self):
        fig = self.explainer.plot_shap_contributions(0)
        self.assertIsInstance(fig, go.Figure)

        fig = self.explainer.plot_shap_contributions(0, cats=False)
        self.assertIsInstance(fig, go.Figure)

        fig = self.explainer.plot_shap_contributions(0, topx=3)
        self.assertIsInstance(fig, go.Figure)

        fig = self.explainer.plot_shap_contributions(0, sort='high-to-low')
        self.assertIsInstance(fig, go.Figure)

        fig = self.explainer.plot_shap_contributions(0, sort='low-to-high')
        self.assertIsInstance(fig, go.Figure)

        fig = self.explainer.plot_shap_contributions(0, sort='importance')
        self.assertIsInstance(fig, go.Figure)

        fig = self.explainer.plot_shap_contributions(X_row=self.explainer.X.iloc[[0]])
        self.assertIsInstance(fig, go.Figure)

        fig = self.explainer.plot_shap_contributions(X_row=self.explainer.X_cats.iloc[[0]])
        self.assertIsInstance(fig, go.Figure)

    def test_plot_shap_interaction(self):
        fig = self.explainer.plot_shap_interaction("Age", "Sex_female")
        self.assertIsInstance(fig, go.Figure)

        fig = self.explainer.plot_shap_interaction("Sex_female", "Age")
        self.assertIsInstance(fig, go.Figure)

        fig = self.explainer.plot_shap_interaction("Gender", "Age")
        self.assertIsInstance(fig, go.Figure)

        fig = self.explainer.plot_shap_interaction("Age", "Sex_female", highlight_index=0)
        self.assertIsInstance(fig, go.Figure)

    def test_plot_pdp(self):
        fig = self.explainer.plot_pdp("Age")
        self.assertIsInstance(fig, go.Figure)

        fig = self.explainer.plot_pdp("Gender")
        self.assertIsInstance(fig, go.Figure)

        fig = self.explainer.plot_pdp("Gender", index=0)
        self.assertIsInstance(fig, go.Figure)

        fig = self.explainer.plot_pdp("Age", index=0)
        self.assertIsInstance(fig, go.Figure)

        fig = self.explainer.plot_pdp("Age", X_row=self.explainer.X.iloc[[0]])
        self.assertIsInstance(fig, go.Figure)

        fig = self.explainer.plot_pdp("Age", X_row=self.explainer.X_cats.iloc[[0]])
        self.assertIsInstance(fig, go.Figure)

    def test_yaml(self):
        yaml = self.explainer.to_yaml()
        self.assertIsInstance(yaml, str)








