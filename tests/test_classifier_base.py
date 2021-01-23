import unittest

import pandas as pd
import numpy as np
from pandas.api.types import is_categorical_dtype, is_numeric_dtype

from sklearn.ensemble import RandomForestClassifier

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
                            model, X_test, y_test, 
                            cats=[{'Gender': ['Sex_female', 'Sex_male', 'Sex_nan']}, 
                                                'Deck', 'Embarked'],
                            target='Survival',
                            labels=['Not survived', 'Survived'],
                            idxs=test_names)

    def test_explainer_len(self):
        self.assertEqual(len(self.explainer), len(titanic_survive()[2]))

    def test_int_idx(self):
        self.assertEqual(self.explainer.get_idx(titanic_names()[1][0]), 0)

    def test_random_index(self):
        self.assertIsInstance(self.explainer.random_index(), int)
        self.assertIsInstance(self.explainer.random_index(return_str=True), str)

    def test_preds(self):
        self.assertIsInstance(self.explainer.preds, np.ndarray)

    def test_row_from_input(self):
        input_row = self.explainer.get_row_from_input(
            self.explainer.X.iloc[[0]].values.tolist())
        self.assertIsInstance(input_row, pd.DataFrame)

        input_row = self.explainer.get_row_from_input(
            self.explainer.X_merged.iloc[[0]].values.tolist())
        self.assertIsInstance(input_row, pd.DataFrame)

        input_row = self.explainer.get_row_from_input(
            self.explainer.X_merged
            [self.explainer.columns_ranked_by_shap()]
            .iloc[[0]].values.tolist(), ranked_by_shap=True)
        self.assertIsInstance(input_row, pd.DataFrame)
        
    def test_pred_percentiles(self):
        self.assertIsInstance(self.explainer.pred_percentiles(), np.ndarray)

    def test_columns_ranked_by_shap(self):
        self.assertIsInstance(self.explainer.columns_ranked_by_shap(), list)

    def test_get_col(self):
        self.assertIsInstance(self.explainer.get_col("Gender"), pd.Series)
        self.assertTrue(is_categorical_dtype(self.explainer.get_col("Gender")))

        self.assertIsInstance(self.explainer.get_col("Deck"), pd.Series)
        self.assertTrue(is_categorical_dtype(self.explainer.get_col("Deck")))

        self.assertIsInstance(self.explainer.get_col("Age"), pd.Series)
        self.assertTrue(is_numeric_dtype(self.explainer.get_col("Age")))

    def test_permutation_importances(self):
        self.assertIsInstance(self.explainer.permutation_importances(), pd.DataFrame)
        
    def test_X_cats(self):
        self.assertIsInstance(self.explainer.X_cats, pd.DataFrame)

    def test_metrics(self):
        self.assertIsInstance(self.explainer.metrics(), dict)

    def test_mean_abs_shap_df(self):
        self.assertIsInstance(self.explainer.mean_abs_shap_df(), pd.DataFrame)

    def test_top_interactions(self):
        self.assertIsInstance(self.explainer.top_shap_interactions("Age"), list)
        self.assertIsInstance(self.explainer.top_shap_interactions("Age", topx=4), list)


    def test_permutation_importances_df(self):
        self.assertIsInstance(self.explainer.get_permutation_importances_df(), pd.DataFrame)
        self.assertIsInstance(self.explainer.get_permutation_importances_df(topx=3), pd.DataFrame)
        self.assertIsInstance(self.explainer.get_permutation_importances_df(cutoff=0.01), pd.DataFrame)

    def test_contrib_df(self):
        self.assertIsInstance(self.explainer.get_contrib_df(0), pd.DataFrame)
        self.assertIsInstance(self.explainer.get_contrib_df(0, topx=3), pd.DataFrame)
        self.assertIsInstance(self.explainer.get_contrib_df(0, sort='high-to-low'), pd.DataFrame)
        self.assertIsInstance(self.explainer.get_contrib_df(0, sort='low-to-high'), pd.DataFrame)
        self.assertIsInstance(self.explainer.get_contrib_df(0, sort='importance'), pd.DataFrame)
        self.assertIsInstance(self.explainer.get_contrib_df(X_row=self.explainer.X.iloc[[0]]), pd.DataFrame)

    def test_contrib_summary_df(self):
        self.assertIsInstance(self.explainer.get_contrib_summary_df(0), pd.DataFrame)
        self.assertIsInstance(self.explainer.get_contrib_summary_df(0, topx=3), pd.DataFrame)
        self.assertIsInstance(self.explainer.get_contrib_summary_df(0, round=3), pd.DataFrame)
        self.assertIsInstance(self.explainer.get_contrib_summary_df(0, sort='low-to-high'), pd.DataFrame)
        self.assertIsInstance(self.explainer.get_contrib_summary_df(0, sort='high-to-low'), pd.DataFrame)
        self.assertIsInstance(self.explainer.get_contrib_summary_df(0, sort='importance'), pd.DataFrame)
        self.assertIsInstance(self.explainer.get_contrib_summary_df(X_row=self.explainer.X.iloc[[0]]), pd.DataFrame)

    def test_shap_base_value(self):
        self.assertIsInstance(self.explainer.shap_base_value(), (np.floating, float))

    def test_shap_values_shape(self):
        self.assertTrue(self.explainer.get_shap_values_df().shape == (len(self.explainer), len(self.explainer.merged_cols)))

    def test_shap_values(self):
        self.assertIsInstance(self.explainer.get_shap_values_df(), pd.DataFrame)

    def test_shap_interaction_values(self):
        self.assertIsInstance(self.explainer.shap_interaction_values(), np.ndarray)

    def test_mean_abs_shap_df(self):
        self.assertIsInstance(self.explainer.mean_abs_shap_df(), pd.DataFrame)

    def test_calculate_properties(self):
        self.explainer.calculate_properties()

    def test_shap_interaction_values_by_col(self):
        self.assertIsInstance(self.explainer.shap_interaction_values_for_col("Age"), np.ndarray)
        self.assertEqual(self.explainer.shap_interaction_values_for_col("Age").shape, 
                        self.explainer.get_shap_values_df().shape)

    def test_prediction_result_df(self):
        df = self.explainer.prediction_result_df(0)
        self.assertIsInstance(df, pd.DataFrame)

    def test_pdp_df(self):
        self.assertIsInstance(self.explainer.pdp_df("Age"), pd.DataFrame)
        self.assertIsInstance(self.explainer.pdp_df("Gender"), pd.DataFrame)
        self.assertIsInstance(self.explainer.pdp_df("Deck"), pd.DataFrame)
        self.assertIsInstance(self.explainer.pdp_df("Age", index=0), pd.DataFrame)
        self.assertIsInstance(self.explainer.pdp_df("Gender", index=0), pd.DataFrame)
        self.assertIsInstance(self.explainer.pdp_df("Age", X_row=self.explainer.X.iloc[[0]]), pd.DataFrame)

    def test_plot_importances(self):
        fig = self.explainer.plot_importances()
        self.assertIsInstance(fig, go.Figure)

        fig = self.explainer.plot_importances(kind='permutation')
        self.assertIsInstance(fig, go.Figure)

        fig = self.explainer.plot_importances(topx=3)
        self.assertIsInstance(fig, go.Figure)

    def test_plot_interactions(self):
        fig = self.explainer.plot_interactions_importance("Age")
        self.assertIsInstance(fig, go.Figure)

        fig = self.explainer.plot_interactions_importance("Gender")
        self.assertIsInstance(fig, go.Figure)

    def test_plot_contributions(self):
        fig = self.explainer.plot_contributions(0)
        self.assertIsInstance(fig, go.Figure)

        fig = self.explainer.plot_contributions(0, topx=3)
        self.assertIsInstance(fig, go.Figure)

        fig = self.explainer.plot_contributions(0, cutoff=0.05)
        self.assertIsInstance(fig, go.Figure)

        fig = self.explainer.plot_contributions(0, sort='high-to-low')
        self.assertIsInstance(fig, go.Figure)

        fig = self.explainer.plot_contributions(0, sort='low-to-high')
        self.assertIsInstance(fig, go.Figure)

        fig = self.explainer.plot_contributions(0, sort='importance')
        self.assertIsInstance(fig, go.Figure)

        fig = self.explainer.plot_contributions(X_row=self.explainer.X.iloc[[0]], sort='importance')
        self.assertIsInstance(fig, go.Figure)

    def test_plot_shap_detailed(self):
        fig = self.explainer.plot_importances_detailed()
        self.assertIsInstance(fig, go.Figure)

        fig = self.explainer.plot_importances_detailed(topx=3)
        self.assertIsInstance(fig, go.Figure)

    def test_plot_interactions_detailed(self):
        fig = self.explainer.plot_interactions_detailed("Age")
        self.assertIsInstance(fig, go.Figure)

        fig = self.explainer.plot_interactions_detailed("Age", topx=3)
        self.assertIsInstance(fig, go.Figure)

        fig = self.explainer.plot_interactions_detailed("Age")
        self.assertIsInstance(fig, go.Figure)

        fig = self.explainer.plot_interactions_detailed("Gender")
        self.assertIsInstance(fig, go.Figure)

    def test_plot_dependence(self):
        fig = self.explainer.plot_dependence("Age")
        self.assertIsInstance(fig, go.Figure)

        fig = self.explainer.plot_dependence("Age", "Gender")
        self.assertIsInstance(fig, go.Figure)

        fig = self.explainer.plot_dependence("Age", highlight_index=0)
        self.assertIsInstance(fig, go.Figure)

        fig = self.explainer.plot_dependence("Gender", highlight_index=0)
        self.assertIsInstance(fig, go.Figure)

    def test_plot_interaction(self):

        fig = self.explainer.plot_interaction("Gender", "Age")
        self.assertIsInstance(fig, go.Figure)

        fig = self.explainer.plot_interaction("Age", "Gender", highlight_index=0)
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

    def test_yaml(self):
        yaml = self.explainer.to_yaml()
        self.assertIsInstance(yaml, str)




