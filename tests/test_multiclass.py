import unittest

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

import plotly.graph_objects as go

from explainerdashboard.explainers import ClassifierExplainer
from explainerdashboard.datasets import titanic_embarked, titanic_names


class MultiClassClassifierBunchTests(unittest.TestCase):
    def setUp(self):
        X_train, y_train, X_test, y_test = titanic_embarked()
        train_names, test_names = titanic_names()

        model = RandomForestClassifier(n_estimators=5, max_depth=2)
        model.fit(X_train, y_train)

        self.explainer = ClassifierExplainer(model, X_test, y_test,  
                            cats=[{'Gender': ['Sex_female', 'Sex_male', 'Sex_nan']}, 
                                                'Deck'],
                            idxs=test_names, 
                            labels=['Queenstown', 'Southampton', 'Cherbourg'])

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
        self.assertEqual(self.explainer.shap_interaction_values_by_col("Age").shape, 
                        self.explainer.shap_values.shape)
        self.assertEqual(self.explainer.shap_interaction_values_by_col("Age", cats=True).shape, 
                        self.explainer.shap_values_cats.shape)

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

        fig = self.explainer.plot_shap_interaction_summary("Gender")
        self.assertIsInstance(fig, go.Figure)

    def test_plot_shap_dependence(self):
        fig = self.explainer.plot_shap_dependence("Age")
        self.assertIsInstance(fig, go.Figure)

        fig = self.explainer.plot_shap_dependence("Sex_female")
        self.assertIsInstance(fig, go.Figure)

        fig = self.explainer.plot_shap_dependence("Age", "Gender")
        self.assertIsInstance(fig, go.Figure)

        fig = self.explainer.plot_shap_dependence("Sex_female", "Age")
        self.assertIsInstance(fig, go.Figure)

        fig = self.explainer.plot_shap_dependence("Age", highlight_index=0)
        self.assertIsInstance(fig, go.Figure)

        fig = self.explainer.plot_shap_dependence("Gender", highlight_index=0)
        self.assertIsInstance(fig, go.Figure)

    def test_plot_shap_interaction(self):
        fig = self.explainer.plot_shap_dependence("Age", "Sex_female")
        self.assertIsInstance(fig, go.Figure)

        fig = self.explainer.plot_shap_dependence("Sex_female", "Age")
        self.assertIsInstance(fig, go.Figure)

        fig = self.explainer.plot_shap_dependence("Gender", "Age")
        self.assertIsInstance(fig, go.Figure)

        fig = self.explainer.plot_shap_dependence("Age", "Gender")
        self.assertIsInstance(fig, go.Figure)

        fig = self.explainer.plot_shap_dependence("Age", "Sex_female", highlight_index=0)
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

    def test_pos_label(self):
        self.explainer.pos_label = 1
        self.explainer.pos_label = "Southampton"
        self.assertIsInstance(self.explainer.pos_label, int)
        self.assertIsInstance(self.explainer.pos_label_str, str)
        self.assertEqual(self.explainer.pos_label, 1)
        self.assertEqual(self.explainer.pos_label_str, "Southampton")

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

    def test_calculate_properties(self):
        self.explainer.calculate_properties()
        
    def test_plot_precision(self):
        fig = self.explainer.plot_precision()
        self.assertIsInstance(fig, go.Figure)

        fig = self.explainer.plot_precision(multiclass=True)
        self.assertIsInstance(fig, go.Figure)

        fig = self.explainer.plot_precision(quantiles=10, cutoff=0.5)
        self.assertIsInstance(fig, go.Figure)

    def test_plot_cumulative_precision(self):
        fig = self.explainer.plot_cumulative_precision()
        self.assertIsInstance(fig, go.Figure)

    def test_plot_confusion_matrix(self):
        fig = self.explainer.plot_confusion_matrix(normalized=False, binary=False)
        self.assertIsInstance(fig, go.Figure)

        fig = self.explainer.plot_confusion_matrix(normalized=False, binary=True)
        self.assertIsInstance(fig, go.Figure)

        fig = self.explainer.plot_confusion_matrix(normalized=True, binary=False)
        self.assertIsInstance(fig, go.Figure)

        fig = self.explainer.plot_confusion_matrix(normalized=True, binary=True)
        self.assertIsInstance(fig, go.Figure)

    def test_plot_lift_curve(self):
        fig = self.explainer.plot_lift_curve()
        self.assertIsInstance(fig, go.Figure)

        fig = self.explainer.plot_lift_curve(percentage=True)
        self.assertIsInstance(fig, go.Figure)

        fig = self.explainer.plot_lift_curve(cutoff=0.5)
        self.assertIsInstance(fig, go.Figure)

    def test_plot_lift_curve(self):
        fig = self.explainer.plot_lift_curve()
        self.assertIsInstance(fig, go.Figure)

        fig = self.explainer.plot_lift_curve(percentage=True)
        self.assertIsInstance(fig, go.Figure)

        fig = self.explainer.plot_lift_curve(cutoff=0.5)
        self.assertIsInstance(fig, go.Figure)

    def test_plot_classification(self):
        fig = self.explainer.plot_classification()
        self.assertIsInstance(fig, go.Figure)

        fig = self.explainer.plot_classification(percentage=True)
        self.assertIsInstance(fig, go.Figure)

        fig = self.explainer.plot_classification(cutoff=0)
        self.assertIsInstance(fig, go.Figure)

        fig = self.explainer.plot_classification(cutoff=1)
        self.assertIsInstance(fig, go.Figure)

    def test_plot_roc_auc(self):
        fig = self.explainer.plot_roc_auc(0.5)
        self.assertIsInstance(fig, go.Figure)

        fig = self.explainer.plot_roc_auc(0.0)
        self.assertIsInstance(fig, go.Figure)

        fig = self.explainer.plot_roc_auc(1.0)
        self.assertIsInstance(fig, go.Figure)

    def test_plot_pr_auc(self):
        fig = self.explainer.plot_pr_auc(0.5)
        self.assertIsInstance(fig, go.Figure)

        fig = self.explainer.plot_pr_auc(0.0)
        self.assertIsInstance(fig, go.Figure)

        fig = self.explainer.plot_pr_auc(1.0)
        self.assertIsInstance(fig, go.Figure)

if __name__ == '__main__':
    unittest.main()

