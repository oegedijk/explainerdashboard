import unittest

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

import plotly.graph_objects as go

from explainerdashboard.explainers import RandomForestClassifierBunch
from explainerdashboard.datasets import titanic_survive, titanic_names


class ClassifierBunchTests(unittest.TestCase):
    def setUp(self):
        X_train, y_train, X_test, y_test = titanic_survive()
        train_names, test_names = titanic_names()

        model = RandomForestClassifier(n_estimators=5, max_depth=2)
        model.fit(X_train, y_train)

        self.explainer = ClassifierBunch(
                            model, X_test, y_test, roc_auc_score, 
                            shap='tree',
                            cats=['Sex', 'Cabin', 'Embarked'],
                            idxs=test_names, 
                            labels=['Not survived', 'Survived'])

    def test_shap_base_value(self):
        self.assertIsInstance(self.explainer.shap_base_value, (np.floating, float))

    def test_shap_values_shape(self):
        self.assertTrue(self.explainer.shap_values.shape == (len(self.explainer), len(self.explainer.columns)))

    def test_pred_percentiles(self):
        self.assertIsInstance(self.explainer.pred_percentiles, np.ndarray)

    def test_columns_ranked_by_shap(self):
        self.assertIsInstance(self.explainer.columns_ranked_by_shap(), list)

    def test_equivalent_col(self):
        self.assertEqual(self.explainer.equivalent_col("Sex_female"), "Sex")
        self.assertEqual(self.explainer.equivalent_col("Sex"), "Sex_female")
        self.assertIsNone(self.explainer.equivalent_col("random"))

    def test_get_col(self):
        self.assertIsInstance(self.explainer.get_col("Sex"), pd.Series)
        self.assertEqual(self.explainer.get_col("Sex").dtype, "object")

        self.assertIsInstance(self.explainer.get_col("Fare"), pd.Series)
        self.assertEqual(self.explainer.get_col("Fare").dtype, np.float)

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
        self.assertIsInstance(self.explainer.shap_top_interactions("Fare"), list)
        self.assertIsInstance(self.explainer.shap_top_interactions("Fare", topx=4), list)
        self.assertIsInstance(self.explainer.shap_top_interactions("Fare", cats=True), list)
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

    def test_plot_confusion_matrix(self):
        fig = self.explainer.plot_confusion_matrix(normalized=False, binary=False)
        self.assertIsInstance(fig, go.Figure)

        fig = self.explainer.plot_confusion_matrix(normalized=False, binary=True)
        self.assertIsInstance(fig, go.Figure)

        fig = self.explainer.plot_confusion_matrix(normalized=True, binary=False)
        self.assertIsInstance(fig, go.Figure)

        fig = self.explainer.plot_confusion_matrix(normalized=True, binary=True)
        self.assertIsInstance(fig, go.Figure)

    def test_plot_precision(self):
        fig = self.explainer.plot_precision()
        self.assertIsInstance(fig, go.Figure)

        fig = self.explainer.plot_precision(multiclass=True)
        self.assertIsInstance(fig, go.Figure)

    def test_plot_cumulutive_precision(self):
        fig = self.explainer.plot_cumulative_precision()
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

