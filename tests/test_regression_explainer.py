import unittest

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

import plotly.graph_objects as go

from explainerdashboard.explainers import RegressionExplainer
from explainerdashboard.datasets import titanic_fare, titanic_names


class RegressionBunchTests(unittest.TestCase):
    def setUp(self):
        X_train, y_train, X_test, y_test = titanic_fare()
        self.test_len = len(X_test)

        train_names, test_names = titanic_names()
        _, self.names = titanic_names()

        model = RandomForestRegressor(n_estimators=5, max_depth=2)
        model.fit(X_train, y_train)

        self.explainer = RegressionExplainer(
                            model, X_test, y_test, r2_score, 
                            shap='tree',
                            cats=['Sex', 'Cabin', 'Embarked'],
                            idxs=test_names)

    def test_residuals(self):
        self.assertIsInstance(self.explainer.residuals, pd.Series)

    def test_prediction_result_markdown(self):
        result_index = self.explainer.prediction_result_markdown(0)
        self.assertIsInstance(result_index, str)
        result_name = self.explainer.prediction_result_markdown(self.names[0])
        self.assertIsInstance(result_name, str)

    def test_metrics(self):
        metrics_dict = self.explainer.metrics()
        self.assertIsInstance(metrics_dict, dict)
        self.assertTrue('rmse' in metrics_dict)
        self.assertTrue('mae' in metrics_dict)
        self.assertTrue('R2' in metrics_dict)

    def test_plot_predicted_vs_actual(self):
        fig = self.explainer.plot_predicted_vs_actual(logs=False)
        self.assertIsInstance(fig, go.Figure)

        fig = self.explainer.plot_predicted_vs_actual(logs=True)
        self.assertIsInstance(fig, go.Figure)

    def test_plot_residuals(self):
        fig = self.explainer.plot_residuals()
        self.assertIsInstance(fig, go.Figure)

        fig = self.explainer.plot_residuals(vs_actual=True)
        self.assertIsInstance(fig, go.Figure)

        fig = self.explainer.plot_residuals(ratio=True)
        self.assertIsInstance(fig, go.Figure)

        fig = self.explainer.plot_residuals(ratio=True, vs_actual=True)
        self.assertIsInstance(fig, go.Figure)

    def test_plot_residuals_vs_feature(self):
        fig = self.explainer.plot_residuals_vs_feature("Age")
        self.assertIsInstance(fig, go.Figure)

        fig = self.explainer.plot_residuals_vs_feature("Age", ratio=True)
        self.assertIsInstance(fig, go.Figure)

        fig = self.explainer.plot_residuals_vs_feature("Age", dropna=True)
        self.assertIsInstance(fig, go.Figure)












if __name__ == '__main__':
    unittest.main()

