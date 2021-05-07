import unittest

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from torch import nn

from skorch import NeuralNetClassifier, NeuralNetRegressor

from explainerdashboard.explainers import RegressionExplainer, ClassifierExplainer

def get_skorch_regressor():
    X, y = make_regression(100, 5, n_informative=3, random_state=0)
    X = X.astype(np.float32)
    y = y / np.std(y)
    y = y.reshape(-1, 1).astype(np.float32)

    X_df = pd.DataFrame(X, columns=['col'+str(i) for i in range(X.shape[1])])

    class MyModule(nn.Module):
        def __init__(self, input_units=5, num_units=5, nonlin=nn.ReLU()):
            super(MyModule, self).__init__()

            self.dense0 = nn.Linear(input_units, num_units)
            self.nonlin = nonlin
            self.dense1 = nn.Linear(num_units, num_units)
            self.output = nn.Linear(num_units, 1)

        def forward(self, X, **kwargs):
            X = self.nonlin(self.dense0(X))
            X = self.nonlin(self.dense1(X))
            X = self.output(X)
            return X

    model = NeuralNetRegressor(
        MyModule,
        max_epochs=20,
        lr=0.2,
        iterator_train__shuffle=True,
    )

    model.fit(X_df.values, y)
    return model, X_df, y


def get_skorch_classifier():
    X, y = make_classification(200, 5, n_informative=3, random_state=0)
    X = X.astype(np.float32)
    y = y.astype(np.int64)

    X_df = pd.DataFrame(X, columns=['col'+str(i) for i in range(X.shape[1])])


    class MyModule(nn.Module):
        def __init__(self, input_units=5, num_units=5, nonlin=nn.ReLU()):
            super(MyModule, self).__init__()

            self.dense0 = nn.Linear(input_units, num_units)
            self.nonlin = nonlin
            self.dense1 = nn.Linear(num_units, num_units)
            self.output = nn.Linear(num_units, 2)
            self.softmax = nn.Softmax(dim=-1)

        def forward(self, X, **kwargs):
            X = self.nonlin(self.dense0(X))
            X = self.nonlin(self.dense1(X))
            X = self.softmax(self.output(X))
            return X


    model = NeuralNetClassifier(
        MyModule,
        max_epochs=20,
        lr=0.1,
    )

    model.fit(X_df.values, y)
    return model, X_df, y


class SkorchRegressorTests(unittest.TestCase):
    def setUp(self):
        model, X, y = get_skorch_regressor()
        self.explainer = RegressionExplainer(model, X, y)

    def test_preds(self):
        self.assertIsInstance(self.explainer.preds, np.ndarray)

    def test_permutation_importances(self):
        self.assertIsInstance(self.explainer.get_permutation_importances_df(), pd.DataFrame)

    def test_shap_base_value(self):
        self.assertIsInstance(self.explainer.shap_base_value(), (np.floating, float))

    def test_shap_values_shape(self):
        self.assertTrue(self.explainer.get_shap_values_df().shape == (len(self.explainer), len(self.explainer.merged_cols)))

    def test_shap_values(self):
        self.assertIsInstance(self.explainer.get_shap_values_df(), pd.DataFrame)

    def test_mean_abs_shap(self):
        self.assertIsInstance(self.explainer.get_mean_abs_shap_df(), pd.DataFrame)

    def test_calculate_properties(self):
        self.explainer.calculate_properties(include_interactions=False)

    def test_pdp_df(self):
        self.assertIsInstance(self.explainer.pdp_df("col1"), pd.DataFrame)


class SkorchClassifierTests(unittest.TestCase):
    def setUp(self):
        model, X, y = get_skorch_classifier()
        self.explainer = ClassifierExplainer(model, X, y)

    def test_preds(self):
        self.assertIsInstance(self.explainer.preds, np.ndarray)

    def test_pred_probas(self):
        self.assertIsInstance(self.explainer.pred_probas(), np.ndarray)

    def test_permutation_importances(self):
        self.assertIsInstance(self.explainer.get_permutation_importances_df(), pd.DataFrame)

    def test_shap_base_value(self):
        self.assertIsInstance(self.explainer.shap_base_value(), (np.floating, float))

    def test_shap_values_shape(self):
        self.assertTrue(self.explainer.get_shap_values_df().shape == (len(self.explainer), len(self.explainer.merged_cols)))

    def test_shap_values(self):
        self.assertIsInstance(self.explainer.get_shap_values_df(), pd.DataFrame)

    def test_mean_abs_shap(self):
        self.assertIsInstance(self.explainer.get_mean_abs_shap_df(), pd.DataFrame)

    def test_calculate_properties(self):
        self.explainer.calculate_properties(include_interactions=False)

    def test_pdp_df(self):
        self.assertIsInstance(self.explainer.pdp_df("col1"), pd.DataFrame)

    def test_metrics(self):
        self.assertIsInstance(self.explainer.metrics(), dict)

    def test_precision_df(self):
        self.assertIsInstance(self.explainer.get_precision_df(), pd.DataFrame)
        
    def test_lift_curve_df(self):
        self.assertIsInstance(self.explainer.get_liftcurve_df(), pd.DataFrame)

    def test_prediction_result_df(self):
        self.assertIsInstance(self.explainer.prediction_result_df(0), pd.DataFrame)
