import unittest

import pandas as pd
import numpy as np

from catboost import CatBoostClassifier, CatBoostRegressor

from explainerdashboard import RegressionExplainer, ClassifierExplainer
from explainerdashboard.datasets import *

class CatBoostRegressionTests(unittest.TestCase):
    def setUp(self):
        X_train, y_train, X_test, y_test = titanic_fare()
        self.test_len = len(X_test)

        train_names, test_names = titanic_names()
        _, self.names = titanic_names()

        model = CatBoostRegressor(iterations=5, verbose=0).fit(X_train, y_train)
        explainer = RegressionExplainer(model, X_test, y_test, 
                                        cats=[{'Gender': ['Sex_female', 'Sex_male', 'Sex_nan']}, 
                                                'Deck', 'Embarked'])
        X_cats, y_cats = explainer.X_cats, explainer.y
        model = CatBoostRegressor(iterations=5, verbose=0).fit(X_cats, y_cats, cat_features=[5, 6, 7])
        self.explainer = RegressionExplainer(model, X_cats, y_cats)


    def test_preds(self):
        self.assertIsInstance(self.explainer.preds, np.ndarray)

    def test_permutation_importances(self):
        self.assertIsInstance(self.explainer.permutation_importances, pd.DataFrame)
        self.assertIsInstance(self.explainer.permutation_importances_cats, pd.DataFrame)

    def test_shap_base_value(self):
        self.assertIsInstance(self.explainer.shap_base_value, (np.floating, float))

    def test_shap_values_shape(self):
        self.assertTrue(self.explainer.shap_values.shape == (len(self.explainer), len(self.explainer.columns)))

    def test_shap_values(self):
        self.assertIsInstance(self.explainer.shap_values, np.ndarray)
        self.assertIsInstance(self.explainer.shap_values_cats, np.ndarray)

    # @unittest.expectedFailure
    # def test_shap_interaction_values(self):
    #     self.assertIsInstance(self.explainer.shap_interaction_values, np.ndarray)
    #     self.assertIsInstance(self.explainer.shap_interaction_values_cats, np.ndarray)

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


class CatBoostClassifierTests(unittest.TestCase):
    def setUp(self):
        X_train, y_train, X_test, y_test = titanic_survive()
        train_names, test_names = titanic_names()

        model = CatBoostClassifier(iterations=100, verbose=0).fit(X_train, y_train)
        explainer = ClassifierExplainer(
                            model, X_test, y_test, 
                            cats=[{'Gender': ['Sex_female', 'Sex_male', 'Sex_nan']}, 
                                                'Deck', 'Embarked'],
                            labels=['Not survived', 'Survived'],
                            idxs=test_names)

        X_cats, y_cats = explainer.X_cats, explainer.y
        model = CatBoostClassifier(iterations=5, verbose=0).fit(X_cats, y_cats, cat_features=[5, 6, 7])
        self.explainer = ClassifierExplainer(model, X_cats, y_cats)

    def test_preds(self):
        self.assertIsInstance(self.explainer.preds, np.ndarray)

    def test_pred_probas(self):
        self.assertIsInstance(self.explainer.pred_probas, np.ndarray)

    def test_permutation_importances(self):
        self.assertIsInstance(self.explainer.permutation_importances, pd.DataFrame)
        self.assertIsInstance(self.explainer.permutation_importances_cats, pd.DataFrame)

    def test_shap_base_value(self):
        self.assertIsInstance(self.explainer.shap_base_value, (np.floating, float))

    def test_shap_values_shape(self):
        self.assertTrue(self.explainer.shap_values.shape == (len(self.explainer), len(self.explainer.columns)))

    def test_shap_values(self):
        self.assertIsInstance(self.explainer.shap_values, np.ndarray)
        self.assertIsInstance(self.explainer.shap_values_cats, np.ndarray)

    # @unittest.expectedFailure
    # def test_shap_interaction_values(self):
    #     self.assertIsInstance(self.explainer.shap_interaction_values, np.ndarray)
    #     self.assertIsInstance(self.explainer.shap_interaction_values_cats, np.ndarray)

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