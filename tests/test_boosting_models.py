import unittest

import pandas as pd
import numpy as np

from sklearn.metrics import r2_score, roc_auc_score

import pdpbox

from xgboost import XGBClassifier, XGBRegressor
from lightgbm.sklearn import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

from explainerdashboard.explainers import RegressionExplainer, ClassifierExplainer
from explainerdashboard.datasets import titanic_fare, titanic_survive, titanic_names


class XGBRegressionTests(unittest.TestCase):
    def setUp(self):
        X_train, y_train, X_test, y_test = titanic_fare()
        self.test_len = len(X_test)

        train_names, test_names = titanic_names()
        _, self.names = titanic_names()

        model = XGBRegressor()
        model.fit(X_train, y_train)
        self.explainer = RegressionExplainer(model, X_test, y_test, 
                                        cats=[{'Gender': ['Sex_female', 'Sex_male', 'Sex_nan']}, 
                                                'Deck', 'Embarked'],
                                        units="$")

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

    def test_pdp_result(self):
        self.assertIsInstance(self.explainer.get_pdp_result("Age"), pdpbox.pdp.PDPIsolate)
        self.assertIsInstance(self.explainer.get_pdp_result("Gender"), pdpbox.pdp.PDPIsolate)
        self.assertIsInstance(self.explainer.get_pdp_result("Deck"), pdpbox.pdp.PDPIsolate)
        self.assertIsInstance(self.explainer.get_pdp_result("Age", index=0), pdpbox.pdp.PDPIsolate)
        self.assertIsInstance(self.explainer.get_pdp_result("Gender", index=0), pdpbox.pdp.PDPIsolate)

class LGBMRegressionTests(unittest.TestCase):
    def setUp(self):
        X_train, y_train, X_test, y_test = titanic_fare()
        self.test_len = len(X_test)

        train_names, test_names = titanic_names()
        _, self.names = titanic_names()

        model = LGBMRegressor()
        model.fit(X_train, y_train)
        self.explainer = RegressionExplainer(model, X_test, y_test, r2_score, 
                                        shap='tree', 
                                        cats=[{'Gender': ['Sex_female', 'Sex_male', 'Sex_nan']}, 
                                                'Deck', 'Embarked'],
                                        idxs=test_names, units="$")

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

    # def test_shap_interaction_values(self):
    #     self.assertIsInstance(self.explainer.shap_interaction_values, np.ndarray)
    #     self.assertIsInstance(self.explainer.shap_interaction_values_cats, np.ndarray)

    def test_mean_abs_shap(self):
        self.assertIsInstance(self.explainer.mean_abs_shap, pd.DataFrame)
        self.assertIsInstance(self.explainer.mean_abs_shap_cats, pd.DataFrame)

    def test_calculate_properties(self):
        self.explainer.calculate_properties(include_interactions=True)

    def test_pdp_result(self):
        self.assertIsInstance(self.explainer.get_pdp_result("Age"), pdpbox.pdp.PDPIsolate)
        self.assertIsInstance(self.explainer.get_pdp_result("Gender"), pdpbox.pdp.PDPIsolate)
        self.assertIsInstance(self.explainer.get_pdp_result("Age", index=0), pdpbox.pdp.PDPIsolate)
        self.assertIsInstance(self.explainer.get_pdp_result("Gender", index=0), pdpbox.pdp.PDPIsolate)

class CatBoostRegressionTests(unittest.TestCase):
    def setUp(self):
        X_train, y_train, X_test, y_test = titanic_fare()
        self.test_len = len(X_test)

        train_names, test_names = titanic_names()
        _, self.names = titanic_names()

        model = CatBoostRegressor(iterations=100, learning_rate=0.1, verbose=0)
        model.fit(X_train, y_train)

        self.explainer = RegressionExplainer(model, X_test, y_test, r2_score, 
                                        cats=[{'Gender': ['Sex_female', 'Sex_male', 'Sex_nan']}, 
                                                'Deck', 'Embarked'],
                                        idxs=test_names, units="$")

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

    def test_pdp_result(self):
        self.assertIsInstance(self.explainer.get_pdp_result("Age"), pdpbox.pdp.PDPIsolate)
        self.assertIsInstance(self.explainer.get_pdp_result("Gender"), pdpbox.pdp.PDPIsolate)
        self.assertIsInstance(self.explainer.get_pdp_result("Deck"), pdpbox.pdp.PDPIsolate)
        self.assertIsInstance(self.explainer.get_pdp_result("Age", index=0), pdpbox.pdp.PDPIsolate)
        self.assertIsInstance(self.explainer.get_pdp_result("Gender", index=0), pdpbox.pdp.PDPIsolate)


class XGBCLassifierTests(unittest.TestCase):
    def setUp(self):
        X_train, y_train, X_test, y_test = titanic_survive()
        train_names, test_names = titanic_names()

        model = XGBClassifier()
        model.fit(X_train, y_train)

        self.explainer = ClassifierExplainer(
                            model, X_test, y_test, 
                            cats=[{'Gender': ['Sex_female', 'Sex_male', 'Sex_nan']}, 
                                                'Deck', 'Embarked'],
                            labels=['Not survived', 'Survived'])

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

    def test_shap_values_all_probabilities(self):
        self.assertTrue(self.explainer.shap_base_value >= 0)
        self.assertTrue(self.explainer.shap_base_value <= 1)
        self.assertTrue(np.all(self.explainer.shap_values.sum(axis=1) + self.explainer.shap_base_value >= 0))
        self.assertTrue(np.all(self.explainer.shap_values.sum(axis=1) + self.explainer.shap_base_value <= 1))

    # def test_shap_interaction_values(self):
    #     self.assertIsInstance(self.explainer.shap_interaction_values, np.ndarray)
    #     self.assertIsInstance(self.explainer.shap_interaction_values_cats, np.ndarray)

    def test_mean_abs_shap(self):
        self.assertIsInstance(self.explainer.mean_abs_shap, pd.DataFrame)
        self.assertIsInstance(self.explainer.mean_abs_shap_cats, pd.DataFrame)

    def test_calculate_properties(self):
        self.explainer.calculate_properties(include_interactions=False)

    def test_pdp_result(self):
        self.assertIsInstance(self.explainer.get_pdp_result("Age"), pdpbox.pdp.PDPIsolate)
        self.assertIsInstance(self.explainer.get_pdp_result("Gender"), pdpbox.pdp.PDPIsolate)
        self.assertIsInstance(self.explainer.get_pdp_result("Age", index=0), pdpbox.pdp.PDPIsolate)
        self.assertIsInstance(self.explainer.get_pdp_result("Gender", index=0), pdpbox.pdp.PDPIsolate)

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


class LGBMClassifierTests(unittest.TestCase):
    def setUp(self):
        X_train, y_train, X_test, y_test = titanic_survive()
        train_names, test_names = titanic_names()

        model = LGBMClassifier()
        model.fit(X_train, y_train)

        self.explainer = ClassifierExplainer(
                            model, X_test, y_test, roc_auc_score, 
                            cats=[{'Gender': ['Sex_female', 'Sex_male', 'Sex_nan']}, 
                                                'Deck', 'Embarked'],
                            labels=['Not survived', 'Survived'],
                            idxs=test_names)

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

    def test_shap_values_all_probabilities(self):
        self.assertTrue(self.explainer.shap_base_value >= 0)
        self.assertTrue(self.explainer.shap_base_value <= 1)
        self.assertTrue(np.all(self.explainer.shap_values.sum(axis=1) + self.explainer.shap_base_value >= 0))
        self.assertTrue(np.all(self.explainer.shap_values.sum(axis=1) + self.explainer.shap_base_value <= 1))

    def test_mean_abs_shap(self):
        self.assertIsInstance(self.explainer.mean_abs_shap, pd.DataFrame)
        self.assertIsInstance(self.explainer.mean_abs_shap_cats, pd.DataFrame)

    def test_calculate_properties(self):
        self.explainer.calculate_properties(include_interactions=False)

    def test_pdp_result(self):
        self.assertIsInstance(self.explainer.get_pdp_result("Age"), pdpbox.pdp.PDPIsolate)
        self.assertIsInstance(self.explainer.get_pdp_result("Gender"), pdpbox.pdp.PDPIsolate)
        self.assertIsInstance(self.explainer.get_pdp_result("Age", index=0), pdpbox.pdp.PDPIsolate)
        self.assertIsInstance(self.explainer.get_pdp_result("Gender", index=0), pdpbox.pdp.PDPIsolate)

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


class CatBoostClassifierTests(unittest.TestCase):
    def setUp(self):
        X_train, y_train, X_test, y_test = titanic_survive()
        train_names, test_names = titanic_names()

        model = CatBoostClassifier(iterations=100, learning_rate=0.1, verbose=0)
        model.fit(X_train, y_train)

        self.explainer = ClassifierExplainer(
                            model, X_test, y_test, roc_auc_score, 
                            cats=[{'Gender': ['Sex_female', 'Sex_male', 'Sex_nan']}, 
                                                'Deck', 'Embarked'],
                            labels=['Not survived', 'Survived'],
                            idxs=test_names)

    def test_preds(self):
        self.assertIsInstance(self.explainer.preds, np.ndarray)

    def test_pred_probas(self):
        self.assertIsInstance(self.explainer.pred_probas, np.ndarray)

    def test_permutation_importances(self):
        self.assertIsInstance(self.explainer.permutation_importances, pd.DataFrame)
        self.assertIsInstance(self.explainer.permutation_importances_cats, pd.DataFrame)

    def test_shap_base_value(self):
        self.assertIsInstance(self.explainer.shap_base_value, (np.floating, float))

    def test_shap_values_all_probabilities(self):
        self.assertTrue(self.explainer.shap_base_value >= 0)
        self.assertTrue(self.explainer.shap_base_value <= 1)
        self.assertTrue(np.all(self.explainer.shap_values.sum(axis=1) + self.explainer.shap_base_value >= 0))
        self.assertTrue(np.all(self.explainer.shap_values.sum(axis=1) + self.explainer.shap_base_value <= 1))

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

    def test_pdp_result(self):
        self.assertIsInstance(self.explainer.get_pdp_result("Age"), pdpbox.pdp.PDPIsolate)
        self.assertIsInstance(self.explainer.get_pdp_result("Gender"), pdpbox.pdp.PDPIsolate)
        self.assertIsInstance(self.explainer.get_pdp_result("Age", index=0), pdpbox.pdp.PDPIsolate)
        self.assertIsInstance(self.explainer.get_pdp_result("Gender", index=0), pdpbox.pdp.PDPIsolate)

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


        