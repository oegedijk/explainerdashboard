import unittest

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from explainerdashboard.explainers import RandomForestClassifierBunch
from explainerdashboard.datasets import titanic_survive, titanic_names


class RandomForestClassifierBunchTests(unittest.TestCase):
    def setUp(self):
        X_train, y_train, X_test, y_test = titanic_survive()
        train_names, test_names = titanic_names()

        model = RandomForestClassifier(n_estimators=50, max_depth=5)
        model.fit(X_train, y_train)

        self.explainer = RandomForestClassifierBunch(
                            model, X_test, y_test, roc_auc_score, 
                            cats=['Sex', 'Cabin', 'Embarked'],
                            idxs=test_names, 
                            labels=['Not survived', 'Survived'])

    def test_shap_base_value(self):
        self.assertIsInstance(self.explainer.shap_base_value, (np.floating, float))

    def test_shap_values_shape(self):
        self.assertTrue(self.explainer.shap_values.shape == (len(self.explainer), len(self.explainer.columns)))


if __name__ == '__main__':
    unittest.main()

