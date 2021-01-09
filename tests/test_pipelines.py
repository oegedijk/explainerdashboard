import unittest
from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

import plotly.graph_objects as go

from explainerdashboard.explainers import ClassifierExplainer
from explainerdashboard.datasets import titanic_survive, titanic_names


class ClassifierBaseExplainerTestsPipeline(unittest.TestCase):
    def setUp(self):
        #X, y = fetch_openml("titanic", version=1, as_frame=True, return_X_y=True)
        df = pd.read_csv(Path.cwd() / "tests" / "test_assets" / "pipeline_data.csv")
        X = df[['age', 'fare', 'embarked', 'sex', 'pclass']]
        y = df['survived'].astype(int)

        numeric_features = ['age', 'fare']
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())])

        categorical_features = ['embarked', 'sex', 'pclass']
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('ordinal', OrdinalEncoder())])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)])

        # Append classifier to preprocessing pipeline.
        # Now we have a full prediction pipeline.
        clf = Pipeline(steps=[('preprocessor', preprocessor),
                            ('classifier', RandomForestClassifier())])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        clf.fit(X_train, y_train)

        self.explainer = ClassifierExplainer(clf, X_test, y_test)

    def test_columns_ranked_by_shap(self):
        self.assertIsInstance(self.explainer.columns_ranked_by_shap(), list)

    def test_permutation_importances(self):
        self.assertIsInstance(self.explainer.permutation_importances, pd.DataFrame)

    def test_metrics(self):
        self.assertIsInstance(self.explainer.metrics(), dict)
        self.assertIsInstance(self.explainer.metrics_descriptions(), dict)

    def test_mean_abs_shap_df(self):
        self.assertIsInstance(self.explainer.mean_abs_shap_df(), pd.DataFrame)

    def test_contrib_df(self):
        self.assertIsInstance(self.explainer.contrib_df(0), pd.DataFrame)
        self.assertIsInstance(self.explainer.contrib_df(0, cats=False), pd.DataFrame)
        self.assertIsInstance(self.explainer.contrib_df(0, topx=3), pd.DataFrame)
        self.assertIsInstance(self.explainer.contrib_df(0, sort='high-to-low'), pd.DataFrame)
        self.assertIsInstance(self.explainer.contrib_df(0, sort='low-to-high'), pd.DataFrame)
        self.assertIsInstance(self.explainer.contrib_df(0, sort='importance'), pd.DataFrame)
        self.assertIsInstance(self.explainer.contrib_df(X_row=self.explainer.X.iloc[[0]]), pd.DataFrame)
        self.assertIsInstance(self.explainer.contrib_df(X_row=self.explainer.X_cats.iloc[[0]]), pd.DataFrame)

    def test_shap_base_value(self):
        self.assertIsInstance(self.explainer.shap_base_value, (np.floating, float))

    def test_shap_values_shape(self):
        self.assertTrue(self.explainer.shap_values.shape == (len(self.explainer), len(self.explainer.columns)))

    def test_shap_values(self):
        self.assertIsInstance(self.explainer.shap_values, np.ndarray)
        self.assertIsInstance(self.explainer.shap_values_cats, np.ndarray)

    def test_pdp_df(self):
        self.assertIsInstance(self.explainer.pdp_df("age"), pd.DataFrame)
        self.assertIsInstance(self.explainer.pdp_df("sex"), pd.DataFrame)
        self.assertIsInstance(self.explainer.pdp_df("age", index=0), pd.DataFrame)
        self.assertIsInstance(self.explainer.pdp_df("sex", index=0), pd.DataFrame)




