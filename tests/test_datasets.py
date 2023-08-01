import unittest

from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from explainerdashboard import ClassifierExplainer, ExplainerDashboard
from explainerdashboard.custom import ShapDependenceComposite
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import os

class CategoricalModelWrapper:
    def __init__(self, model) -> None:
        self._model = model
        pass

    def _perform_one_hot_encoding(self, X, feature, values):
        one_hot_enc = OneHotEncoder(dtype='int64', sparse_output=False, handle_unknown="ignore").set_output(transform="pandas")
        one_hot_enc.fit(values)
        result = one_hot_enc.transform(X[[feature]])
        for col in result.columns:
            result = result.rename(columns={ col : col.replace("x0", feature)})
        return pd.concat([X, result], axis=1).drop(columns=[feature])
    
    def _perform_label_encoding(self, y):
        label_enc = LabelEncoder()
        label_enc.fit([["unacc"],["acc"],["good"],["vgood"]])
        return pd.Series(label_enc.transform(y.values), name=y.name, index=y.index)
        
    def _perform_label_decoding(self, y):
        label_enc = LabelEncoder()
        label_enc.fit([["unacc"],["acc"],["good"],["vgood"]])
        return pd.Series(label_enc.inverse_transform(y), name=y.name)

    def _preprocessor(self, X):
        #Emulate a manual pipeline, e.g. what AutoML solutions can produce
        #preprocess buying
        X = self._perform_one_hot_encoding(X, "buying", [["vhigh"],["high"],["med"],["low"]])
        X = self._perform_one_hot_encoding(X, "maint", [["vhigh"],["high"],["med"],["low"]])
        X = self._perform_one_hot_encoding(X, "doors", [["2"],["3"],["4"],["5more"]])
        X = self._perform_one_hot_encoding(X, "persons", [["2"],["4"],["more"]])
        X = self._perform_one_hot_encoding(X, "lug_boot", [["small"],["med"],["big"]])
        X = self._perform_one_hot_encoding(X, "safety", [["low"],["med"],["high"]])
        return X

    def _postprocessor(self, y):
        return self._perform_label_decoding(y)

    def predict(self, X):
        X = self._preprocessor(X)
        y = self._model.predict(X)
        return self._postprocessor(y)

    def predict_proba(self, X):
        X = self._preprocessor(X)
        probabilities_raw = self._model.predict_proba(X)
        return probabilities_raw

def generate_categorical_dataset_model_wrapper(categorical_label=False):
    df = pd.read_csv(os.path.join(os.getcwd(), "tests\\test_assets\\car.csv"))
    X_train, X_test, y_train, y_test = train_test_split(df.drop(["class"], axis=1), df["class"], test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=5, max_depth=2)
    wrapper = CategoricalModelWrapper(model)
    X_train = wrapper._preprocessor(X_train)
    y_train = wrapper._perform_label_encoding(y_train)
    if categorical_label == False:
        #We only test categorical features and numerical target
        y_test = wrapper._perform_label_encoding(y_test)
    model.fit(X_train, y_train)
    return CategoricalModelWrapper(model), X_test, y_test

def test_NaN_containing_categorical_dataset():
    _wrapper, _test_X, _test_y = generate_categorical_dataset_model_wrapper()
    explainer = ClassifierExplainer(
                    _wrapper, _test_X, _test_y)
    dashboard = ExplainerDashboard(explainer)
    assert "NaN" in explainer.categorical_dict["buying"]
    
def test_categorical_label():
    _wrapper, _test_X, _test_y = generate_categorical_dataset_model_wrapper(True)
    explainer = ClassifierExplainer(
                    _wrapper, _test_X, _test_y)
    dashboard = ExplainerDashboard(explainer)
    assert "unacc" in explainer.labels
