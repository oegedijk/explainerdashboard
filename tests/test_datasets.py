from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from explainerdashboard import ClassifierExplainer, ExplainerDashboard
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import os
import numpy as np

class CategoricalModelWrapper:
    def __init__(self, model, categorical_label_test) -> None:
        self._model = model
        self._categorical_label_test = categorical_label_test
        pass
    
    def _perform_label_encoding(self, y):
        label_enc = LabelEncoder()
        label_enc.fit([["Survived"],["Not Survived"]])
        return pd.Series(label_enc.transform(y.values), name=y.name, index=y.index)
        
    def _perform_label_decoding(self, y):
        label_enc = LabelEncoder()
        label_enc.fit([["Survived"],["Not Survived"]])
        return pd.Series(label_enc.inverse_transform(y), name=y.name)

    def _preprocessor(self, X):
        return X.drop(["Name"], axis=1)

    def _postprocessor(self, y):
        if self._categorical_label_test == True:
            y = self._perform_label_decoding(y)
        return y

    def predict(self, X):
        X = self._preprocessor(X)
        y = self._model.predict(X)
        return self._postprocessor(y)

    def predict_proba(self, X):
        X = self._preprocessor(X)
        probabilities_raw = self._model.predict_proba(X)
        return probabilities_raw

def generate_categorical_dataset_model_wrapper(categorical_label_test=False):
    model = RandomForestClassifier(n_estimators=5, max_depth=2)
    wrapper = CategoricalModelWrapper(model, categorical_label_test)
    df = pd.read_csv(os.path.join(os.getcwd(), "tests\\test_assets\\data.csv"))
    if categorical_label_test == True:
        #Test for categorical label, convert titanic binary numeric label to categorical ["Survived"],["Not Survived"]
        df["Survival"] = wrapper._perform_label_decoding(df["Survival"])
    else:
        #We only test NaN in categorical features and numerical target
        df["Name"][0] = np.nan
        df["Name"][10] = np.nan
        df["Name"][20] = np.nan
        df["Name"][30] = np.nan
        df["Name"][40] = np.nan
        df["Name"][50] = np.nan
        df["Name"][60] = np.nan
        df["Name"][70] = np.nan
        df["Name"][80] = np.nan
    X_train, X_test, y_train, y_test = train_test_split(df.drop(["Survival"], axis=1), df["Survival"], test_size=0.2, random_state=42)

    X_train = wrapper._preprocessor(X_train)

    if categorical_label_test == True:
        y_train = wrapper._perform_label_encoding(y_train)
        
    model.fit(X_train, y_train)
    return CategoricalModelWrapper(model, categorical_label_test), X_test, y_test

def test_NaN_containing_categorical_dataset():
    _wrapper, _test_X, _test_y = generate_categorical_dataset_model_wrapper()
    explainer = ClassifierExplainer(
                    _wrapper, _test_X, _test_y)
    dashboard = ExplainerDashboard(explainer)
    assert "NaN" in explainer.categorical_dict["Name"]
    
def test_categorical_label():
    _wrapper, _test_X, _test_y = generate_categorical_dataset_model_wrapper(True)
    explainer = ClassifierExplainer(
                    _wrapper, _test_X, _test_y)
    dashboard = ExplainerDashboard(explainer)
    assert "Survived" in explainer.labels
