__all__ = [
    "titanic_survive",
    "titanic_fare",
    "titanic_embarked",
    "titanic_names",
    "feature_descriptions",
    "train_csv",
    "test_csv",
]

import numpy as np
import pandas as pd
from pathlib import Path

train_csv = Path(__file__).resolve().parent / "datasets" / "titanic_train.csv"
test_csv = Path(__file__).resolve().parent / "datasets" / "titanic_test.csv"
d_train = pd.read_csv(train_csv)
d_test = pd.read_csv(test_csv)

feature_descriptions = {
    "Sex": "Gender of passenger",
    "Gender": "Gender of passenger",
    "Deck": "The deck the passenger had their cabin on",
    "PassengerClass": "The class of the ticket: 1st, 2nd or 3rd class",
    "Fare": "The amount of money people paid for their ticket",
    "Embarked": "the port where the passenger boarded the Titanic. Either Southampton, Cherbourg or Queenstown",
    "Age": "Age of the passenger",
    "No_of_siblings_plus_spouses_on_board": "The sum of the number of siblings plus the number of spouses on board",
    "No_of_parents_plus_children_on_board": "The sum of the number of parents plus the number of children on board",
}


def titanic_survive():
    X_train = d_train.drop(["Survival", "Name"], axis=1)
    X_train.index = d_train.Name
    X_train.index.name = "Passenger"
    y_train = d_train["Survival"]
    X_test = d_test.drop(["Survival", "Name"], axis=1)
    X_test.index = d_test.Name
    X_test.index.name = "Passenger"
    y_test = d_test["Survival"]
    return X_train, y_train, X_test, y_test


def titanic_fare():
    X_train = d_train.drop(["Fare", "Name"], axis=1)
    X_train.index = d_train.Name
    X_train.index.name = "Passenger"
    y_train = d_train["Fare"]
    X_test = d_test.drop(["Fare", "Name"], axis=1)
    X_test.index = d_test.Name
    X_test.index.name = "Passenger"
    y_test = d_test["Fare"]
    return X_train, y_train, X_test, y_test


def titanic_embarked():
    d_train2 = d_train.copy()
    d_train2 = d_train2[d_train2.Embarked_Unknown == 0]
    X_train = d_train2.drop(
        [
            "Embarked_Cherbourg",
            "Embarked_Queenstown",
            "Embarked_Southampton",
            "Embarked_Unknown",
            "Name",
        ],
        axis=1,
    )
    X_train.index = d_train2.Name
    X_train.index.name = "Passenger"

    y_train = pd.Series(
        np.where(
            d_train2.Embarked_Queenstown == 1,
            0,
            np.where(
                d_train2.Embarked_Southampton == 1,
                1,
                np.where(d_train2.Embarked_Cherbourg == 1, 2, 3),
            ),
        ),
        name="Embarked",
    )
    X_test = d_test.drop(
        [
            "Embarked_Cherbourg",
            "Embarked_Queenstown",
            "Embarked_Southampton",
            "Embarked_Unknown",
            "Name",
        ],
        axis=1,
    )
    X_test.index = d_test.Name
    X_test.index.name = "Passenger"
    y_test = pd.Series(
        np.where(
            d_test.Embarked_Queenstown == 1,
            0,
            np.where(
                d_test.Embarked_Southampton == 1,
                1,
                np.where(d_test.Embarked_Cherbourg == 1, 2, 3),
            ),
        ),
        name="Embarked",
    )
    return X_train, y_train, X_test, y_test


def titanic_names(train_only=False, test_only=False):
    if train_only:
        return d_train["Name"].values.tolist()
    if test_only:
        return d_test["Name"].values.tolist()
    return (d_train["Name"].values.tolist(), d_test["Name"].values.tolist())
