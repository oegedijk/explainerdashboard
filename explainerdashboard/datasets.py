__all__ = ['titanic_survive', 'titanic_fare', 'titanic_class', 'titanic_names']

import pandas as pd
from pathlib import Path


d_train = pd.read_csv(Path(__file__).resolve().parent / 'datasets'/ 'titanic_train.csv')
d_test = pd.read_csv(Path(__file__).resolve().parent / 'datasets'/'titanic_test.csv')

def titanic_survive():
    X_train = d_train.drop(['Survived', 'Name'], axis=1)
    y_train = d_train['Survived']
    X_test = d_test.drop(['Survived', 'Name'], axis=1)
    y_test = d_test['Survived']
    return X_train, y_train, X_test, y_test

def titanic_fare():
    X_train = d_train.drop(['Fare', 'Name'], axis=1)
    y_train = d_train['Fare']
    X_test = d_test.drop(['Fare', 'Name'], axis=1)
    y_test = d_test['Fare']
    return X_train, y_train, X_test, y_test

def titanic_class():
    X_train = d_train.drop(['PassengerClass', 'Name'], axis=1)
    y_train = d_train['PassengerClass'] - 1
    X_test = d_test.drop(['PassengerClass', 'Name'], axis=1)
    y_test = d_test['PassengerClass'] - 1
    return X_train, y_train, X_test, y_test


def titanic_names():
    return (d_train['Name'].values.tolist(), d_test['Name'].values.tolist())
