__all__ = ['titanic_survive', 
            'titanic_fare', 
            'titanic_embarked', 
            'titanic_names']

import numpy as np
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


def titanic_embarked():
    d_train2 = d_train.copy()
    d_train2 = d_train2[d_train2.Embarked_Unknown==0]
    X_train = d_train2.drop(['Embarked_Cherbourg', 'Embarked_Queenstown', 
        'Embarked_Southampton', 'Embarked_Unknown', 'Name'], axis=1)
    y_train = pd.Series(np.where(d_train2.Embarked_Queenstown==1, 0, 
                            np.where(d_train2.Embarked_Southampton==1, 1, 
                                np.where(d_train2.Embarked_Cherbourg==1, 2, 3))), 
                           
                    name="Embarked")
    X_test = d_test.drop(['Embarked_Cherbourg', 'Embarked_Queenstown', 
        'Embarked_Southampton', 'Embarked_Unknown', 'Name'], axis=1)
    y_test = pd.Series(np.where(d_test.Embarked_Queenstown==1, 0, 
                            np.where(d_test.Embarked_Southampton==1, 1, 
                                np.where(d_test.Embarked_Cherbourg==1, 2, 3))), 
                            
                    name="Embarked")
    return X_train, y_train, X_test, y_test


def titanic_names():
    return (d_train['Name'].values.tolist(), d_test['Name'].values.tolist())
