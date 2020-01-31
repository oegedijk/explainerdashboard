import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, r2_score

from explainerdashboard.explainers import *
from explainerdashboard.dashboards import *
from explainerdashboard.datasets import *

# RandomForestRegressor

# RandomForestClassifier:

X_train, y_train, X_test, y_test = titanic_survive()
train_names, test_names = titanic_names()

model = RandomForestClassifier(n_estimators=50, max_depth=5)
model.fit(X_train, y_train)

explainer = RandomForestClassifierBunch(model, X_test, y_test, roc_auc_score, 
                               cats=['Sex', 'Deck', 'Embarked'],
                               idxs=test_names, #names of passengers 
                               labels=['Not survived', 'Survived'])

db = ExplainerDashboard(explainer,
#                         model_summary=True,
#                         contributions=True,
#                         shap_dependence=True,
#                         shap_interaction=True,
                        shadow_trees=True)
db.run(8052, debug=True)