# from xgboost import XGBClassifier

# import dash

# from explainerdashboard.explainers import ClassifierExplainer
# from explainerdashboard.datasets import titanic_survive, titanic_names
# from explainerdashboard.dashboard_tabs import ImportancesTab
# from explainerdashboard.dashboards import *

# def get_explainer():
#     X_train, y_train, X_test, y_test = titanic_survive()
#     train_names, test_names = titanic_names()

#     model = XGBClassifier()
#     model.fit(X_train, y_train)

#     explainer = ClassifierExplainer(
#                         model, X_test, y_test, 
#                         cats=['Sex', 'Cabin', 'Embarked'],
#                         labels=['Not survived', 'Survived'],
#                         idxs=test_names)
#     return explainer

# def test_importances_tab(dash_duo):
#     explainer = get_explainer()

#     db = ExplainerDashboard(explainer, ImportancesTab, title="testing")

#     dash_duo.start_server(db.app)
#     dash_duo.wait_for_text_to_equal("h1", "testing", timeout=4)
#     assert dash_duo.get_logs() == [], "browser console should contain no error"