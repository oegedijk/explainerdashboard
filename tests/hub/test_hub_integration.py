from unittest import TestCase

from sklearn.ensemble import RandomForestClassifier

from explainerdashboard import *
from explainerdashboard.datasets import *
from explainerdashboard.custom import *

from unittest import TestCase

class UserTest(TestCase):
   def setUp(self):
        X_train, y_train, X_test, y_test = titanic_survive()
        model = RandomForestClassifier(n_estimators=5, max_depth=2).fit(X_train, y_train)
        explainer = ClassifierExplainer(model, X_test, y_test)
        db1 = ExplainerDashboard(explainer, description="Super interesting dashboard")
        db2 = ExplainerDashboard(explainer, title="Dashboard Two", 
                    logins=[['user', 'password']], name='db2')
        self.hub = ExplainerHub([db1, db2])
        self.hub.app.config["TESTING"] = True
        self.hub.app.config["WTF_CSRF_CHECK_DEFAULT"] = False
        self.client = self.hub.app.test_client()
        self._ctx = self.hub.app.test_request_context()
        self._ctx.push()

   def tearDown(self):
       if self._ctx is not None:
           self._ctx.pop()

   def test_user_index(self):
       with self.client:
            data = {"username": "user", "password": "password", "next": "/"}       
            response = self.client.post("/login/", data=data)
            self.assertEqual(response.status_code, 200)

