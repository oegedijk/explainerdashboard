

# def test_ngboost():
#     try:
#         import ngboost
#     except Exception as e:
#         print("Skipping test_ngboost!")
#         return


# try:
#         import lightgbm
#     except:
#         print("Skipping test_lightgbm!")
#         return
#     import shap

#     # train lightgbm model
#     X, y = shap.datasets.boston()
#     model = lightgbm.sklearn.LGBMRegressor(categorical_feature=[8])
#     model.fit(X, y)

#     # explain the model's predictions using SHAP values
#     ex = shap.TreeExplainer(model)



# def test_catboost():
#     try:
#         import catboost
#         from catboost.datasets import amazon
#     except:
#         print("Skipping test_catboost!")
#         return
#     import shap

#     # train catboost model
#     X, y = shap.datasets.boston()
#     X["RAD"] = X["RAD"].astype(np.int)
#     model = catboost.CatBoostRegressor(iterations=300, learning_rate=0.1, random_seed=123)
#     p = catboost.Pool(X, y, cat_features=["RAD"])
#     model.fit(p, verbose=False, plot=False)

#     # explain the model's predictions using SHAP values
#     ex = shap.TreeExplainer(model)
#     shap_values = ex.shap_values(p)

#     predicted = model.predict(X)

#     assert np.abs(shap_values.sum(1) + ex.expected_value - predicted).max() < 1e-4, \
#         "SHAP values don't sum to model output!"
    
#     train_df, _ = amazon()
#     ix = 100
#     X_train = train_df.drop('ACTION', axis=1)[:ix]
#     y_train = train_df.ACTION[:ix]
#     X_val = train_df.drop('ACTION', axis=1)[ix:ix+20]
#     y_val = train_df.ACTION[ix:ix+20]
#     model = catboost.CatBoostClassifier(iterations=100, learning_rate=0.5, random_seed=12)
#     model.fit(
#         X_train,
#         y_train,
#         eval_set=(X_val, y_val),        
#         verbose=False,
#         plot=False
#     )
#     shap.TreeExplainer(model)
