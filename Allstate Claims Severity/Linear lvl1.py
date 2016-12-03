#Predict_Bay_Ridge, list_mae_BRid, Predict_Bay_Ridge_test = clf_sklearn(BayesianRidge(normalize=True),'_BRD',x_train,y_train,x_test,Predict_Bay_Ridge_test)
Predict_Lasso, list_mae_Las, Predict_Lasso_test = clf_sklearn(LassoCV(alphas = [0.01,0.001,0.0001]),'_LS',x_train,y_train,x_test,Predict_Lasso_test) 
Predict_Ridge, list_mae_Rid, Predict_Ridge_test = clf_sklearn(Ridge(alpha = 50),'_RD',x_train,y_train,x_test,Predict_Ridge_test) 

#xg_params = {
#    'n_estimators': 800,
#    'learning_rate': 0.01,
#    'subsample': 0.5,
#    'max_depth': 6,
#    'colsample_bytree': 0.5,
#    'min_child_weight':3
#}

params = {
    'seed': 1,
    'colsample_bytree': 0.3085,
    'subsample': 0.9930,
    'eta': 0.01,
    #'lambda':5,
    'gamma': 0.49,
    'booster' :  'gbtree',    
    'objective': 'reg:linear',
    'max_depth': 7,
    'min_child_weight': 4.28
}
Predict_XG, list_mae_xg, Predict_XG_test = clf_xgb(1000,params, x_train,y_train,x_test,Predict_XG_test,'_2')
print(list_mae_xg)

print('MAE Random Forest: ',mean_absolute_error(np.exp(y_train),np.exp(Predict_RF)))
print('MAE Extra Trees: ',mean_absolute_error(np.exp(y_train),np.exp(Predict_ET)))
print('MAE XGboost: ',mean_absolute_error(np.exp(y_train),np.exp(Predict_XG)))
print('MAE XGboost: ',mean_absolute_error(np.exp(y_train),np.exp(Predict_XGB)))
print('MAE Lasso CV: ',mean_absolute_error(np.exp(y_train),np.exp(Predict_Lasso)))
print('MAE Ridge: ',mean_absolute_error(np.exp(y_train),np.exp(Predict_Ridge)))
print('MAE Bayesian Ridge: ',mean_absolute_error(np.exp(y_train),np.exp(Predict_Bay_Ridge)))