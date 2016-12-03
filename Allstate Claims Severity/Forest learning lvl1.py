#creation of dataframes
Predict_RF = create_pd('Predict_RF','_RF',x_train) 
Predict_ET = create_pd('Predict_ET','_ET',x_train) 
Predict_XG = create_pd('Predict_XG','_XGB_2',x_train) 
Predict_Lasso = create_pd('Predict_Lasso','_LS',x_train) 
Predict_Ridge = create_pd('Predict_Ridge','_RD',x_train)
Predict_Bay_Ridge = create_pd('Predict_Bay_Ridge','_BRD',x_train)

Predict_RF_test = create_pd('Predict_RF_test','_RF',x_test) 
Predict_ET_test = create_pd('Predict_ET_test','_ET',x_test) 
Predict_XG_test = create_pd('Predict_XG_test','_XGB_2',x_test) 
Predict_Lasso_test = create_pd('Predict_Lasso_test','_LS',x_test) 
Predict_Ridge_test = create_pd('Predict_Ridge_test','_RD',x_test)
Predict_Bay_Ridge_test = create_pd('Predict_Bay_Ridge_test','_BRD',x_test)


rf_params = {
    'n_jobs': 16,
    'n_estimators': 750,
    'max_features': 0.8,
    'max_depth': 20,
    'min_samples_leaf': 2,
}

#Predict_RF, list_mae_rf, Predict_RF_test = clf_sklearn(RandomForestRegressor(**rf_params),'_RF',x_train,y_train,x_test,Predict_RF_test) 

et_params = {
    'n_jobs': 16,
    'n_estimators': 750,
    'max_features': 0.8,
    'max_depth': 20,
    'min_samples_leaf': 2,
}

#Predict_ET, list_mae_et, Predict_ET_test = clf_sklearn(ExtraTreesRegressor(**et_params),'_ET',x_train,y_train,x_test,Predict_ET_test) 


#print(list_mae_rf)
#print(list_mae_et)