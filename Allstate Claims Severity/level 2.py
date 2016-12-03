N_FOLDS = 2
kf = KFold(ntrain, n_folds = N_FOLDS, shuffle = True, random_state = 0)

list_result = [Predict_XGB, Predict_Lasso, Predict_Ridge, Predict_XG ]
list_result_test = [Predict_XGB_test, 
                    Predict_Lasso_test ,Predict_Ridge_test, Predict_XG_test ]

result_lvl1 = pd.concat(list_result,axis=1) 
result_lvl1_test = pd.concat(list_result_test,axis=1)

#train_cont14 = pd.DataFrame(train_test.iloc[:ntrain,:].cont14)
#train_cont14.reset_index(inplace=True)

#test_cont14 = pd.DataFrame(train_test.iloc[ntrain:,:].cont14)
#test_cont14.reset_index(inplace=True)

#cont14
#result_lvl1 = pd.concat([result_lvl1, train_cont14], axis=1)
#result_lvl1_test = pd.concat([result_lvl1_test, test_cont14], axis=1)

x_train_lvl2 = np.array(result_lvl1)
x_test_lvl2 = np.array(result_lvl1_test)

Predict_LIN2 = create_pd('Predict_LIN2','_LIN2',x_train) 
Predict_LIN2_test2 = create_pd('Predict_LIN2','_LIN2',x_test) 

Predict_RD2 = create_pd('Predict_RD2','_RD2',x_train)
Predict_Ridge_test2 = create_pd('Predict_RD_test2','_RD2',x_test)


Predict_XG2 = create_pd('Predict_XG2','_XG2',x_train) 
Predict_XG2_test2 = create_pd('Predict_XG2_test2','_XG2',x_test) 
#Predict_ADA = create_pd('Predict_ADA','_AD')

Predict_XG2, list_mae_xg2, Predict_XG2_test2 = clf_sklearn(xgb.XGBRegressor(n_estimators = 700, learning_rate = 0.01, max_depth = 3),'_XG2',x_train_lvl2,y_train,x_test_lvl2,Predict_XG2_test2) 
#Predict_LIN2, list_mae_LIN2, Predict_LIN2_test2 = clf_sklearn(LinearRegression(),'_XG2',x_train_lvl2,y_train,x_test_lvl2,Predict_LIN2_test2) 
#Predict_RD2, list_mae_brd2, Predict_Ridge_test2 = clf_sklearn(BayesianRidge(normalize=True),'_RD2',x_train_lvl2,y_train,x_test_lvl2,Predict_Ridge_test2) 
#Predict_ADA, list_mae_Ada = clf_sklearn(AdaBoostRegressor(n_estimators = 100, learning_rate=0.01),'_AD',x_train_lvl2,y_train) 
#print(np.mean(list_mae_LIN2))
print(np.mean(list_mae_xg2))


test_sub = pd.DataFrame(np.hstack([pd.DataFrame(id_test),  np.exp(pd.DataFrame(Predict_XG2_test2))]))
test_sub[[0]]=test_sub[[0]].astype(int)
test_sub[[0,1]].to_csv("C:/Users/Ju/Desktop/Kaggle/Allsafe/submit_stacking.csv",index=False,
header=['id','loss'],dtype=float)  