#functions and parameters
N_FOLDS = 2
kf = KFold(ntrain, n_folds = N_FOLDS, shuffle = True, random_state = 0)

def create_pd(name,algo,x):
    a = np.zeros(shape = len(x))
    name = pd.DataFrame(a)
    name.columns = ['Predict'+algo]
    return name

def clf_sklearn(clf, algo,X_train,Y_train,X_test,Predict_test):
    list_mae = []
    Pred = np.zeros(shape = len(X_train))
    for i, (train_index, test_index) in enumerate(kf):
        x_tr = X_train[train_index]
        y_tr = Y_train[train_index]
        x_te = X_train[test_index]
        y_te = Y_train[test_index]

        clf.fit(x_tr, y_tr)
        list_mae.append(mean_absolute_error(np.exp(y_te),np.exp(clf.predict(x_te))))
        
        Predict_new = clf.predict(x_te)
        Pred[test_index] = Predict_new
        
        DataFrame_test = pd.DataFrame(clf.predict(X_test))
        DataFrame_test.columns = ['Predict'+algo]
        
        Predict_test = Predict_test.add(DataFrame_test)
                         
    Pred = pd.DataFrame(Pred)
    Pred.columns = ['Predict'+algo]
    print(algo, 'completed...')
    return Pred/(N_FOLDS-1), list_mae, Predict_test/N_FOLDS     

def clf_xgb(num_rounds,params, X_train,Y_train,X_test,Predict_test,algo):
    list_mae = []
    Pred = np.zeros(shape = len(X_train))
    for i, (train_index, test_index) in enumerate(kf):
        x_tr = X_train[train_index]
        y_tr = Y_train[train_index]
        x_te = X_train[test_index]
        y_te = Y_train[test_index]
        
        xgdmat = xgb.DMatrix(X_train, Y_train)
        bst = xgb.train(params, xgdmat, num_boost_round = num_rounds)
        test_xgb = xgb.DMatrix(x_te)
        list_mae.append(mean_absolute_error(np.exp(y_te),np.exp(bst.predict(test_xgb))))
        
        Predict_new = bst.predict(test_xgb)
        Pred[test_index] = Predict_new
        
        xgb_test = xgb.DMatrix(X_test)
        DataFrame_test = pd.DataFrame(bst.predict(xgb_test))
        DataFrame_test.columns = ['Predict_XGB'+algo]
        
        Predict_test = Predict_test.add(DataFrame_test)
        
    Pred = pd.DataFrame(Pred)
    Pred.columns = ['Predict_XGB'+algo]
    print('XGB completed...')
    return Pred/(N_FOLDS-1), list_mae, Predict_test/N_FOLDS    

Predict_XGB = create_pd('Predict_XGB','_XGB_1',x_train)
Predict_XGB_test = create_pd('Predict_XGB_test0','_XGB_1',x_test)

params = {'eta': 0.01, 'seed':0, 'subsample': 0.5, 'colsample_bytree': 0.5, 
             'objective': 'reg:linear', 'max_depth':6, 'min_child_weight':3,'n_jobs':4} 

#Predict_XGB, list_mae_XGB, Predict_XGB_test =clf_xgb(9000,params, x_train,y_train,x_test,Predict_XGB_test)
#print(list_mae_XGB)

#7000
Predict_XGB, list_mae_XGB, Predict_XGB_test =clf_xgb(7000,params, x_train,y_train,x_test,Predict_XGB_test,'_1')
print(list_mae_XGB)

test_sub = pd.DataFrame(np.hstack([pd.DataFrame(id_test),  np.exp(pd.DataFrame(Predict_XGB_test))]))
test_sub[[0]]=test_sub[[0]].astype(int)
test_sub[[0,1]].to_csv("C:/Users/Ju/Desktop/Kaggle/Allsafe/submit_XG1.csv",index=False,
header=['id','loss'],dtype=float)  