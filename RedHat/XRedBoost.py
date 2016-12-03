import xgboost as xgb
from sklearn import metrics

param = {
'eta':0.01,#0.1
'max_depth':10,#10
'subsample':0.7,#0.9
'colsample_bytree':0.7,#0.85
'objective':'binary:logistic',
'booster':'gbtree',#gblinear
'eval_metric':'auc',
'silent':1,
'seed':42
#'min_child_weight':1 #1
}

param1 = {
'eta':0.01,#0.1
'max_depth':11,#10
'subsample':0.9,#0.9
'colsample_bytree':0.85,#0.85
'objective':'binary:logistic',
'booster':'gbtree',#gblinear
'eval_metric':'auc',
'silent':1,
'seed':42
#'min_child_weight':1 #1
}

param2 = {
'eta':0.01,#0.1
'max_depth':15,#10
'subsample':0.7,#0.9
'colsample_bytree':0.7,#0.85
'objective':'binary:logistic',
'booster':'gbtree',#gblinear
'eval_metric':'auc',
'silent':1,
'seed':42
#'min_child_weight':1 #1
}

param3 = {
'eta':0.01,#0.1
'max_depth':25,#10
'subsample':0.7,#0.9
'colsample_bytree':0.7,#0.85
'objective':'binary:logistic',
'booster':'gbtree',#gblinear
'eval_metric':'auc',
'silent':1,
'seed':42
#'min_child_weight':1 #1
}


#Test V0
xgmat = xgb.DMatrix(x_train[features], label=y_train)
xgmat_test = xgb.DMatrix(x_test[features],label=y_test)
watchlist=[(xgmat,'train'),(xgmat_test,'eval')]

bst = xgb.train(param,xgmat,100,watchlist,verbose_eval=True)
pred = pd.DataFrame(bst.predict(xgmat_test))

print(metrics.roc_auc_score(y_test,pred))

#submit V0
xgmat = xgb.DMatrix(train_num[features], label=to_predict)
xgmat_test = xgb.DMatrix(test_num[features])
    
def xgb_xgb(parameters):
    bst = xgb.train(parameters,xgmat,1000)
    pred = pd.DataFrame(bst.predict(xgmat_test))
    return pred

#pred1 = xgb_xgb(param)
pred2 = xgb_xgb(param1)
pred3 = xgb_xgb(param2)

pred5 = xgb_xgb(param3)
submit('xgb_1000_pred5',pred5)
t
submit('xgb_1000_pred3',pred1+pred3)
#submit('xgb_100_pred2',pred1+pred3)
#submit('xgb_100_pred3',pred1+pred2)
submit('xgb_1000_pred2pred3',pred2+pred3)
#submit('xgb_100_pred1pred2pred3',pred1+pred2+pred3)

#submit V1 with CV
from sklearn.cross_validation import StratifiedKFold

skf = StratifiedKFold(to_predict,n_folds=10,random_state=42)

train_skf=np.array(train_num[features])

i = 0
xgmat_test_final = xgb.DMatrix(test_num[features])

for train_index, test_index in skf:
    i = i + 1
    x_train_skf, x_test_skf = train_skf[train_index], train_skf[test_index] 
    y_train_skf, y_test_skf = to_predict[train_index], to_predict[test_index]

    xgmat = xgb.DMatrix(x_train_skf, label=y_train_skf)
    xgmat_test = xgb.DMatrix(x_test_skf)
    bst = xgb.train(param3,xgmat,1000)
    pred = pd.DataFrame(bst.predict(xgmat_test))
    print(metrics.roc_auc_score(y_test_skf,pred))
    if i == 1:
        pred_final =  pd.DataFrame(bst.predict(xgmat_test_final))
    if i > 1:
        pred_final = pred_final + pd.DataFrame(bst.predict(xgmat_test_final))
        
submit('xgb_1000_cv10',pred_final)
