from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt
import xgboost as xgb

x_train, x_test, y_train, y_test = train_test_split(train_select,Target,test_size=0.30, 
                                                    random_state=30,stratify=Target)
#simple gmb
#clf_gbm=GradientBoostingClassifier(n_estimators=100,random_state=10).fit(x_train,y_train)
#clf_probs = clf_gbm.predict_proba(x_test)
#print(log_loss(y_test, clf_probs))


#feature importance with gbm
#import matplotlib.pyplot as plt
#feature_importance = clf_gbm.feature_importances_
#feature_importance = 100.0 * (feature_importance / feature_importance.max())
#sorted_idx = np.argsort(feature_importance)
#pos = np.arange(sorted_idx.shape[0]) + .5
#plt.subplot(1, 2, 2)
#plt.barh(pos[134:], feature_importance[sorted_idx][134:], align='center')
#plt.yticks(pos[134:], x_train.columns.values[sorted_idx][134:])
#plt.xlabel('Relative Importance')
#plt.title('Variable Importance')
#plt.show()

dtrain = xgb.DMatrix(x_train,y_train,missing = -9999)
dtest = xgb.DMatrix(x_test,y_test,missing = -9999)

param = {'max_depth':7, 'eta':0.1, 'silent':1, 'objective':'multi:softprob','num_class':5,
        'eval_metric':'mlogloss','subsample':0.75,'colsample_bytree':0.85}
#watchlist  = [(dtest,'eval'), (dtrain,'train')]
num_round = 125
#bst = xgb.train(param, dtrain, num_round)
#ypred2 = bst.predict(dtest)
#print(log_loss(y_test, ypred2))

param = {'max_depth':6, 'eta':0.1, 'silent':1, 'objective':'multi:softprob','num_class':5,
        'eval_metric':'mlogloss','subsample':0.85,'colsample_bytree':0.75}
num_round = 125
#bst2 = xgb.train(param, dtrain, num_round)
#ypred22 = bst2.predict(dtest)
#print(log_loss(y_test, ypred2))
#print(log_loss(y_test, (ypred2+ypred22)/2))

param = {'max_depth':8, 'eta':0.1, 'silent':1, 'objective':'multi:softprob','num_class':5,
        'eval_metric':'mlogloss','subsample':0.65,'colsample_bytree':0.75}
#watchlist  = [(dtest,'eval'), (dtrain,'train')]
num_round = 125
#bst3 = xgb.train(param, dtrain, num_round)
#ypred222 = bst3.predict(dtest)

#print(log_loss(y_test, ypred22))
#print(log_loss(y_test, (ypred2+ypred22)/2))
#print(log_loss(y_test, (ypred2+ypred22+ypred222)/3))

param = {'max_depth':9, 'eta':0.1, 'silent':1, 'objective':'multi:softprob','num_class':5,
        'eval_metric':'mlogloss','subsample':0.55,'colsample_bytree':0.65}
#watchlist  = [(dtest,'eval'), (dtrain,'train')]
num_round = 125
#bst4 = xgb.train(param, dtrain, num_round)
#ypred2222 = bst4.predict(dtest)

#print(log_loss(y_test, ypred22))
#print(log_loss(y_test, (ypred2+ypred22)/2))
#print(log_loss(y_test, (ypred2+ypred22+ypred222)/3))
#print(log_loss(y_test, (ypred2+ypred22+ypred222+ypred2222)/4))

param = {'max_depth':12, 'eta':0.1, 'silent':1, 'objective':'multi:softprob','num_class':5,
        'eval_metric':'mlogloss','subsample':1,'colsample_bytree':1}
#watchlist  = [(dtest,'eval'), (dtrain,'train')]
num_round = 125
#bst5 = xgb.train(param, dtrain, num_round)
#ypred22222 = bst5.predict(dtest)

#print(log_loss(y_test, ypred22))
#print(log_loss(y_test, (ypred2+ypred22)/2))
#print(log_loss(y_test, (ypred2+ypred22+ypred222)/3))
#print(log_loss(y_test, (ypred2+ypred22+ypred222+ypred2222)/4))
#print(log_loss(y_test, (ypred2+ypred22+ypred222+ypred2222+ypred22222)/5))


dtrain = xgb.DMatrix(train_select,Target,missing = -9999)
dtest = xgb.DMatrix(test_select,missing = -9999)
num_round = 125

param1 = {'max_depth':7, 'eta':0.1, 'silent':1, 'objective':'multi:softprob','num_class':5,
        'eval_metric':'mlogloss','subsample':0.75,'colsample_bytree':0.85}


param2 = {'max_depth':6, 'eta':0.1, 'silent':1, 'objective':'multi:softprob','num_class':5,
        'eval_metric':'mlogloss','subsample':0.85,'colsample_bytree':0.75}

param3 = {'max_depth':8, 'eta':0.1, 'silent':1, 'objective':'multi:softprob','num_class':5,
        'eval_metric':'mlogloss','subsample':0.65,'colsample_bytree':0.75}

param4 = {'max_depth':9, 'eta':0.1, 'silent':1, 'objective':'multi:softprob','num_class':5,
        'eval_metric':'mlogloss','subsample':0.55,'colsample_bytree':0.65}

param5 = {'max_depth':12, 'eta':0.1, 'silent':1, 'objective':'multi:softprob','num_class':5,
        'eval_metric':'mlogloss','subsample':1,'colsample_bytree':1}

bst1 = xgb.train(param1, dtrain, num_round)
bst2 = xgb.train(param2, dtrain, num_round)
bst3 = xgb.train(param3, dtrain, num_round)
bst4 = xgb.train(param3, dtrain, num_round)
bst5 = xgb.train(param3, dtrain, num_round)

ypred_submit = (bst1.predict(dtest) + bst2.predict(dtest) + bst3.predict(dtest) +  bst4.predict(dtest) +  bst5.predict(dtest))/5


ypred_submit=pd.DataFrame(ypred_submit)

submission = pd.DataFrame()
submission["id"] = Id_ts.values
submission["Adoption"]= ypred_submit[2]
submission["Died"]= ypred_submit[4]
submission["Euthanasia"]= ypred_submit[1]
submission["Return_to_owner"]= ypred_submit[0]
submission["Transfer"]= ypred_submit[3]

submission.to_csv("sub.csv",index=False)