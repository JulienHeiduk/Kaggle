from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
import xgboost as xgb
from sklearn.metrics import log_loss
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import BaggingClassifier

train_select=pd.DataFrame(train2.select_dtypes(include=['float64','int64','int32','float32']))
#target=pd.DataFrame(train_select['Target'])

#train_select.drop(
#['191_br1','97_br2','198_br1','197_br1','196_br1','195_br1','194_br1',
# '193_br1','98_br2','44_br2','190_br1','2_sh1','56_br2','184_br1','7_sh1',
# '182_br1','1_sh2','180_br1','200_br1','201_br1','202_br1','203_br1',
# '221_br1','220_br1','26_col2','218_br1','217_br1','216_br1','215_br1'], axis=1,inplace=True)

#train_select.drop('Target',axis=1,inplace=True)
train_select=train_select.fillna(-9999)
#from sklearn.feature_selection import SelectKBest
#selection = SelectKBest(k=10)
#train_selection=selection.fit_transform(train_select,target)

x_train, x_test, y_train, y_test = train_test_split(train_select,target,test_size=0.30, 
                                                    random_state=30,stratify=target)
print(x_train.shape)
#gbm
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt

clf_gbm=GradientBoostingClassifier(n_estimators=100,random_state=10).fit(x_train,y_train)
clf_probs = clf_gbm.predict_proba(x_test)
print(log_loss(y_test, clf_probs))
#from sklearn.ensemble import RandomForestClassifier

#clf_xgbm=xgb.XGBClassifier(n_estimators=10,objective='multi:softprob',missing=-9999).fit(x_train,y_train)
#clf_probs_x = clf_xgbm.predict_proba(x_test)
#print(log_loss(y_test, clf_probs_x))

import matplotlib.pyplot as plt
feature_importance = clf_gbm.feature_importances_
# make importances relative to max importance
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.subplot(1, 2, 2)
plt.barh(pos[570:], feature_importance[sorted_idx][570:], align='center')
plt.yticks(pos[570:], x_train.columns.values[sorted_idx][570:])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.show()

print(x_train.columns.values[sorted_idx][0:])
#0.630956
dtrain = xgb.DMatrix(x_train,y_train,missing = -9999)
dtest = xgb.DMatrix(x_test,y_test,missing = -9999)
param = {'max_depth':6, 'eta':0.2, 'silent':1, 'objective':'multi:softprob','num_class':5,
        'eval_metric':'mlogloss','subsample':0.75,'colsample_bytree':0.85}
watchlist  = [(dtest,'eval'), (dtrain,'train')]
num_round = 125
bst = xgb.train(param, dtrain, num_round, watchlist)
 
print ('start testing prediction from first n trees')
### predict using first 1 tree
#label = dtest.get_label()
#ypred1 = bst.predict(dtest, ntree_limit=1)

#ypred2 = bst.predict(dtest)

#print(log_loss(y_test, ypred2))


from sklearn.cross_validation import StratifiedKFold

def estimate_performance_xgboost(training_file, target_train, test_file, y_val, param, num_round, folds,mode):
    # Load training data
    X=training_file.copy()   
    X['a']=range(0,len(target_train))
    X=X.set_index(['a'],drop=True)
    
    labels=pd.DataFrame(target_train)
    labels['a']=range(0,len(target_train))
    labels=labels.set_index(['a'],drop=True)
    target_train.drop('a',axis=1,inplace=True)
    
    xgmat_trainingfile = xgb.DMatrix(training_file, label=target_train )
    xgmat_test_sub = xgb.DMatrix(test_file)

    if mode=='base':
        kf = StratifiedKFold(target_train['Target'], n_folds=folds)
        i=0

        for train_indices, test_indices in kf:
            i=i+1
            X_train, X_test = X.ix[train_indices], X.ix[test_indices]
            y_train, y_test = labels.ix[train_indices], labels.ix[test_indices]
            
            # construct xgboost.DMatrix
            xgmat = xgb.DMatrix(X_train, label=y_train )    
        
            # Construct matrix for test set
            xgmat_test = xgb.DMatrix(X_test, label=y_test)
            watchlist=[(xgmat,'train'),(xgmat_test,'eval')]
            bst = xgb.train(param, xgmat, num_round,watchlist,verbose_eval=False,early_stopping_rounds=10000)
            
	    global y_out_test_c
	    global y_pred
            if i==1:
                #Initialisation prediction
                y_out = pd.DataFrame(bst.predict(xgmat_test))
                y_out_test=pd.DataFrame(bst.predict(xgmat_test_sub))
      
            if i>1:
                y_out = y_out + pd.DataFrame(bst.predict(xgmat_test))
                y_out_test = y_out_test + pd.DataFrame(bst.predict(xgmat_test_sub))
 		    #y_out_test_c = y_out_test.join(pd.DataFrame(bst.predict(xgmat_test_sub)))


            if i==folds:
                	y_pred=pd.DataFrame(y_out_test/folds)
                	print("resultats:")
                	pred=y_pred
			return y_pred, log_loss(y_val, y_pred.as_matrix())#metrics.roc_auc_score(y_val,y_pred)

    if mode=='RS':
        for j in range(0,2):
            #Cross validation
            kf = StratifiedKFold(target_train, n_folds=folds,shuffle=True,random_state=j)
            i=0
            print("RS numero" % j)        
            for train_indices, test_indices in kf:
                i=i+1
                X_train, X_test = X.ix[train_indices], X.ix[test_indices]
                y_train, y_test = labels.ix[train_indices], labels.ix[test_indices]
                
                # construct xgboost.DMatrix
                xgmat = xgb.DMatrix(X_train, label=y_train )   
         
                # Construct matrix for test set
                xgmat_test = xgb.DMatrix(X_test, label=y_test)
                watchlist=[(xgmat,'train'),(xgmat_test,'eval')]
                bst = xgb.train(param, xgmat, num_round,watchlist,verbose_eval=False,early_stopping_rounds=50)
            
                if i==1:
                    #Initialisation prediction
                    y_out = pd.DataFrame(bst.predict(xgmat_test))
                    y_out_test=pd.DataFrame(bst.predict(xgmat_test_sub))

            
                if i>1:
                    y_out = y_out + pd.DataFrame(bst.predict(xgmat_test))
                    y_out_test = y_out_test + pd.DataFrame(bst.predict(xgmat_test_sub))
		        #y_out_test_c = y_out_test.join(pd.DataFrame(bst.predict(xgmat_test_sub)))


                if i==folds and j==0:
                    y_pred=y_out_test/folds
                    #y_out_meta=y_pred
                    #print("Resultats:")
                    #print(metrics.roc_auc_score(y_val,y_pred))

                if i==folds and j>0 and j<1:
                    y_pred=y_pred+(y_out_test/folds)
                    #y_out_meta=y_out_meta.join(y_out_test/folds,lsuffix=lsuffix[j])
                    #print("Resultats:")
                    #print(metrics.roc_auc_score(y_val,y_pred/(j+1)))

                if i==folds and j==1:
                    y_pred=y_pred+(y_out_test/folds)
                    y_pred=y_pred/(j+1)
                    #y_out_meta=y_out_meta.join(y_out_test/folds,lsuffix=lsuffix[j])
                    #print("resultats:")
                    pred=y_pred
                    test_sub=pd.DataFrame(np.hstack ([ pd.DataFrame(id),  pd.DataFrame(pd.DataFrame(pred)) ]) )
                    test_sub[0]=test_sub[0].astype(int)
                    test_sub.to_csv("C:/Users/Ju/Desktop/Satander/submit.csv",index=False,header=['ID','TARGET'],dtype=int)
                    return metrics.roc_auc_score(y_val,y_pred)#, y_out_meta
                    #return y_pred

estimate_performance_xgboost(x_train, y_train, x_test, y_test, param, 100, 10,'base')