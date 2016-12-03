import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesRegressor
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import BaggingClassifier
import scipy as sc

train=pd.read_csv('C:/Users/Ju/Desktop/Satander/train.csv')
#test=pd.read_csv('../input/test.csv')
print(train.shape)

remove=[]
for col in train.columns:
        if train[col].std() == 0:
            remove.append(col)

train=train.drop('ID',axis=1)
target=train['TARGET']
train=train.drop('TARGET',axis=1)


train.drop(remove,axis=1,inplace=True)

remove2=[]
def remove_feat(train):
    #Algo suppression feature corr=1    
    for i in range(0,train.shape[1]):
        for j in range(i+1,train.shape[1]):
            corr=float(sc.stats.pearsonr(np.array(train[[i]]),np.array(train[[j]]))[0])
            if corr==1 or corr==-1:
                  remove2.append(train.columns[j])
    return remove2
       
remove_feat(train)
train.drop(remove2,axis=1,inplace=True)
#test.drop(remove2,axis=1,inplace=True)
print(train.shape)
def feat1(df):
    #Count zero by rows
    df['zeros_by_rows']=(df == 0).astype(int).sum(axis=1)
    #df['_1_by_rows']=(df == -1).astype(int).sum(axis=1)
    #df['9_by_rows']=(df == 9999999999).astype(int).sum(axis=1)
    #df['99_by_rows']=(df == 99).astype(int).sum(axis=1)
    #df['_9_by_rows']=(df == -999999).astype(int).sum(axis=1)
    #df['1_by_rows']=(df == 1).astype(int).sum(axis=1)
    #df['sup_1_by_rows']=(df > 1).astype(int).sum(axis=1)
    #df['inf_0_by_rows']=(df < 0).astype(int).sum(axis=1)
    #df['sup_0_by_rows']=(df > 0).astype(int).sum(axis=1)

feat1(train)

#'_1_by_rows','9_by_rows','99_by_rows','_9_by_rows','1_by_rows','sup_1_by_rows','inf_0_by_rows','sup_0_by_rows',
#Variables Continues
list_cont=['zeros_by_rows','var15','imp_ent_var16_ult1','imp_op_var39_comer_ult1','imp_op_var39_comer_ult3',
'imp_op_var40_comer_ult1','imp_op_var40_comer_ult3','imp_op_var40_efect_ult1','imp_op_var40_efect_ult3',
'imp_op_var40_ult1','imp_op_var41_comer_ult1','imp_op_var41_comer_ult3','imp_op_var41_efect_ult1',
'imp_op_var41_efect_ult3','imp_op_var41_ult1','imp_op_var39_efect_ult1','imp_op_var39_efect_ult3',
'imp_op_var39_ult1','imp_sal_var16_ult1','saldo_var1','saldo_var5','saldo_var6','saldo_var8','saldo_var12',
'saldo_var13_corto','saldo_var13_largo','saldo_var13_medio','saldo_var13','saldo_var14','saldo_var17','saldo_var18',
'saldo_var20','saldo_var24','saldo_var26','saldo_var25','saldo_var30','saldo_var31','saldo_var32','saldo_var33',
'saldo_var34','saldo_var37','saldo_var40','saldo_var42','saldo_var44','delta_imp_amort_var18_1y3',
'delta_imp_amort_var34_1y3','delta_imp_aport_var13_1y3','delta_imp_aport_var17_1y3','delta_imp_aport_var33_1y3',
'delta_imp_compra_var44_1y3','delta_imp_reemb_var13_1y3','delta_imp_reemb_var17_1y3','delta_imp_reemb_var33_1y3',
'delta_imp_trasp_var17_in_1y3','delta_imp_trasp_var17_out_1y3','delta_imp_trasp_var33_in_1y3',
'delta_imp_trasp_var33_out_1y3','delta_imp_venta_var44_1y3','delta_num_aport_var17_1y3','delta_num_compra_var44_1y3',
'imp_amort_var18_ult1','imp_amort_var34_ult1','imp_aport_var13_hace3','imp_aport_var13_ult1','imp_aport_var17_hace3',
'imp_aport_var17_ult1','imp_aport_var33_hace3','imp_aport_var33_ult1','imp_var7_emit_ult1','imp_var7_recib_ult1',
'imp_compra_var44_hace3','imp_compra_var44_ult1','imp_reemb_var13_ult1','imp_reemb_var17_hace3','imp_reemb_var17_ult1',
'imp_reemb_var33_ult1','imp_var43_emit_ult1','imp_trans_var37_ult1','imp_trasp_var17_in_hace3','imp_trasp_var17_in_ult1',
'imp_trasp_var17_out_ult1','imp_trasp_var33_in_hace3','imp_trasp_var33_in_ult1','imp_trasp_var33_out_ult1',
'imp_venta_var44_hace3','imp_venta_var44_ult1','saldo_medio_var5_hace2','saldo_medio_var5_hace3','saldo_medio_var5_ult1',
'saldo_medio_var5_ult3','saldo_medio_var8_hace2','saldo_medio_var8_hace3','saldo_medio_var8_ult1','saldo_medio_var8_ult3',
'saldo_medio_var12_hace2','saldo_medio_var12_hace3','saldo_medio_var12_ult1','saldo_medio_var12_ult3',
'saldo_medio_var13_corto_hace2','saldo_medio_var13_corto_hace3','saldo_medio_var13_corto_ult1',
'saldo_medio_var13_corto_ult3','saldo_medio_var13_largo_hace2','saldo_medio_var13_largo_hace3',
'saldo_medio_var13_largo_ult1','saldo_medio_var13_largo_ult3','saldo_medio_var13_medio_hace2',
'saldo_medio_var13_medio_ult3','saldo_medio_var17_hace2','saldo_medio_var17_hace3','saldo_medio_var17_ult1',
'saldo_medio_var17_ult3','saldo_medio_var29_hace2','saldo_medio_var29_hace3','saldo_medio_var29_ult1',
'saldo_medio_var29_ult3','saldo_medio_var33_hace2','saldo_medio_var33_hace3','saldo_medio_var33_ult1',
'saldo_medio_var33_ult3','saldo_medio_var44_hace2','saldo_medio_var44_hace3','saldo_medio_var44_ult1',
'saldo_medio_var44_ult3','var38']

train_cont=pd.DataFrame(StandardScaler().fit_transform(train[list_cont]))
print(train_cont.shape)
train.drop(list_cont,axis=1,inplace=True)
print(train.shape)
train=train.join(train_cont,rsuffix='z')
print(train.shape)


#from sklearn.preprocessing import OneHotEncoder

#enc=OneHotEncoder()
#enc.fit(train[list_ord].abs())
#train_ord=enc.transform(train[list_ord].abs())
#print(train_ord.shape)
#train.drop(list_ord,axis=1,inplace=True)
#print(train.shape)
#train=train.join(pd.DataFrame(train_ord.toarray()),rsuffix='zz')
#print(train.shape)
x_train, x_test, y_train, y_test = train_test_split(train,target,test_size=0.30, random_state=42,stratify=target)

gbm2=BaggingClassifier(xgb.XGBClassifier(n_estimators=100,max_depth=5,learning_rate=0.1,subsample=1,
                                         colsample_bytree=1),n_estimators=5,verbose=1,random_state=42).fit(x_train,y_train)

y_pred0=pd.DataFrame(gbm2.predict_proba(x_test))
print(metrics.roc_auc_score(y_test,y_pred0[1]))
#Optimisation gradient boosting
#Choose all predictors except target & IDcols
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier

#param_test1 = {'n_estimators':[20,30,40,50,60,80,100]}
#param_test2 = {'max_depth':[5,7,9,11,12,15], 'min_samples_split':[200,400,600,800,1000]}
#param_test3 = {'min_samples_leaf':[30,40,50,60,70]}
#param_test4 = {'max_features':[17,18,19,20]}
#param_test5 = {'subsample':[0.6,0.7,0.75,0.8,0.85,0.9]}
#param_test1 = {'n_estimators':[320,650,1200],'learning_rate':[0.025,0.01,0.005]}

#gbm final
#gsearch1 = GridSearchCV(estimator = GradientBoostingClassifier(n_estimators=320,max_depth=7,min_samples_split=400,learning_rate=0.025,min_samples_leaf=50,
#                                                               max_features=17,subsample=0.8,random_state=10), 
#param_grid = param_test1, scoring='roc_auc',n_jobs=-1,iid=False, cv=2)

#gsearch1.fit(train,target)
#gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_
from sklearn.ensemble import GradientBoostingClassifier
#gbm
clf_gbm=GradientBoostingClassifier(n_estimators=320,max_depth=7,min_samples_split=400,learning_rate=0.025,min_samples_leaf=50,
                                                               max_features=17,subsample=0.8,random_state=10).fit(x_train,y_train)


y_pred1=pd.DataFrame(clf_gbm.predict_proba(x_test))
print(metrics.roc_auc_score(y_test,y_pred1[1]))
from sklearn.linear_model import LogisticRegression, LinearRegression, Perceptron, ElasticNetCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier

#log_bag=BaggingClassifier(LinearRegression(),n_estimators=5,verbose=1,random_state=42).fit(x_train,y_train)
#log_bag=AdaBoostClassifier(LinearRegression(),n_estimators=50, learning_rate=0.1,random_state=42).fit(x_train,y_train)
#log_bag=LinearRegression().fit(x_train,y_train)
log_bag=ElasticNetCV(normalize=True,cv=5).fit(x_train,y_train)
#clf=LinearRegression()
#clf.fit(x_train,y_train)

#y_pred2=pd.DataFrame(log_bag.predict_proba(x_test))
y_pred2=pd.DataFrame(log_bag.predict(x_test))
#print(metrics.roc_auc_score(y_test,y_pred2[1]))
print(metrics.roc_auc_score(y_test,y_pred2))
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

clf=AdaBoostClassifier(RandomForestClassifier(n_estimators=200,max_depth=2,random_state=42),
                       n_estimators=100, learning_rate=0.5,random_state=42)
clf.fit(x_train,y_train)
y_pred3=pd.DataFrame(clf.predict_proba(x_test))
print(metrics.roc_auc_score(y_test,y_pred3[1]))
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

clf2=AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                        n_estimators=100, learning_rate=0.5,random_state=42)
clf2.fit(x_train,y_train)
y_pred4=pd.DataFrame(clf2.predict_proba(x_test))
print(metrics.roc_auc_score(y_test,y_pred4[1]))

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier

clfex=AdaBoostClassifier(ExtraTreesClassifier(n_estimators=50,max_depth=3,random_state=42),
                         n_estimators=200, learning_rate=0.25,random_state=42)
clfex.fit(x_train,y_train)
y_pred5=pd.DataFrame(clfex.predict_proba(x_test))
print(metrics.roc_auc_score(y_test,y_pred5[1]))
#print(metrics.roc_auc_score(y_test,(y_pred0[1]+y_pred1[1]+y_pred3[1]+y_pred4[1])/5))

#y_pred00=pd.DataFrame(gbm2.predict_proba(x_train))
#y_pred11=pd.DataFrame(clf_gbm.predict_proba(x_train))
#y_pred33=pd.DataFrame(clf.predict_proba(x_train))
#y_pred44=pd.DataFrame(clf2.predict_proba(x_train))
#y_pred55=pd.DataFrame(clfex.predict_proba(x_train))

#pred_meta=pd.DataFrame(y_pred00[1]).join(pd.DataFrame(y_pred11[1]),rsuffix='l').join(pd.DataFrame(y_pred33[1]),rsuffix='r').join(pd.DataFrame(y_pred44[1]),rsuffix='rr').join(pd.DataFrame(y_pred55[1]),rsuffix='rrr')
#pred_x_test=pd.DataFrame(y_pred0[1]).join(pd.DataFrame(y_pred1[1]),rsuffix='l').join(pd.DataFrame(y_pred3[1]),rsuffix='r').join(pd.DataFrame(y_pred4[1]),rsuffix='rr').join(pd.DataFrame(y_pred5[1]),rsuffix='rrr')

#clfmeta=AdaBoostClassifier(RandomForestClassifier(n_estimators=100,max_depth=1,random_state=42),
#                           n_estimators=100, learning_rate=0.5,random_state=42)
#clfmeta.fit(pred_meta,y_train)

#gbm3=BaggingClassifier(xgb.XGBClassifier(n_estimators=100,max_depth=1,learning_rate=0.1,subsample=1,
#                                         colsample_bytree=1),n_estimators=5,verbose=1,random_state=42).fit(pred_meta,y_train)
#gbm3.fit(pred_meta,y_train)

y_predmeta=pd.DataFrame(clfmeta.predict_proba(pred_x_test))
#y_predmeta2=pd.DataFrame(gbm3.predict_proba(pred_x_test))

print(metrics.roc_auc_score(y_test,y_predmeta[1]))
#print(metrics.roc_auc_score(y_test,y_predmeta2[1]))
#print(metrics.roc_auc_score(y_test,(y_predmeta2[1]+y_predmeta[1])/2))