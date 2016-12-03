import pandas as pd
import numpy as np
import scipy as sc
from sklearn import metrics
from sklearn.cross_validation import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesRegressor
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import BaggingClassifier
from sklearn.preprocessing import OneHotEncoder, Binarizer
from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier

train=pd.read_csv('C:/Users/Ju/Desktop/Satander/train.csv')
test=pd.read_csv('C:/Users/Ju/Desktop/Satander/test.csv')
#train['poids']=1
#train['poids'].loc[train['TARGET']==1]=2

#Colonnes identiques
id=test['ID']
remove=[]
for col in train.columns:
        if train[col].std() == 0:
            remove.append(col)

#poids=train['poids']
#train=train.drop('poids',axis=1)    
train=train.drop('ID',axis=1)
test=test.drop('ID',axis=1)
target=train['TARGET']
train=train.drop('TARGET',axis=1)

train.drop(remove,axis=1,inplace=True)
test.drop(remove,axis=1,inplace=True)

def feat1(df):
    #Count zero by rows
    df['zeros_by_rows']=(df == 0).astype(int).sum(axis=1)
    df['1_by_rows']=(df == 1).astype(int).sum(axis=1)
    df['sup_1_by_rows']=(df > 1).astype(int).sum(axis=1)
    df['sup_0_by_rows']=(df > 0).astype(int).sum(axis=1)
    df['sup_100k']=(df > 100000).astype(int).sum(axis=1)
    df['sup_1k']=(df > 1000).astype(int).sum(axis=1)
    df['saldo30_42']=(df['saldo_var30'] == df['saldo_var42']).astype(int)

train['var38mc'] = np.isclose(train.var38, 117310.979016).astype(int)
train['logvar38']=np.log(train['var38'].ix[train.var38mc == 0])
train[np.isnan(train)] = 0

test['var38mc'] = np.isclose(test.var38, 117310.979016).astype(int)
test['logvar38']=np.log(test['var38'].ix[test.var38mc == 0])
test[np.isnan(test)] = 0


feat1(train)
feat1(test)

list_drop=['delta_imp_aport_var17_1y3','delta_imp_reemb_var17_1y3','delta_imp_reemb_var33_1y3',
           'delta_imp_trasp_var17_in_1y3','delta_imp_trasp_var17_out_1y3',
          'delta_imp_trasp_var33_in_1y3','delta_imp_trasp_var33_out_1y3',
           'delta_imp_venta_var44_1y3','delta_num_aport_var17_1y3',
          'delta_imp_reemb_var13_1y3']
          
train.drop(list_drop,axis=1,inplace=True)
test.drop(list_drop,axis=1,inplace=True)

train['saldo_var30/saldo_var5'] = train['saldo_var30']-train['saldo_var5']
train['var38/var15'] = train['var38']/train['var15']
train['log(var38)/var15'] = train['logvar38']/train['var15']
train.drop('var38',axis=1,inplace=True)

test['saldo_var30/saldo_var5'] = test['saldo_var30']-test['saldo_var5']
test['var38/var15'] = test['var38']/test['var15']
test['log(var38)/var15'] = test['logvar38']/test['var15']
test.drop('var38',axis=1,inplace=True)

remove2=[]
def remove_feat(train):
    #Algo suppression feature corr=1    
    for i in range(0,train.shape[1]):
        for j in range(i+1,train.shape[1]):
            corr=float(sc.stats.pearsonr(np.array(train[[i]]),np.array(train[[j]]))[0])
            if corr==1 or corr==-1:
                  print(train.columns[i])
                  print(train.columns[j])           
                  print(corr)
                  remove2.append(train.columns[j])
    return remove2
       
remove_feat(train)
train.drop(remove2,axis=1,inplace=True)
test.drop(remove2,axis=1,inplace=True)
#Liste variables de base
feat=train.columns.values.tolist()

from sklearn.feature_selection import SelectPercentile,SelectFdr,f_classif,f_regression,SelectKBest,SelectFpr,SelectFwe
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier

#train['saldo_var30/saldo_var42'] = train['saldo_var30']-train['saldo_var42']
#train['saldo_var42/saldo_var5'] = train['saldo_var5']-train['saldo_var42']
#train['saldo_var30/saldo_var5'] = train['saldo_var30']-train['saldo_var5']
from functools import reduce

def extract(train,target,score_function):
 global a
 Extract1=SelectPercentile(score_function,percentile=50).fit(train,target).get_support(indices=1)
 Extract2=SelectKBest(score_function,k=100).fit(train,target).get_support(indices=1).tolist()
 Extract3=SelectFpr(score_function,alpha=0.05).fit(train,target).get_support(indices=1)
 Extract4=SelectFdr(score_function,alpha=0.05).fit(train,target).get_support(indices=1)
 Extract5=SelectFwe(score_function,alpha=0.05).fit(train,target).get_support(indices=1)
 Extract6=SelectFromModel(RandomForestClassifier()).fit(train,target).get_support(indices=1)
 #a=Extract1 & Extract2 & Extract3 & Extract4 & Extract5 & Extract6
 a=reduce(np.intersect1d, (Extract1,Extract2,Extract3,Extract4,Extract5,Extract6))
 #a=Extract1
 return a
 
#extract(train_new,target,f_classif)
#a.tolist()
#train_new=train_new[train_new.columns[a].tolist()]
#test_new=test_new[test_new.columns[a].tolist()]

#train=train.join(train_new)
#test=test.join(test_new)
#train.shape
#list(train.columns[b])

def put_corr_feat(data,test1,target):
	length=data.shape[1]
	for i in range(0,2):
         print(i)
         for j in range(i+1,length):
             corr1=float(sc.stats.pearsonr(np.array(data[[i]]),np.array(pd.DataFrame(target)))[0])
             corr2=float(sc.stats.pearsonr(np.array(data[[j]]),np.array(pd.DataFrame(target)))[0])
             corr_sum=float(sc.stats.pearsonr(np.array(data[[i]])+np.array(data[[j]]),np.array(pd.DataFrame(target)))[0])
             corr_sou=float(sc.stats.pearsonr(np.array(data[[i]])-np.array(data[[j]]),np.array(pd.DataFrame(target)))[0])
             corr_mul=float(sc.stats.pearsonr(np.array(data[[i]])*np.array(data[[j]]),np.array(pd.DataFrame(target)))[0])
		
             if abs(corr_sum)>abs(corr1) and abs(corr_sum)>abs(corr2):
                 data=data.join(pd.DataFrame(np.array(data[[i]])+np.array(data[[j]]),columns=data[[j]].columns.values),rsuffix=str(i+0.1))
                 test1=test1.join(pd.DataFrame(np.array(test[[i]])+np.array(test[[j]]),columns=test[[j]].columns.values),rsuffix=str(i+0.1))
             if abs(corr_sou)>abs(corr1) and abs(corr_sou)>abs(corr2):
			data=data.join(pd.DataFrame(np.array(data[[i]])-np.array(data[[j]]),columns=data[[j]].columns.values),rsuffix=str(i+0.2))
			test1=test1.join(pd.DataFrame(np.array(test[[i]])-np.array(test[[j]]),columns=test[[j]].columns.values),rsuffix=str(i+0.2))
             if abs(corr_mul)>abs(corr1) and abs(corr_mul)>abs(corr2):
			data=data.join(pd.DataFrame(np.array(data[[i]])*np.array(data[[j]]),columns=data[[j]].columns.values),rsuffix=str(i+0.3))
			test1=test1.join(pd.DataFrame(np.array(test[[i]])*np.array(test[[j]]),columns=test[[j]].columns.values),rsuffix=str(i+0.3))
        
        global train_new
        global test_new
        train_new=data
        test_new=test1
        return train_new
 
put_corr_feat(train,test,target)
train_new.drop(feat,axis=1,inplace=True)
test_new.drop(feat,axis=1,inplace=True)

train=train.join(train_new)
test=test.join(test_new)

def del_corr_low(data,test,target,tol):
    len=data.shape[1]
    for i in range(0,data.shape[1]):
        corr=float(sc.stats.pearsonr(np.array(data[[i]]),np.array(pd.DataFrame(target)))[0])
        if abs(corr)<tol:
            data.drop(data.columns[[i]],axis=1,inplace=True)
            test.drop(test.columns[[i]],axis=1,inplace=True)
    print("Nombre de variables supprimées:")
    return data.shape[1]-len



#Imputation des valeurs manquantes
def imput(mod,var,nan,train,test):
    
    train_test=pd.concat([train,test])
    
    target=train_test[train_test[var] != nan][var]
    
    target_train=train[train[var] != nan][var]
    target_test=test[test[var] != nan][var]
    
    train_pred=train[train[var] == nan]
    test_pred=test[test[var] == nan]
    train_pred=train_pred.drop(var,axis=1)
    test_pred=test_pred.drop(var,axis=1)
    
    train_test_imp=train_test[train_test[var] != nan].drop(var,axis=1)
    train_imp=train[train[var] != nan].drop(var,axis=1)
    test_imp=test[test[var] != nan].drop(var,axis=1)
    
    clf=mod.fit(train_test_imp,target)
    
    train_new=clf.predict(train_pred)
    test_new=clf.predict(test_pred)
    
    test_imp=test_imp.join(target_test)
    train_imp=train_imp.join(target_train)
    
    global train_out
    global test_out
    train_pred=train_pred.join(pd.DataFrame(train_new,columns=['var3']))
    train_out=pd.concat([train_imp,train_pred])
    test_pred=test_pred.join(pd.DataFrame(test_new,columns=['var3']))
    test_out=pd.concat([test_imp,test_pred])
    
    return pd.DataFrame(train_new).describe(), pd.DataFrame(test_new).describe()

#One Hot Encoding sur integer pour Metafeatures
def OHE(df, test, list_var, mod, target):
	#One Hote Encoding
	enc = OneHotEncoder()
	train_OHE=enc.fit(df[list_var])
	test_OHE=enc.transform(test[list_var])
	
	#MetaFeatures
	model=BaggingClassifier(mod,n_estimators=20).fit(train_OHE,target)
	meta_train=mod.predict_proba(train_OHE)[1]
	meta_test=mod.predict_proba(test_OHE)[1]
	df=np.concatenate(df,meta_train)
	test=np.concatenate(test,meta_test)

	df=df.drop(list_var,axis=1)
	test=test.drop(list_var,axis=1)

	return df ,test


#Variables Continues
list_cont=['zeros_by_rows','_1_by_rows','9_by_rows','99_by_rows','_9_by_rows','1_by_rows','sup_1_by_rows','inf_0_by_rows','sup_0_by_rows',
'var15','imp_ent_var16_ult1','imp_op_var39_comer_ult1','imp_op_var39_comer_ult3',
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

#train_cont=pd.DataFrame(StandardScaler().fit_transform(train[list_cont]))
#print(train_cont.shape)
#train.drop(list_cont,axis=1,inplace=True)
#print(train.shape)
#train=train.join(train_cont,rsuffix='z')
#print(train.shape)

#train['saldo_var30/saldo_var42'] = train['saldo_var30']-train['saldo_var42']
#train['saldo_var42/saldo_var5'] = train['saldo_var5']-train['saldo_var42']
#train['saldo_var30/saldo_var5'] = train['saldo_var30']-train['saldo_var5']

train = train.replace(-999999,9)
test = test.replace(-999999,9)
x_train, x_test, y_train, y_test = train_test_split(train,target,test_size=0.30, 
                                                    random_state=42,stratify=target)


gbm2=BaggingClassifier(xgb.XGBClassifier(n_estimators=560,max_depth=5,learning_rate=0.01,subsample=0.6815,
                                         colsample_bytree=1),n_estimators=5,verbose=1,random_state=42).fit(x_train,y_train)

gbm2=xgb.XGBClassifier(n_estimators=1560200,max_depth=5,learning_rate=0.02,subsample=1,colsample_bytree=1).fit(x_train,y_train)
y_pred0=pd.DataFrame(gbm2.predict_proba(x_test))
print(metrics.roc_auc_score(y_test,y_pred0[1]))  

test_sub=pd.DataFrame(np.hstack ([ pd.DataFrame(id),  pd.DataFrame(gbm2.predict_proba(test)) ]) )
test_sub.drop('1',axis=1,inplace=True)
del test_sub[1]
#test_sub[2]=test_sub[2].astype(int)
test_sub.to_csv("C:/Users/Ju/Desktop/Satander/submit1.csv",index=False,header=['ID','TARGET'],dtype=int)

                                                       
# setup parameters for xgboost
param = {
'max_depth':5,
'eta':0.02,
'objective':'binary:logistic',
'booster':'gbtree',
'subsample':0.7,#0.5#0.55#0.7
'base_score':0.5,
'colsample_bytree':0.7,#0.65
'eval_metric':'auc'
}

xgmat = xgb.DMatrix(train, label=target )            
# Construct matrix for test set
xgmat_test = xgb.DMatrix(test)
bst = xgb.train(param, xgmat, 560)
pred = pd.DataFrame(bst.predict(xgmat_test))

test_sub=pd.DataFrame(np.hstack ([ pd.DataFrame(id),  pd.DataFrame(pd.DataFrame(pred)) ]) )
test_sub[0]=test_sub[0].astype(int)
test_sub.to_csv("C:/Users/Ju/Desktop/Satander/submit.csv",index=False,header=['ID','TARGET'],dtype=int)
                    
def estimate_performance_xgboost(training_file, target_train,test_val, test_file, y_val, param, num_round, folds,mode):
    # Load training data
    X=training_file.copy()   
    X['a']=range(0,len(target_train))
    X=X.set_index(['a'],drop=True)
    
    labels=pd.DataFrame(target_train)
    labels['a']=range(0,len(target_train))
    labels=labels.set_index(['a'],drop=True)
   
    xgmat_trainingfile = xgb.DMatrix(training_file, label=target_train )
    xgmat_test_sub = xgb.DMatrix(test_file)
    xgmat_val = xgb.DMatrix(test_val)      
    #print(X.shape)
    lsuffix=['0','1','2','3','4']
    if mode=='base':
        kf = StratifiedKFold(target_train, n_folds=folds)
        i=0
        #training_file=training_file.drop('a',axis=1)
        for train_indices, test_indices in kf:
            i=i+1
            X_train, X_test = X.ix[train_indices], X.ix[test_indices]
            y_train, y_test = labels.ix[train_indices], labels.ix[test_indices]
            
            # construct xgboost.DMatrix
            xgmat = xgb.DMatrix(X_train, label=y_train )            
            # Construct matrix for test set
            xgmat_test = xgb.DMatrix(X_test, label=y_test)
            watchlist=[(xgmat,'train'),(xgmat_test,'eval')]
            bst = xgb.train(param, xgmat, num_round,watchlist,verbose_eval=False,early_stopping_rounds=60)
            #print(bst.best_iteration)
            global a
            if i==1:
                #Initialisation prediction
                #y_out = pd.DataFrame(bst.predict(xgmat_test))
                y_out_test=pd.DataFrame(bst.predict(xgmat_test_sub))
                y_out_val=pd.DataFrame(bst.predict(xgmat_val))
                print('Initialisation')
                print(metrics.roc_auc_score(y_val,y_out_val))
      
            if i>1:
                #y_out = y_out + pd.DataFrame(bst.predict(xgmat_test))
                #y_out_test = y_out_test + 0.5*pd.DataFrame(bst.predict(xgmat_test_sub))
                for j in range(0,6):
                        a=metrics.roc_auc_score(y_val,y_out_val + (j)*pd.DataFrame(bst.predict(xgmat_val)))
                        b=metrics.roc_auc_score(y_val,y_out_val + (j+1)*pd.DataFrame(bst.predict(xgmat_val)))
                        if float(a)<float(b):
                                continue
                        if float(a)>float(b) and j==0:
                                c=metrics.roc_auc_score(y_val,y_out_val)                              
                                break
                        if float(a)>float(b) and j>0:
                                c=metrics.roc_auc_score(y_val,y_out_val + (j)*pd.DataFrame(bst.predict(xgmat_val)))                               
                                break
                print(j)
                y_out_val = y_out_val + (j)*pd.DataFrame(bst.predict(xgmat_val))
                y_out_test = y_out_test + (j)*pd.DataFrame(bst.predict(xgmat_test_sub))
                #print('seul')
                #print(metrics.roc_auc_score(y_val,pd.DataFrame(bst.predict(xgmat_test_sub))))
                print('cumule')
                print(metrics.roc_auc_score(y_val,y_out_val))               
                #print(metrics.roc_auc_score(y_test,y_out_test + pd.DataFrame(bst.predict(xgmat_test))))
            
            if i==folds:
                #bst = xgb.train(param, xgmat_trainingfile, num_round,watchlist,verbose_eval=False,early_stopping_rounds=60)
                y_out_val=   y_out_val #+ 1*pd.DataFrame(bst.predict(xgmat_val))              
                y_out_test = y_out_test #+ 1*pd.DataFrame(bst.predict(xgmat_test_sub)) #3
                #print(metrics.roc_auc_score(y_val,pd.DataFrame(bst.predict(xgmat_test_sub))))
                y_pred=y_out_val
                print("resultats:")
                global pred
                pred=y_pred
                global test_sub 
                pred_sub=y_out_test
                test_sub=pd.DataFrame(np.hstack ([ pd.DataFrame(id),  pd.DataFrame(pd.DataFrame(pred_sub)) ]) )
                test_sub[0]=test_sub[0].astype(int)
                #test_sub.to_csv("C:/Users/Ju/Desktop/Satander/submit.csv",index=False,header=['ID','TARGET'],dtype=int)
                return pred, metrics.roc_auc_score(y_val,y_pred)#, y_out_meta
                
    if mode=='RS':
        for j in range(0,20):
            #Cross validation
            kf = StratifiedKFold(target_train, n_folds=folds,shuffle=True,random_state=j)
            i=0
            print(j)        
            #training_file=training_file.drop('a',axis=1)
            for train_indices, test_indices in kf:
                i=i+1
                X_train, X_test = X.ix[train_indices], X.ix[test_indices]
                y_train, y_test = labels.ix[train_indices], labels.ix[test_indices]
                #w_train, w_test = weights[train_indices], weights[test_indices]
                
                # construct xgboost.DMatrix
                xgmat = xgb.DMatrix(X_train, label=y_train )            
                # Construct matrix for test set
                xgmat_test = xgb.DMatrix(X_test, label=y_test)
                watchlist=[(xgmat,'train'),(xgmat_test,'eval')]
                bst = xgb.train(param, xgmat, num_round,watchlist,verbose_eval=False,early_stopping_rounds=1000000)
                print(bst.best_iteration)
            
                if i==1:
                    #Initialisation prediction
                    y_out = pd.DataFrame(bst.predict(xgmat_test))
                    y_out_test=pd.DataFrame(bst.predict(xgmat_test_sub))
            
                if i>1:
                    y_out = y_out + pd.DataFrame(bst.predict(xgmat_test))
                    y_out_test = y_out_test + pd.DataFrame(bst.predict(xgmat_test_sub))
                    #metric
                    #print(metrics.roc_auc_score(y_test,pd.DataFrame(bst.predict(xgmat_test))))

                if i==folds and j==0:
                    y_pred=y_out_test/folds
                    #y_out_meta=y_pred
                    #print("Resultats:")
                    #print(metrics.roc_auc_score(y_val,y_pred))

                if i==folds and j>0 and j<19:
                    y_pred=y_pred+(y_out_test/folds)
                    #y_out_meta=y_out_meta.join(y_out_test/folds,lsuffix=lsuffix[j])
                    #print("Resultats:")
                    #print(metrics.roc_auc_score(y_val,y_pred/(j+1)))

                if i==folds and j==19:
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

#Optimisation avec y_pred par rapport à y_test
def fopt_pred(pars, data):
    return np.dot(data, pars)

def fopt(pars):
    fpr, tpr, thresholds = metrics.roc_curve(y_train, fopt_pred(pars, B_train))
    return -metrics.auc(fpr, tpr)

x0 = np.ones((n_models, 1)) / n_models
xopt = fmin(fopt, x0)
preds = fopt_pred(xopt, B_test)
#minimize with method 'Nelder-Mead'

      
#B est: eta=0.01 num_round=1k score test:0.8472...
#Best 2: eta=0.02 num_round=560 score test:8476213833312547
#Best3: eta=0.02 num_round=560 score test:
#best3+maxdepth=5 score test=0.8483854511684934 ajout de variable plus correlés avec i=0

#Pour soumission
#estimate_performance_xgboost(x_train, y_train, x_test, y_test, param, 570, 4,'base')
#best avec 50
estimate_performance_xgboost(train, target, x_test, test, y_test, param, 560, 50,'base')

test_final=test.join(test_sub)

test_final.ix[test_final.var15 < 23, 1] = 0
test_final.ix[test_final.saldo_medio_var5_hace2 > 160000, 1] = 0
test_final.ix[test_final.saldo_var33 > 0, 1] = 0
test_final.ix[test_final.logvar38 > np.log(3988596), 1] = 0
test_final.ix[test_final.num_var33+test_final.saldo_medio_var33_ult3+test_final.saldo_medio_var44_hace2+test_final.saldo_medio_var44_hace3+
test_final.saldo_medio_var33_ult1+test_final.saldo_medio_var44_ult1>0, 1] = 0
test_final.ix[test_final.var21 > 7500, 1] = 0
test_final.ix[test_final.num_var30 > 9, 1] = 0
test_final.ix[test_final.num_var13_0 > 6, 1] = 0

test_final.ix[test_final.imp_ent_var16_ult1 > 51003, 1] = 0
test_final.ix[test_final.imp_op_var39_comer_ult3 > 13184, 1] = 0
test_final.ix[test_final.saldo_medio_var5_ult3 > 108251, 1] = 0
test_final.ix[test_final.num_var37_0 > 45, 1] = 0
test_final.ix[test_final.saldo_var5 > 137615, 1] = 0

test_final.ix[test_final.saldo_var8 > 60099, 1] = 0
test_final.ix[test_final.saldo_var14 > 19053.78, 1] = 0
test_final.ix[test_final.saldo_var17 > 288188.97, 1] = 0
test_final.ix[test_final.saldo_var26 > 10381.29, 1] = 0
test_final.ix[test_final.num_var13_largo_0 > 3, 1] = 0

test_final.ix[test_final.imp_op_var40_comer_ult1 > 3639.87, 1] = 0
test_final.ix[test_final.var15 + test_final.num_var45_hace3 + test_final.num_var45_hace3 +  test_final.var36 <= 24, 1] = 0

test_final.ix[test_final.saldo_medio_var13_largo_ult1 > 0, 1] = 0
test_final.ix[test_final.num_meses_var13_largo_ult3 > 0, 1] = 0
#test_final.ix[test_final.num_var20_0 > 0, 1] = 0
test_final.ix[test_final.saldo_var13_largo > 150000, 1] = 0

test_final.ix[test_final.num_var17_0 > 21, 1] = 0
test_final.ix[test_final.num_var24_0 > 3, 1] = 0
test_final.ix[test_final.num_var26_0 > 12, 1] = 0



test_final[1]=test_final[1]/test_final[1].max()   
test_final[[0,1]].to_csv("C:/Users/Ju/Desktop/Satander/submit.csv",index=False,header=['ID','TARGET'],dtype=int)
   

#estimate_performance_xgboost(train, target, test, y_test, param, 560, 100,'base')

x_test2=x_test.reset_index(drop=True)
y_test2=y_test.reset_index(drop=True)

x_test3=x_test2.join(pred)
#print(x_test3.shape[1])

x_test3.ix[x_test3.var15 < 23, 0] = 0
x_test3.ix[x_test3.saldo_medio_var5_hace2 > 160000, 1] = 0
x_test3.ix[x_test3.saldo_var33 > 0, 0] = 0
x_test3.ix[x_test3.logvar38 > np.log(3988596), 0] = 0
x_test3.ix[x_test3.num_var33+x_test3.saldo_medio_var33_ult3+x_test3.saldo_medio_var44_hace2+x_test3.saldo_medio_var44_hace3+
x_test3.saldo_medio_var33_ult1+x_test3.saldo_medio_var44_ult1>0, 0] = 0
x_test3.ix[x_test3.var21 > 7500, 0] = 0
x_test3.ix[x_test3.num_var30 > 9, 0] = 0
x_test3.ix[x_test3.num_var13_0 > 6, 0] = 0

y_pred000=x_test3[0]
print(metrics.roc_auc_score(y_test,y_pred000))
print(metrics.roc_auc_score(y_test,pred))


#optimisation
import pandas as pd
import numpy as np


def intersect(a, b):
    return list(set(a) & set(b))


def get_features(train, test):
    trainval = list(train.columns.values)
    testval = list(test.columns.values)
    output = intersect(trainval, testval)
    #output.remove('ID')
    return output


def prepare_dataset(tr,ts):
    train1 = tr
    test2 = ts
    features = tr.columns.values
    
    norm_f = []
    for f in features:
        u = tr[f].unique()
        if len(u) != 1:
            norm_f.append(f)


    remove = []
    for i in range(len(norm_f)):
        v1 = tr[norm_f[i]].values
        for j in range(i+1, len(norm_f)):
            v2 = tr[norm_f[j]].values
            if np.array_equal(v1, v2):
                remove.append(norm_f[j])
    
    for r in remove:
        norm_f.remove(r)

    train1 = tr[norm_f]
    #norm_f.remove('TARGET')
    ts = ts[norm_f]
    features = get_features(tr, ts)
    return tr, ts, features


def find_min_max_features(df, f):
    return df[f].min(), df[f].max()


def analayze_data(train, test):
    print('Length of train: ', len(train.index))
    train_zero = train[train['TARGET'] == 0]
    print('Length of train [TARGET = 0]: ', len(train_zero.index))
    train_one = train[train['TARGET'] == 1]
    print('Length of train [TARGET = 1]: ', len(train_one.index))
    one_range = dict()
    for f in train.columns:
        mn0, mx0 = find_min_max_features(train_zero, f)
        mn1, mx1 = find_min_max_features(train_one, f)
        mnt = 'N/A'
        mxt = 'N/A'
        if mn1>mn0 and f!='TARGET' and f!='ID':
            x_test4.ix[x_test4[f]< mn1, 0] = 0
            print(metrics.roc_auc_score(y_test,x_test3[0]))
            print(metrics.roc_auc_score(y_test,x_test4[0]))
            if float(metrics.roc_auc_score(y_test,x_test3[0]))<float(metrics.roc_auc_score(y_test,x_test4[0])):
                test_final.ix[test_final[f]< mn1, 1] = 0            
                x_test3.ix[x_test3[f]< mn1, 0] = 0                
                print(metrics.roc_auc_score(y_test,x_test3[0]))
        if mx1<mx0 and f!='TARGET' and f!='ID':
            x_test4.ix[x_test4[f]> mx1, 0] = 0
            if metrics.roc_auc_score(y_test,x_test3[0])<metrics.roc_auc_score(y_test,x_test4[0]):
                test_final.ix[test_final[f]> mx1, 1] = 0
                x_test3.ix[x_test3[f]> mx1, 0] = 0
                print(metrics.roc_auc_score(y_test,x_test3[0]))
        if f in test.columns:
            mnt, mxt = find_min_max_features(test, f)
        one_range[f] = (mn1, mx1)
        #if mn0 != mn1 or mn1 != mnt or mx0 != mx1 or mx1 != mxt:
            #print("\nFeature {}".format(f))
            #print("Range target=0  ({} - {})".format(mn0, mx0))
            #print("Range target=1  ({} - {})".format(mn1, mx1))
            #print("Range in test   ({} - {})".format(mnt, mxt))

test_final=test.join(test_sub)
x_test3=x_test2.join(pred)
x_test4=x_test3
train1, test1, features = prepare_dataset(train,test)
train1=train1.join(target)
analayze_data(train1, test1)


#test_final=test.join(test_sub)
test_final[1]=test_final[1]/test_final[1].max()   
test_final[[0,1]].to_csv("C:/Users/Ju/Desktop/Satander/submit.csv",index=False,header=['ID','TARGET'],dtype=int)
   