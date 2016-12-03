import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
import math
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import  AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold
from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_selection import SelectKBest
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.decomposition import TruncatedSVD
from sklearn import neighbors
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn import tree
from sklearn.multiclass import OneVsRestClassifier
from sklearn import linear_model
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc
from sklearn.metrics import zero_one_loss
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import log_loss
from  sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold, chi2, SelectPercentile, f_classif, SelectKBest
from sklearn import neighbors
from sklearn.preprocessing  import PolynomialFeatures
from sklearn.cross_validation import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import RidgeClassifierCV
from sklearn.linear_model import RandomizedLogisticRegression
from sklearn.cross_validation import KFold
import keras
import seaborn
import xgboost as xgb
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import OneHotEncoder

# For .read_csv, always use header=0 when you know row 0 is the header row
df = pd.read_csv('C:/Users/Ju/Desktop/Teslra/train_finale_sans4.csv', header=0, sep =',')
test = pd.read_csv('C:/Users/Ju/Desktop/Teslra/test_finale_sans4.csv', header=0, sep =',')

targets_tr=df['fault_severity']
targets_tr1=df[df['severity_type3']>1]['fault_severity']
targets_tr2=df[df['severity_type3']==1]['fault_severity']
df=df.drop(['location'], axis=1)
test=test.drop(['location'], axis=1)
df=df.drop(['id'], axis=1)
id=test['id']
test=test.drop(['id'], axis=1)

train_test=pd.DataFrame(np.vstack([pd.DataFrame(df),  pd.DataFrame(test)]))
train_test.fillna(0)

from sklearn.pipeline import  FeatureUnion
from sklearn.feature_selection import SelectKBest, chi2, f_regression, f_classif
from sklearn.feature_selection import VarianceThreshold
sel = VarianceThreshold()
train_test=pd.DataFrame(sel.fit_transform(train_test))
train_test.shape

df=train_test[train_test[0]<9]
test=train_test[train_test[0]==9]

train=df

train=train.drop([0], axis=1)
test=test.drop([0], axis=1)
#Traitement de train et test
#frames=[df,test]
#Train_Test=pd.concat(frames)
#from sklearn.feature_selection import VarianceThreshold
#sel = VarianceThreshold()
#Train_Test=sel.fit_transform(Train_Test)
#Train_Test.shape


#df=Train_Test[Train_Test['Fault_severity']<>NaN]
#test=Train_Test[Train_Test['Fault_severity']==NaN]

df=df.drop([1], axis=1)
df=df.fillna(0)
test=test.fillna(0)

df=df.drop([0], axis=1)
df.dtypes
test.dtypes

df.shape
test.shape

train=df

id=test[1]
#id1=test[test['severity_type3']==1]['id']
#id2=test[test['severity_type3']>1]['id']
test=test.drop([0], axis=1)
test=test.drop([], axis=1)

#from sklearn.pipeline import  FeatureUnion
#from sklearn.feature_selection import SelectKBest, chi2, f_regression, f_classif
#from sklearn.feature_selection import VarianceThreshold
#sel = VarianceThreshold()
#train=sel.fit_transform(train)
#train.shape
#test=sel.fit_transform(test)
#test.shape

#TT=sel.fit_transform(TT)
#select = SelectKBest(chi2, k=100)
#selected_features = FeatureUnion(("select", select))
#450 gradient boosting avec location dummies
#442
#sf=SelectKBest(f_classif, k=15000).fit(pd.DataFrame(df),targets_tr)
#train=sf.transform(df)
#(test=test.drop(['test'],axis=1)
#test=sf.transform(test)
#train.shape
train.shape
test.shape

x_train1, x_test1, y_train1, y_test1 = train_test_split(train, targets_tr, test_size=0.30,random_state=42)

x_train2=x_train1
x_test2=x_test1

import sklearn
print('The scikit-learn version is {}.'.format(sklearn.__version__))
from sklearn.feature_selection import SelectFromModel

def modele(clf,tr,y_tr,ts,y_ts):
    #cv = KFold(n=tr.shape[0], n_folds=3)    
    #for train, test in cv:
    #    clf.fit(tr[train], y_tr[train])
    #model=SelectFromModel(ExtraTreesClassifier()).fit(tr,y_tr)
    #tr=model.transform(tr)
    #ts=model.transform(ts)
    clf.fit(tr, y_tr)
    pred=clf.predict_proba(ts)
    pred2=clf.predict(ts)
    y_test_array=np.array(y_ts)
    score = log_loss(y_ts, pred)
   # score2=r2_score(pred2,y_ts)
    #◙print(clf)
    print(score)
    #print(score2)
    del pred
    del y_test_array
    del score
    #pred_test=clf.predict_proba(test.drop(['id'], axis=1))
    #return pred_test
    
#grid search
from sklearn import grid_search

paramRF = {'n_estimators':[100], 'criterion':('gini', 'entropy'), 'max_depth':[3,4,5,10,15,20]}
paramET = {'n_estimators':[100], 'criterion':('gini', 'entropy'), 'max_depth':[3,4,5,10,15,20]}
paramXG = {'n_estimators':[100], 'learning_rate':[0.1], 'reg_alpha':[0],'colsample_bytree':[0.1],'colsample_bylevel':[0.1],'max_depth':[5]}
param_DT = {'criterion':('gini', 'entropy'), 'max_depth':[3,4,5,10,15,20], 'max_features':[None,'auto','sqrt']}
# 'reg_alpha':[0.2,0.3,0.5,0.7], 'reg_lambda':[0,1,5,10]
clfRF = grid_search.GridSearchCV(RandomForestClassifier() , paramRF, cv=2, scoring='log_loss')
clfRF.fit( x_train , y_train )
clfRF.best_estimator_
log_loss(y_test, clfRF.predict_proba(x_test))

clfET = grid_search.GridSearchCV(ExtraTreesClassifier() , paramET, cv=2)
clfET.fit( x_train , y_train )
clfET.best_estimator_
log_loss(y_test, clfET.predict_proba(x_test))

clfXG = grid_search.GridSearchCV(xgb.XGBClassifier() , paramXG, cv=2, scoring='log_loss')
clfXG.fit( x_train , y_train )
clfXG.best_estimator_
log_loss(y_test, clfXG.predict_proba(x_test))

clfDT = grid_search.GridSearchCV(DecisionTreeClassifier() , param_DT, cv=2)
clfDT.fit( x_train , y_train )
clfDT.best_estimator_
log_loss(y_test, clfDT.predict_proba(x_test))

from sklearn.grid_search import GridSearchCV
parameters = {'n_estimators':[750],'base_score':[1],'objective':["multi:softmax"],
'max_depth':[3],'learning_rate':[0.1],'colsample_bytree':[0.2],'colsample_bylevel':[0.2]}
clf = GridSearchCV(xgb.XGBClassifier(),parameters, cv=2,scoring='log_loss')
clf.fit(train,targets_tr) 
clf.best_estimator_
train.shape
test.shape

pred_test=clf.predict_proba(test)

test_sub=pd.DataFrame(np.hstack([pd.DataFrame(id),  pd.DataFrame(pred_test)]))
pred_test.shape

#4test_sub=np.round(test_sub,4)

test_sub[0]=test_sub[0].astype(int)

test_sub[[0,1,2,3]].to_csv("C:/Users/Ju/Desktop/Teslra/submit.csv",index=False,
header=['id','predict_0','predict_1','predict_2'],dtype=int)
#id,predict_0,predict_1,predict_2
#test.dtypes
test_sub[0].astype(int)
#V2
modele(xgb.XGBClassifier(base_score=1, colsample_bylevel=0.1, colsample_bytree=0.1,
       gamma=0, learning_rate=0.01, max_delta_step=0, max_depth=9,
       min_child_weight=1, missing=None, n_estimators=750, nthread=-1,
       objective='multi:softprob', reg_alpha=0, reg_lambda=1,
       scale_pos_weight=0, seed=0, silent=True, subsample=1),
       x_train,y_train,x_test,y_test)
 
#modele(RandomForestClassifier(n_estimators=4000,max_depth=70, max_features='auto', criterion='entropy',random_state=42),
#       x_train,y_train,x_test,y_test)
#avec location max depth=40
modele(RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',
            max_depth=15, max_features='auto', max_leaf_nodes=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=750, n_jobs=-1,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False),
       x_train2,y_train,x_test2,y_test)

#modele(CalibratedClassifierCV(RandomForestClassifier(n_estimators=750,max_depth=35,max_features='auto', criterion='entropy',random_state=42),cv=5,method='isotonic'),
#       x_train2,y_train,x_test2,y_test)
       
#modele(ExtraTreesClassifier(n_estimators=4000,max_depth=70,max_features='auto',criterion='entropy',random_state=42),
#       x_train,y_train,x_test,y_test)
       
modele(LogisticRegression(C=0.01),x_train,y_train,x_test,y_test)

#modele(AdaBoostClassifier(base_estimator=GradientBoostingClassifier(max_features='auto',learning_rate=0.1,random_state=42,
#                                  max_depth=3,n_estimators=500,warm_start=True),learning_rate=0.1),x_train,y_train,x_test,y_test)

#☺modele(CalibratedClassifierCV(LogisticRegression(C=0.1),cv=3,method='isotonic'),x_train,y_train,x_test,y_test)     
  
modele(GradientBoostingClassifier(max_features='auto',learning_rate=0.1,random_state=42,
                                  max_depth=3,n_estimators=500,warm_start=True),x_train,y_train,x_test,y_test)
#XGBOOST
#dtrain = xgb.DMatrix(x_train2)
#dtest = xgb.DMatrix(x_test2)                          

clf=RandomForestClassifier(n_estimators=750).fit(train,targets_tr)      
                            
# Plot feature importance
feature_importance = clf.feature_importances_
# make importances relative to max importance
import matplotlib.pyplot as plt
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.subplot(1, 2, 2)
plt.barh(pos[20040:], feature_importance[sorted_idx][20040:], align='center')
plt.yticks(pos[20040:], x_train.columns.values[sorted_idx][20040:])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.show()                               
  


#Optimisationc
clf1=xgb.XGBClassifier(base_score=1,max_depth=3,n_estimators=750, learning_rate=0.1,colsample_bytree=0.2).fit( x_train , y_train )
log_loss(y_test, clf1.predict_proba(x_test))
pred1=clf1.predict_proba(x_test)

clf2=RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',
            max_depth=10, max_features='auto', max_leaf_nodes=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=750, n_jobs=-1,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False).fit( x_train , y_train )
log_loss(y_test, clf2.predict_proba(x_test))
pred2=clf2.predict_proba(x_test)

clf3=LogisticRegression(C=0.01).fit( x_train , y_train )
log_loss(y_test, clf3.predict_proba(x_test))
pred3=clf3.predict_proba(x_test)

from scipy.optimize import minimize

cons = ({'type': 'eq', 'fun': lambda x:  1 - sum(x)})

fun  = lambda x: log_loss(y_test,x[0]*pred1+x[1]*pred2+x[2]*pred3)
bnds = ((0, 1), (0, 1),(0,1))

res = minimize(fun,(0.5, 0.5,0.5), method='SLSQP',bounds=bnds,constraints=cons)   
res.x
log_loss(y_test,res.x[0]*pred1+res.x[1]*pred2)


from sklearn.grid_search import GridSearchCV
clf1.fit(train,targets_tr) 
clf2.fit(train,targets_tr) 

pred_test=res.x[0]*clf1.predict_proba(test)+res.x[1]*clf2.predict_proba(test)

test_sub=pd.DataFrame(np.hstack([pd.DataFrame(id),  pd.DataFrame(pred_test)]))
pred_test.shape

test_sub[[0]]=test_sub[[0]].astype(int)
test_sub[[0,1,2,3]].to_csv("C:/Users/Ju/Desktop/Teslra/submit.csv",index=False,
header=['id','predict_0','predict_1','predict_2'],dtype=float)     
                       
#modele(CalibratedClassifierCV(GradientBoostingClassifier(max_features='auto',learning_rate=0.1,random_state=42,max_depth=3, n_estimators=500,warm_start=True),cv=5,method='isotonic')
#,x_train2,y_train,x_test2,y_test)

#modele(BaggingClassifier(n_estimators=750),x_train2,y_train,x_test2,y_test)

#modele(DecisionTreeClassifier(),x_train2,y_train,x_test2,y_test)

def modele_ens(clf,tr,y_tr,ts,y_ts):
    clf.fit(tr, y_tr)
    pred=clf.predict(ts)
    y_test_array=np.array(y_ts)
    score = accuracy_score(y_ts, pred)
    print(clf)
    print(score)
    del pred
    del y_test_array
    del score

#Ensemblistes
modele_ens(SVC( C=0.6, kernel='linear'),x_train2,y_train,x_test2,y_test)

modele_ens(PassiveAggressiveClassifier(C=1,random_state=42),x_train2,y_train,x_test2,y_test)

modele_ens(Perceptron(n_iter=3,alpha=1,eta0=0.01,warm_start=True),x_train2,y_train,x_test2,y_test)

#memory error
#modele_ens(RidgeClassifierCV(alphas=0.1),x_train2,y_train,x_test2,y_test)
    
#Non supervisés

#Neural Network

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.layers.advanced_activations import PReLU
from keras.utils import np_utils

columns = pd.DataFrame(x_train).columns
len(columns)

y_train3 = np_utils.to_categorical(y_train)
y_train3.shape

 
model = Sequential()
model.add(Dense(32, input_dim=len(columns), init='glorot_uniform'))
model.add(Activation('sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(input_dim=32, output_dim=32, init='glorot_uniform'))
model.add(Activation('sigmoid'))
model.add(Dropout(0.01))
model.add(Dense(output_dim=3, init='glorot_uniform'))
model.add(Activation('sigmoid'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

model.fit(x_train, y_train3, nb_epoch=150, batch_size=32,verbose=1)

#objective_score = model.evaluate(x_test2, y_train3, batch_size=32)

classes = model.predict_classes(x_test, batch_size=64)
proba = model.predict_proba(x_test, batch_size=64)

t1=pd.DataFrame(y_test)
score = log_loss(t1, proba)
score
accuracy=accuracy_score(y_test, classes)

t2=pd.DataFrame(classes)

pd.DataFrame(y_test2).describe()
pd.DataFrame(classes).describe()
#Mélange 2 modeles



import matplotlib.pyplot as plt
from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_selection import RFECV

# Create the RFE object and compute a cross-validated score.
svc = LogisticRegression(C=0.6)
# The "accuracy" scoring is proportional to the number of correct
# classifications
rfecv = RFECV(estimator=svc,scoring='log_loss')
rfecv.fit(train,targets_tr)

print("Optimal number of features : %d" % rfecv.n_features_)

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()
