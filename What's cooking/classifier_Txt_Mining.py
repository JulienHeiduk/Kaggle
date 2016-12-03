from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import nltk
import re
from nltk.stem import WordNetLemmatizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
import sklearn.metrics
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from sklearn import grid_search
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from nltk.corpus.reader.wordnet import NOUN
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import codecs
import sys
from sklearn.feature_selection import VarianceThreshold, chi2, SelectPercentile, f_classif, SelectKBest
from collections import Counter
from sklearn import linear_model
from sklearn.pipeline import Pipeline, FeatureUnion
from  sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.cross_validation import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import time  
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import StratifiedKFold
from sklearn.preprocessing import scale
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import VotingClassifier
import keras
import xgboost

encoder=LabelEncoder()
#nltk.download()
traindf = pd.read_json("C:/Users/Ju/Desktop/Kaggle What's cooking/train.json")
traindf['ingredients_clean_string'] = [' , '.join(z).strip() for z in traindf['ingredients']]  
traindf['ingredients_string'] = [' '.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line)) for line in lists]).strip() for lists in traindf['ingredients']]       

testdf = pd.read_json("C:/Users/Ju/Desktop/Kaggle What's cooking/test.json") 
testdf['ingredients_clean_string'] = [' , '.join(z).strip() for z in testdf['ingredients']]
testdf['ingredients_string'] = [' '.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line)) for line in lists]).strip() for lists in testdf['ingredients']]       

corpustr = traindf['ingredients_string']
corpusts = testdf['ingredients_string']


# feature engineering
train_long=len(traindf['ingredients_string'])
length_tr={'length': pd.Series(range(0,train_long))}

test_long=len(testdf['ingredients_string'])
length_ts={'length': pd.Series(range(0,test_long))}

df_train = pd.DataFrame(np.random.randn(train_long,1),columns=list('l'))
df_test = pd.DataFrame(np.random.randn(test_long,1),columns=list('l'))

def longueur(table,longueur,nom_serie):
    for i in range(0,longueur):
        nom_serie['l'][i]=len(table['ingredients_string'][i])

longueur(traindf,train_long,df_train)
longueur(testdf,test_long,df_test)

df_train2=MinMaxScaler().fit_transform(df_train)
df_test2=MinMaxScaler().fit_transform(df_test)

df_train3=pd.DataFrame(df_train2,columns=list('l'))
df_test3=pd.DataFrame(df_test2,columns=list('l'))

#Analyzer word rang(1,1)+SVC=0.78
#Analyzer char????
#différence???
vectorizertr = TfidfVectorizer(stop_words='english',
                             ngram_range = ( 1 , 1 ),analyzer="word", 
                             max_df=.57, binary=False , token_pattern=r'\w+' , sublinear_tf=False, norm='l2')
tfidftr=vectorizertr.fit_transform(corpustr).todense()
tfidftr.shape

#♥V2=HashingVectorizer(n_features=7000)
#V3=V2.fit_transform(corpustr).todense()

corpusts = testdf['ingredients_string']
vectorizerts = TfidfVectorizer(stop_words='english',
                             ngram_range = ( 1 , 1 ),analyzer="word", 
                             max_df=.57, binary=False , token_pattern=r'\w+' , sublinear_tf=False)
tfidfts=vectorizertr.transform(corpusts).todense()

#V22=HashingVectorizer(n_features=7000)
#V33=V22.fit_transform(corpusts).todense()

#predictors_tr = np.hstack([tfidftr, V3])

targets_tr = encoder.fit_transform(traindf['cuisine'])

#predictors_ts = np.hstack([tfidfts, V33])

predictors_tr=tfidftr
predictors_ts=tfidfts
print(predictors_ts.shape)
print(predictors_tr.shape)


#Feature Selection
def feature_selection(train_instances):
    print('Crossvalidation started... ')
    selector = VarianceThreshold(threshold=0.00000000001)
    selector.fit(train_instances)
    print('Number of features used... ' +
              str(Counter(selector.get_support())[True]))
    print('Number of features ignored... ' +
              str(Counter(selector.get_support())[False]))
    return selector

#Learn the features to filter from train set
fs=feature_selection(predictors_tr)
test=fs.transform(predictors_ts)
train=fs.transform(predictors_tr)

#Selection de variables
combined_features = FeatureUnion([("feat", SelectPercentile(chi2,Percentile=95)), 
                                  ("pca", PCA(n_componets=5))])
# Use combined features to transform dataset:
train_features = combined_features.fit(train, targets_tr).transform(train)
#☺combined_features = FeatureUnion([("pca", pca), ("univ_select", selection)])
# Use combined features to transform dataset:
#X_features = combined_features.fit(X_train, y_train).transform(X_train)
#X_test_features=combined_features.transform(X_test)
df_predictors_tr=pd.DataFrame(predictors_tr)
df_predictors_ts=pd.DataFrame(predictors_ts)

train=df_predictors_tr.join(df_train3)
test=df_predictors_ts.join(df_test3)

test=predictors_ts
train=predictors_tr

test=pd.DataFrame(predictors_ts).join(df_test3)
train=pd.DataFrame(predictors_tr).join(df_train3)

#test=df_predictors_ts.join(df_test3)
#train=df_predictors_tr.join(df_train3)

#pca = PCA(n_components=50)
#predictors_tr.shape
#pca_fit=pca.fit(predictors_tr)
#pca_fit2=pca.fit(predictors_ts)
#train=pca_fit.transform(predictors_tr)
#test=pca_fit.transform(predictors_ts)

#selection=SelectKBest(chi2, k=1)
#combined_features = FeatureUnion([("pca", pca), ("univ_select", selection)])
#train_feat = combined_features.fit(predictors_tr, targets_tr).transform(predictors_tr)
#train_feat.shape

#chi2_fit=chi2.fit(predictors_tr,targets_tr)
#train_chi2=chi2_fit.transform(predictors_tr)
#train_chi2.shape

#Classifier = OneVsRestClassifier()
#C=1
#svc = svm.SVC(kernel='linear', C=C).fit(train, targets_tr)
#rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(train, targets_tr)
#poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(train, targets_tr)
#lin_svc = svm.LinearSVC(C=C).fit(train, targets_tr)

#classifier = SVC(kernel="linear", C=0.025)
#parameters = {'C':[1, 10]}
#clf = RandomForestClassifier(n_estimators=500,n_jobs=-1)
parameters = {"max_depth": [None,3,5,10,20],
              "min_samples_split": [1,2,5,10],
              "min_samples_leaf": [1,2,5,10],
              "bootstrap": [True],
              "criterion": ['gini','entropy']}
              
parameters = {"C":  [1.1,1,0.9,0.95,0.80,0.85,0.75]
              #"loss": ['hinge'],
              #"penalty": ['l1'],
              #"dual": [True,False]
              #○"tol": [0.00001],
              #"max_iter": [1000],
              #"multi_class": ['ovr','crammer_singer']
              }

parameters = {"C":  [0.5,1,5,10],
              "penalty":['l1','l2'],
              "solver":["newton-cg", "lbfgs", "liblinear"],
              #"loss": ['hinge'],
              #"penalty": ['l1'],
              #"dual": [True,False]
              #○"tol": [0.00001],
              #"max_iter": [1000],
              "multi_class": ['ovr','multinomial']
              }

from scipy.stats import randint as sp_randint
from sklearn.grid_search import RandomizedSearchCV

param_dist = {"eta":[0.05,0.1,0.2,0.5,0.6,0.9,1],
              "gamma":sp_randint(0,10),
              "max_depth":sp_randint(1,50),
              "min_child_weight":sp_randint(1,100),
              "max_delta_step":sp_randint(1,10),
              "subsample":[0.1,0.5,0.75,1],
              "colsample_bytree":[0.1,0.5,0.75,1],
              "objective":["multi:softmax"],
              "eval_metric":["map"],
              "num_class":[20]
              }

param_dist = {"eta":0.9,
              "gamma":0,
              "max_depth":25,
              "min_child_weight":5,
              "max_delta_step":5,
              "subsample":1,
              "colsample_bytree":1,
              "objective":"multi:softmax",
              "eval_metric":"map",
              "num_class":20
              }

    
x_train, x_test, y_train, y_test = train_test_split(train, targets_tr, test_size=0.40,random_state =42)
#x_train_pca, x_test_pca, y_train_pca, y_test_pca = train_test_split(train_pca, targets_tr, test_size=0.40)
n_iter_search = 10
dtrain=xgb.DMatrix(x_train,label=y_train)
dtest=xgb.DMatrix(x_test)
watchlist = [(dtrain,'train'), (dtest, 'test')]
clf=xgb.train(param_dist,dtrain, num_boost_round=1000, evals=(), 
              early_stopping_rounds=10, evals_result=[(dtest,"eval"), (x_train,"train")], verbose_eval=True)
              
random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                   n_iter=n_iter_search)

start_time=time.time()
dtrain=xgb.DMatrix(x_train,label=y_train)
dtest=xgb.DMatrix(x_test)
bst = xgb.train(param_dist,dtrain)
ypred = bst.predict(dtest,ntree_limit=bst.best_ntree_limit)
gmbscore = accuracy_score(y_test, ypred)
interval=time.time()-start_time
####################################################################
#Probabilistes
clf = RandomForestClassifier(n_estimators=750,n_jobs=-1,verbose=10)
clf_rf=clf.fit(x_train, y_train)
prediction_rf=clf_rf.predict(x_test)
score_rf = accuracy_score(y_test, prediction_rf)

clf_ext = ExtraTreesClassifier(n_estimators=750,n_jobs=-1,verbose=10)
clf_ext.fit(x_train, y_train)
#mod1=clf_ext.predict_proba(x_test)
score_ext = accuracy_score(y_test, clf_ext.predict(x_test))

#clf_ext=ExtraTreesClassifier(bootstrap=True, class_weight=None, criterion='gini',
#           max_depth=None, max_features='auto', max_leaf_nodes=None,
#           min_samples_leaf=1, min_samples_split=1,
#           min_weight_fraction_leaf=0.0, n_estimators=750, n_jobs=-1,
#           oob_score=False, random_state=None, verbose=10,
#           warm_start=False)
#Version log améliore le modèle
svc_clf8 = LinearSVC(C=0.8)          
sig_clf = CalibratedClassifierCV(svc_clf8.fit(x_train, y_train),method="sigmoid", cv="prefit")
sig_clf.fit(x_train, y_train)
prediction_sig=sig_clf.predict(x_test)
mod2=sig_clf.predict_proba(x_test)
sig_score = accuracy_score(y_test, prediction_sig)
prediction_test=sig_clf.predict(test)


#clf_ext =  xgb.XGBClassifier(max_depth=3, n_estimators=200, 
#                        learning_rate=0.09,objective="multi:softmax")
#clf_ext.fit(prediction_proba, y_test)
#prediction_ext=clf_ext.predict(sig_clf.predict_proba(test))

logistic_clf = LogisticRegression(C=5)
logistic_clf.fit(x_train, y_train)
prediction_logistic=logistic_clf.predict(x_test)
mod3=logistic_clf.predict_proba(x_test)
logistic_score = accuracy_score(y_test, prediction_logistic)

eclf1 = VotingClassifier(estimators=[('lsvc', LinearSVC(C=0.8)), 
('ext', ExtraTreesClassifier(n_estimators=750,n_jobs=-1,verbose=10))])
 
eclf1 = eclf1.fit(x_train, y_train)
prediction_eclf1=eclf1.predict(x_test)
eclf1_score = accuracy_score(y_test, prediction_eclf1)

mod11=pd.DataFrame(mod1)
mod22=pd.DataFrame(mod2)
mod33=pd.DataFrame(mod3)

mod_final=mod11.join(mod22,rsuffix='_r').join(mod33,rsuffix='_s')

ada_clf=AdaBoostClassifier(n_estimators=1000)
ada_clf.fit(mod_final, y_test)
#prediction_ada=ada_clf.predict(x_test)
#ada_score = accuracy_score(y_test, prediction_ada)

start_time=time.time()
gbm = xgb.XGBClassifier(max_depth=5, n_estimators=200, 
                        learning_rate=0.09,objective="multi:softmax").fit(x_train, y_train)
prediction_gbm = gbm.predict(x_test)
gmbscore = accuracy_score(y_test, prediction_gbm)
interval=time.time()-start_time
#eta=0.3 max_depth=25 obj=mult num_class=20 

#Ensemblistes
svc_clf8 = LinearSVC(C=0.8)
svc_clf8.fit(np.log(x_train+1), y_train)
decision_svc=svc_clf8.decision_function(x_test)
prediction_svc8=svc_clf8.predict(x_test)
svc_score8 = accuracy_score(y_test, prediction_svc8)

Ridge_clf = RidgeClassifier(alpha=1)
Ridge_clf.fit(x_train, y_train)
decision_ridge=Ridge_clf.decision_function(x_test)
prediction_ridge=Ridge_clf.predict(x_test)
Ridge_clf_score = accuracy_score(y_test, prediction_ridge)

PAC_clf = PassiveAggressiveClassifier(C=0.1)
PAC_clf.fit(x_train, y_train)
decision_pac=PAC_clf.decision_function(x_test)
prediction_PAC=PAC_clf.predict(x_test)
PAC_clf_score = accuracy_score(y_test, prediction_PAC)

from sklearn.linear_model import RandomizedLogisticRegression
RandomizedLogisticRegression_clf = RandomizedLogisticRegression(C=5,n_jobs=-1)
RandomizedLogisticRegression_clf.fit(x_train, y_train)

prediction_RandomizedLogisticRegression=RandomizedLogisticRegression_clf.predict(x_test)
RandomizedLogisticRegression_clf_score = accuracy_score(y_test, prediction_RandomizedLogisticRegression)

####################################################################
#Affichage des score des différents modèles
print('Score modele %s est de %s' % ('RF',score_rf))
print('Score modele %s est de %s' % ('Ext',score_ext))
print('Score modele %s est de %s' % ('Sig',sig_score))
print('Score modele %s est de %s' % ('Reg Log',logistic_score))
print('Score modele %s est de %s' % ('SVC C=0.8',svc_score8))
print('Score modele %s est de %s' % ('XGB',gmbscore))
print('Score modele %s est de %s' % ('Ridge',Ridge_clf_score))
print('Score modele %s est de %s' % ('PAC',PAC_clf_score))
####################################################################
#stratified k fold
skf=list(StratifiedKFold(targets_tr,5))
cv=KFold(train.shape[0], n_folds=5,shuffle=False)
modele=[LogisticRegression(C=5),RandomForestClassifier(n_estimators=750,n_jobs=-1,verbose=10),
        ExtraTreesClassifier(n_estimators=750,n_jobs=-1,verbose=10),
        CalibratedClassifierCV(LinearSVC(C=0.8).fit(x_train,y_train), method="sigmoid", cv="prefit")]
Indice=[0,1,2,3]  
score_skf=[0,1,2,3]
score_cv=[0,1,2,3]
score=[0,1,2,3]
     
for i in Indice:
    mod=modele[i]
    print(i)
    for k in enumerate(skf):
      mod.fit(x_train, y_train)
      score_skf[i]=mod.score(x_test, y_test)

for i in Indice:
    mod=modele[i]                                   
    for k in enumerate(cv):
      mod.fit(x_train, y_train)
      score_cv[i]=mod.score(x_test, y_test)

for i in Indice:
    mod=modele[i]                                   
    mod.fit(x_train, y_train)
    score[i]=mod.score(x_test, y_test)

print('Score sans CV')
print(score[0],score[1],score[2],score[3])
print('Score avec CV')
print(score_cv[0],score_cv[1],score_cv[2],score_cv[3])
print('Score avec CV skf')
print(score_skf[0],score_skf[1],score_skf[2],score_skf[3])

# CROSS VALIDATION sur svc
skf=list(StratifiedKFold(targets_tr,5))
cv=KFold(train.shape[0], n_folds=5,shuffle=False)  
clf_ext=ExtraTreesClassifier(n_estimators=750,n_jobs=-1,verbose=10)
clf_sig =CalibratedClassifierCV(LinearSVC(C=0.8).fit(train,targets_tr), method="sigmoid", cv="prefit")

#for k in enumerate(skf):
#    clf_skf.fit(train, targets_tr)

                                   
for k, (tr, ts) in enumerate(cv):
     clf_ext.fit(train, targets_tr)
     print(k)

for k, (tr, ts) in enumerate(cv):
     clf_sig.fit(train, targets_tr)
     print(k)

clf_skf_probs=clf_skf.predict_proba(x_test)
clf_probs=clf.predict_proba(x_test)
#pred_cv=clf.predict(train)
#print(accuracy_score(targets_tr, pred_cv))

#0.8615678584
#print(clf.score(x_test, y_test))
#print(svc_score8)

#Réalisation d'un blend pour classifier avec proba
#clf_rf_probs=clf_rf.predict_proba(x_test)
clf_ext = ExtraTreesClassifier(n_estimators=750,n_jobs=-1,verbose=10)
clf_ext.fit(np.log(x_train+1), y_train)

svc_clf8 = LinearSVC(C=0.8)          
sig_clf = CalibratedClassifierCV(svc_clf8.fit(np.log(x_train+1), y_train),method="sigmoid", cv="prefit")
sig_clf.fit(np.log(x_train+1), y_train)

clf_ext_probs=clf_ext.predict_proba(test)
clf_sig_probs=clf_sig.predict_proba(test)

clf_ext_probs_tr=clf_ext.predict_proba(train)
clf_sig_probs_tr=clf_sig.predict_proba(train)
#clf_log_probs=logistic_clf.predict_proba(x_test)
#clf_xgboost=gbm.predict_proba(x_test)
#clf_ridge_probs=Ridge_clf.predict_proba(x_test)
#clf_pac_probs=PAC_clf.predict_proba(x_test)

# blend de classifier probabilistes
modela=[clf_ext_probs,clf_sig_probs]
model_char=['Extra Trees','Sigmoid']
Indice=[0,1]


for i in Indice:
    model_a=modela[i]
    
    poids_score={'Poids a' : pd.Series(range(1,100)),
                 'Poids b':pd.Series(range(1,100)),
                 'Score':pd.Series(range(1,100))}

    for j in Indice:
        model_b=modela[j]
        for w in zip(range(1,100,1),range(99,0,-1)):
            score = accuracy_score(y_test,(model_a*w[0]/100.0+model_b*w[1]/100.0).argmax(1))
            poids_score['Poids a'][w[1]-1]=w[0]
            poids_score['Poids b'][w[1]-1]=w[1]
            poids_score['Score'][w[1]-1]=score*1000000000000
            if w[0]==99:
                df_score=pd.DataFrame(poids_score)
                #print("Poids du modèle 1")
                #print(df_score['Poids a'][df_score['Score'].argmax()])
                #print("Poids du modèle 2")
                #print(df_score['Poids b'][df_score['Score'].argmax()])
                print(model_char[i], model_char[j], df_score['Poids a'][df_score['Score'].argmax()] , 
                                 df_score['Poids b'][df_score['Score'].argmax()],df_score['Score'][df_score['Score'].argmax()])      


cv=KFold(train.shape[0], n_folds=5,shuffle=False)  

def cross_v(clf):                                   
    for k, (tr, ts) in enumerate(cv):
        clf.fit(train, targets_tr)
        print(k)

skf=list(StratifiedKFold(targets_tr,5))

def cross_skf(clf)
    for k in enumerate(skf):
        clf.fit(train, targets_tr)
        print(k)
#avec cv skf
        
#avec cv
svc_clf8 = LinearSVC(C=0.8)
clf1 = CalibratedClassifierCV(svc_clf8.fit(train, targets_tr),method="sigmoid", cv="prefit")
cross_v(clf1)
prediction_sigmoid=clf1.predict_proba(test)

clf2 = ExtraTreesClassifier(n_estimators=750,n_jobs=-1,verbose=10)
cross_v(clf2)
prediction_ext=clf2.predict_proba(test)
#sans cv
svc_clf8 = LinearSVC(C=0.8)
clf1 = CalibratedClassifierCV(svc_clf8.fit(train, targets_tr),method="sigmoid", cv="prefit")
clf1.fit(train, targets_tr)
prediction_sigmoid=clf1.predict_proba(test)

clf2 = ExtraTreesClassifier(n_estimators=750,n_jobs=-1,verbose=10)
clf2.fit(train, targets_tr)
prediction_ext=clf2.predict_proba(test)
#cv et sigmoid log
logistic_clf = LogisticRegression(C=5)
clf1 = CalibratedClassifierCV(logistic_clf.fit(train, targets_tr),method="sigmoid", cv="prefit")
cross_v(clf1)
prediction_sigmoid=clf1.predict_proba(test)

clf2 = ExtraTreesClassifier(n_estimators=750,n_jobs=-1,verbose=10)
cross_v(clf2)
prediction_ext=clf2.predict_proba(test)

# best estimator 13k features
score_blend = accuracy_score(y_test, (prediction_logistic*53/100+prediction_ext*47/100).argmax(1))
#best estimators 3k features sigmoid(svc) + ext
score_blend = accuracy_score(y_test, (prediction_sigmoid*35/100+prediction_ext*65/100).argmax(1))
#best estimators 3k features sigmoid(log) + ext
score_blend = accuracy_score(y_test, (prediction_sigmoid*47/100+prediction_ext*53/100).argmax(1))
#feature engineering taille ingred 
score_blend = accuracy_score(y_test, (prediction_sigmoid*34/100+prediction_ext*66/100).argmax(1))
##
score_blend = accuracy_score(targets_tr, (clf_sig_probs_tr*34/100+clf_ext_probs_tr*66/100).argmax(1))
#Log san s cv
clf_ext = ExtraTreesClassifier(n_estimators=750,n_jobs=-1,verbose=10)
clf_ext.fit(np.log(train+1), targets_tr)
prediction_ext=clf_ext.predict_proba(test)

svc_clf8 = LinearSVC(C=0.8)          
sig_clf = CalibratedClassifierCV(svc_clf8.fit(np.log(train+1), targets_tr),method="sigmoid", cv="prefit")
sig_clf.fit(np.log(train+1), targets_tr)
prediction_sig=sig_clf.predict_proba(np.log(test+1))



#### Blend de Classifier Ensemblistes 
modela=[decision_svc,decision_ridge,decision_pac]
model_char=['SVC','Ridge','PAC']
Indice=[0,1,2]


for i in Indice:
    model_a=modela[i]
    
    poids_score={'Poids a' : pd.Series(range(1,100)),
                 'Poids b':pd.Series(range(1,100)),
                 'Score':pd.Series(range(1,100))}

    for j in Indice:
        model_b=modela[j]
        for w in zip(range(1,100,1),range(99,0,-1)):
            score = accuracy_score(y_test,(model_a*w[0]/100.0+model_b*w[1]/100.0).argmax(1))
            poids_score['Poids a'][w[1]-1]=w[0]
            poids_score['Poids b'][w[1]-1]=w[1]
            poids_score['Score'][w[1]-1]=score*1000000000000
            if w[0]==99:
                df_score=pd.DataFrame(poids_score)
                #print("Poids du modèle 1")
                #print(df_score['Poids a'][df_score['Score'].argmax()])
                #print("Poids du modèle 2")
                #print(df_score['Poids b'][df_score['Score'].argmax()])
                print(model_char[i], model_char[j], df_score['Poids a'][df_score['Score'].argmax()] , 
                                 df_score['Poids b'][df_score['Score'].argmax()],df_score['Score'][df_score['Score'].argmax()])      


#####Stacking 2
clf_ext = ExtraTreesClassifier(n_estimators=750,n_jobs=-1,verbose=10)
clf_ext.fit(train, targets_tr)
mod1=clf_ext.predict_proba(train)

svc_clf8 = LinearSVC(C=0.8)          
sig_clf = CalibratedClassifierCV(svc_clf8.fit(train, targets_tr),method="sigmoid", cv="prefit")
sig_clf.fit(train, targets_tr)
mod2=sig_clf.predict_proba(train)

svc_clf88 = LinearSVC(C=0.8)          
sig_clf2 = CalibratedClassifierCV(svc_clf88.fit(np.log(train+1), targets_tr),method="sigmoid", cv="prefit")
sig_clf2.fit(np.log(train+1), targets_tr)
mod4=sig_clf2.predict_proba(np.log(train+1))

#mod5=(65/100)*mod1+(35/100)*mod2

logistic_clf = LogisticRegression(C=5)
logistic_clf.fit(train, targets_tr)
mod3=logistic_clf.predict_proba(train)

mod11=pd.DataFrame(mod1)
mod22=pd.DataFrame(mod2)
mod33=pd.DataFrame(mod3)
mod44=pd.DataFrame(mod4)
#mod55=pd.DataFrame(mod5)

mod_final=mod11.join(mod22,rsuffix='_r').join(mod33,rsuffix='_s').join(mod44,rsuffix='_t')#.join(mod55,rsuffix='_u')
#mod_final=mod22.join(mod33,rsuffix='_r').join(mod44,rsuffix='_s')

#st_clf=LogisticRegression(C=0.5)
logistic_clf2=LogisticRegression(C=5)
st_clf=CalibratedClassifierCV(logistic_clf2.fit(mod_final, targets_tr),method="sigmoid", cv="prefit")
#LogisticRegression(C=0.5) 0.80219
#cv=KFold(train.shape[0], n_folds=5,shuffle=False)  
#def cross_v(clf):                                   
#    for k, (tr, ts) in enumerate(cv):
#        clf.fit(mod_final, targets_tr)
#        print(k)
#cross_v(st_clf)  

st_clf.fit(mod_final, targets_tr)      
st_clf.predict(mod_final)

mod111=pd.DataFrame(clf_ext.predict_proba(test))
mod222=pd.DataFrame(sig_clf.predict_proba(test))
mod333=pd.DataFrame(logistic_clf.predict_proba(test))
mod444=pd.DataFrame(sig_clf2.predict_proba(np.log(test+1)))
#mod555=pd.DataFrame((65/100)*clf_ext.predict_proba(test)+(35/100)*sig_clf.predict_proba(test))

mod_final2=mod111.join(mod222,rsuffix='_r').join(mod333,rsuffix='_s').join(mod444,rsuffix='_t')#.join(mod555,rsuffix='_u')
#mod_final2=mod222.join(mod333,rsuffix='_r').join(mod444,rsuffix='_s')

predictions=st_clf.predict(mod_final2)

#encoder.inverse_transform
prediction_ext=clf_skf.predict_proba(test)
prediction_sigmoid=clf.predict_proba(test)
#####################################################
predictions=(prediction_sigmoid*35/100+prediction_ext*65/100).argmax(1)
predictions=(prediction_sig*40/100+prediction_ext*60/100).argmax(1)
predictions=(clf_sig_probs*35/100+clf_ext_probs*65/100).argmax(1)
#targets_tr = encoder.fit_transform(traindf['cuisine'])
predcition_blend=encoder.inverse_transform(predictions)
testdf['cuisine'] = predcition_blend
testdf = testdf.sort('id' , ascending=True)

#testdf[['id' , 'ingredients_clean_string' , 'cuisine' ]].to_csv("C:\Users\Ju\Desktop\Kaggle What's cooking\submission.csv",encoding='utf-8')
testdf[['id' , 'cuisine']].to_csv("C:/Users/Ju/Desktop/Kaggle What's cooking/submission_blen_featu.csv",encoding='utf-8',index=False,index_col = False,columns=['id','cuisine'])