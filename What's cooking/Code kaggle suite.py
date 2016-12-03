#Stacking
clf_rf = RandomForestClassifier(n_estimators=750,n_jobs=-1,verbose=10)
clf_rf.fit(x_train, y_train)
mod1=pd.DataFrame(clf_rf.predict_proba(x_test))

clf_ext = ExtraTreesClassifier(n_estimators=750,n_jobs=-1,verbose=10)
clf_ext.fit(x_train, y_train)
mod2=pd.DataFrame(clf_ext.predict_proba(x_test))

sig_clf = CalibratedClassifierCV(svc_clf8.fit(x_train, y_train), method="sigmoid", cv="prefit")
sig_clf.fit(x_train, y_train)
mod3=pd.DataFrame(sig_clf.predict_proba(x_test))

x=mod1.join(mod2).join(mod3)

def stack(model,train,y_train,test,nom):
    model.fit(train,test)
    nom=model.predict(test)

stack(LogisticRegression(C=5),x,y_test,stack_log)    
stack(RandomForestClassifier(n_estimators=750,n_jobs=-1,verbose=10),x,y_test,stack_rf)   
stack(ExtraTreesClassifier(n_estimators=750,n_jobs=-1,verbose=10),x,y_test,stack_ext)   
stack(CalibratedClassifierCV(clf, method="sigmoid", cv="prefit"),x,y_test,stack_sig)   

predictions=stack_log
testdf['cuisine'] = predictions
testdf = testdf.sort('id' , ascending=True)

testdf[['id' , 'cuisine']].to_csv("C:/Users/Ju/Desktop/Kaggle What's cooking/submission_stack_log.csv",
    encoding='utf-8',index=False,index_col = False,columns=['id','cuisine'])
    
    
#stratified k fold
skf=list(StratifiedKFold(targets_tr,5))
cv=KFold(train.shape[0], n_folds=5,shuffle=False)
modele=[LogisticRegression(C=5),RandomForestClassifier(n_estimators=750,n_jobs=-1,verbose=10),
        ExtraTreesClassifier(n_estimators=750,n_jobs=-1,verbose=10),
        CalibratedClassifierCV(LinearSVC(C=0.8).fit(X_train,y_ytrain), method="sigmoid", cv="prefit")]
Indice=[0,1,2,3]  
score_skf=[0,1,2,3]
score_cv=[0,1,2,3]
score=[0,1,2,3]
     
for i in Indice:
    mod=modele[i]
    for k in enumerate(skf):
      mod.fit(X_train, y_train)
      score_skf[i]=modele.score(X_test, y_test)

for i in Indice:
    mod=modele[i]                                   
    for k in enumerate(cv):
      mod.fit(X_train, y_train)
      score_cv[i]=modele.score(X_test, y_test)

for i in Indice:
    mod=modele[i]                                   
    mod.fit(X_train, y_train)
    score[i]=modele.score(X_test, y_test)

print('Score sans CV')
print(score[0],score[1],score[2],score[3],score[4],score[5])
print('Score avec CV')
print(score_cv[0],score_cv[1],score_cv[2],score_cv[3],score_cv[4],score_cv[5])
print('Score avec CV skf')
print(score_skf[0],score_skf[1],score_skf[2],score_skf[3],score_skf[4],score_skf[5])

#grid search
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

# run randomized search
n_iter_search = 10
watchlist = [(dtrain,'train'), (dtest, 'test')]
clf=xgboost.train(dtrain, num_boost_round=1000, evals=(), 
              early_stopping_rounds=10, evals_result=[(dtest,"eval"), (dtrain,"train")], verbose_eval=True)
              
random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                   n_iter=n_iter_search)
                                   
####"
n_estimators=100
#verif avec scikit
param_verif = {"eta":[0.1],
              "gamma":[0],
              "max_depth":[3],
              "min_child_weight":[1],
              "max_delta_step":[0],
              "subsample":[1],
              "colsample_bytree":[1],
              "objective":["multi:softmax"],
              "num_class":[20],
              "scale_pos_weight":[1],
              "colsample_bylevel":[1]
              }