from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn import metrics
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.feature_selection import SelectFromMode
from sklearn.cross_validation import StratifiedKFold

def quick_dirty_Forest(clf,x_train,x_test,y_train,y_test,feat,mode,name):
    if mode=='test': 
        #x_train, x_test, y_train, y_test = train_test_split(train_num[feat],train_num['outcome'],random_state=42,test_size=0.2,stratify=train_num['outcome'])
        clf.fit(x_train[feat],y_train)
        pred=clf.predict_proba(x_test[feat])[:,1]
        print(metrics.roc_auc_score(y_test,pred))

    if mode=='submit':
        clf.fit(x_train[feat],y_train)
        pred=pd.DataFrame(clf.predict_proba(x_test[feat])[:,1],columns=['outcome'])    
        submit(name,pred)
        
    if mode=='submit_cv':
        clf.fit(x_train[feat],y_train)
        pred=pd.DataFrame(clf.predict_proba(x_test[feat])[:,1],columns=['outcome'])            
        kf = StratifiedKFold(target_train, n_folds=5)
        for train_indices, test_indices in kf:
            X_train, X_test = X.ix[train_indices], X.ix[test_indices]
            y_train, y_test = labels.ix[train_indices], labels.ix[test_indices]
            clf.fit(X_train,y_train)
            
        submit(name,pred)
    if mode=='submit_nodup':
        x_train = x_train[feat].join(y_test).join(y_train)
        x_train_nodup = x_train.drop_duplicates(['people_id','outcome'])
        to_predict = x_train_nodup['outcome']
        x_train_nodup.drop('outcome',axis=1,inplace=True)
        x_train_nodup.drop('people_id',axis=1,inplace=True)
        
        clf.fit(x_train_nodup,to_predict)
        pred=pd.DataFrame(clf.predict_proba(x_test[feat])[:,1],columns=['outcome'])    
        submit(name,pred)
        
    return pred

#train_num.join(to_predict).drop_duplicates(['people_id','outcome'])

def RF_one_feature(clf,train_num,features):
    x_train, x_test, y_train, y_test = train_test_split(train_num,to_predict,test_size=0.2,stratify=to_predict)
    m = []
    features4 = []
    for i in features:
        clf.fit(pd.DataFrame(x_train[i]),y_train)
        pred=clf.predict_proba(pd.DataFrame(x_test[i]))[:,1]
        print("AUC for",i,metrics.roc_auc_score(y_test,pred))
        a=metrics.roc_auc_score(y_test,pred)
        m.append(a)
        if a>0.501:
            features4.append(i)
    out=pd.DataFrame(m,columns=['AUC']).join(pd.DataFrame(features,columns=['features']))
    clf.fit(pd.DataFrame(x_train[features]),y_train)
    pred=clf.predict_proba(pd.DataFrame(x_test[features]))[:,1]
    print("AUC with all features:",metrics.roc_auc_score(y_test,pred))    
    return m,features4

#selection with extra trees
#clf_select = RandomForestClassifier(n_estimators=100,n_jobs=-1)
#clf_select.fit(train_num[features],to_predict)
#model = SelectFromModel(clf_select, prefit=True)
#train_num_select_ET = pd.DataFrame(model.transform(train_num[features_bis]))
#test_num_select_ET = pd.DataFrame(model.transform(test_num[features_bis]))

#AUC of features one by one with all 0.989721193269
#AUC,features_select=RF_one_feature(RandomForestClassifier(n_estimators=1,random_state=42,n_jobs=-1),train_num,features3)

#mode test with 100 trees:kaggle: 0.963895 on test: 0.999717521375 with prob and freq: 0.999906189094
#mode test with 200 trees:kaggle: 0.965441 on test:                with prob and freq: 
#mode test with 300 trees:kaggle: 0.965441 on test:                with prob and freq: 

#mode subm with 100 trees:kaggle: 0.964988 on test:
#mode subm with 200 trees:kaggle: 0.965999 on test:
#mode subm with 300 trees:kaggle: 0.966296 on test:

#features0
#features
#features3
#features_bis
#hasher = RandomTreesEmbedding(n_estimators=10,random_state=42)
#train_transformed=hasher.fit_transform(train_num[features])
#test_transformed=hasher.transform(test_num[features])

clf1 = RandomForestClassifier(n_estimators=700,random_state=42,n_jobs=-1,verbose=100)

#pred1=quick_dirty_Forest(clf2,train_num[features0],test_num[features0],'submit','ET_100_V0')
#pred1=quick_dirty_Forest(clf1,train_num[features3],test_num[features3],'test','RF_100_with_max_std')
#pred2=quick_dirty_Forest(clf2,train_num[features2],test_num[features2],'submit','ET_100_without')
#features2 = features + ['freq_people']

#pred1=quick_dirty_Forest(clf1,train_num[features0],test_num[features0],'test','RF_100_with_max_std')
#pred1=quick_dirty_Forest(clf1,train_num[features],test_num[features],'test','RF_100_with_max_std')
#pred1=quick_dirty_Forest(clf1,train_num[features3],test_num[features3],'test','RF_100_with_max_std')
#pred1=quick_dirty_Forest(clf1,x_train,x_test,y_train,y_test,features,'test','RF_100_with_max_std')

pred1=quick_dirty_Forest(clf1,train_num,test_num,to_predict,train_num,features,'submit','RF_700')

#pred1=quick_dirty_Forest(clf1,train_num,test_num,to_predict,people_id_test,features,'submit_nodup','RF_100_None_nodup')

#pred1=quick_dirty_Forest(clf1,train_num,test_num,to_predict,y_test,features3,'submit','RF_100_None')

#pred1=quick_dirty_Forest(clf1,x_train2,x_test,y_train2,y_test,features,'test','RF_100_with_max_std')
#pred1=quick_dirty_Forest(clf1,x_train2,x_test,y_train2,y_test,features2,'test','RF_100_with_max_std')

#pred2=quick_dirty_Forest(clf1,train_num_select_ET,test_num_select_ET,'test','RF_100_with_max_std')
