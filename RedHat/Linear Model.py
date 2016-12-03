from sklearn.cross_validation import train_test_split
from sklearn import linear_model
from sklearn import metrics

path2='/Users/stephanieperignon/Desktop/DataScience/redhat'
lootrain = pd.read_csv(path2+"/train_linear_model.csv")
lootest = pd.read_csv(path2+"/test_linear_model.csv")
                       
def quick_dirty_LM(clf,train_num,test_num,mode,name):
    if mode=='test_LR':
        x_train, x_test, y_train, y_test = train_test_split(train_num,to_predict,test_size=0.2,stratify=to_predict)
        clf.fit(x_train,y_train)
        pred=clf.predict_proba(x_test)[:,1]
        print(metrics.roc_auc_score(y_test,pred))
    if mode=='submit_LR':
        clf.fit(train_num,to_predict)
        pred=pd.DataFrame(clf.predict(test_num))
        submit(name,pred)
    if mode=='test_no_LR':
        x_train, x_test, y_train, y_test = train_test_split(train_num,to_predict,test_size=0.2,stratify=to_predict)
        clf.fit(x_train,y_train)
        pred=clf.predict(x_test)
        print(metrics.roc_auc_score(y_test,pred))
    if mode=='submit_no_LR':
        clf.fit(train_num,to_predict)
        pred=pd.DataFrame(clf.predict(test_num))
        submit(name,pred)
    return pred

#linear model with LOO
def LOO(data1, data2, columnName, useLOO=False):
    grpOutcomes = data1.groupby(columnName).mean().reset_index()
    outcomes = data2['outcome'].values
    x = pd.merge(data2[[columnName, 'outcome']], grpOutcomes,
                 suffixes=('x_', ''),
                 how='left',
                 on=columnName,
                 left_index=True)['outcome']
    if(useLOO):
        x = ((x*x.shape[0])-outcomes)/(x.shape[0]-1)
    return x.fillna(x.mean())

def LOO_use(train_num,test_num,features): 
    for i in features:
        if i=='activity_category':
            lootrain=pd.DataFrame(LOO(train_num,train_num,i,True).values,columns=[i]) 
            lootest=pd.DataFrame(LOO(train_num,test_num,i,False).values,columns=[i])
        if i!='activity_category' and i!='outcome':
            print(i)
            lootrain=lootrain.join(pd.DataFrame(LOO(train_num,train_num,i,True).values,columns=[i]))
            lootest=lootest.join(pd.DataFrame(LOO(train_num,test_num,i,False).values,columns=[i]))
        
    return lootrain, lootest

#### Data for linear model
#test_num['outcome'] = 0
#lootrain, lootest = LOO_use(train_num,test_num,features)
#path2='/Users/stephanieperignon/Desktop/DataScience/redhat'

#def submit2(table,name):
#    table.to_csv(path2+name+'.csv',index=False,header=list(table.columns.values),dtype=int)

#submit2(lootrain,'train_linear_model')
#submit2(lootest,'test_linear_model')
####

#test. to delete
def Linear_Model(clf,mode):
    if mode=='LR':
        clf.fit(pd.DataFrame(lootrain),y_train)
        pred=clf.predict_proba(pd.DataFrame(lootest))[:,1]
        print(metrics.roc_auc_score(y_test,pd.DataFrame(pred)))
    if mode!='LR':
        clf.fit(pd.DataFrame(lootrain),y_train)
        pred=clf.predict(pd.DataFrame(lootest))
        print(metrics.roc_auc_score(y_test,pd.DataFrame(pred)))
    return pred
    
#pred1=Linear_Model(linear_model.Ridge(),'ridge') 
#pred2=Linear_Model(linear_model.LinearRegression(),'lasso') 
#pred3=Linear_Model(linear_model.LogisticRegression(),'LR')
#pred4=Linear_Model(linear_model.BayesianRidge(),'baysian')

#pred_all=pred1+pred2+3*pred3+3*pred4
#print(metrics.roc_auc_score(y_test,pd.DataFrame(pred_all)))

quick_dirty_LM(linear_model.Ridge(),lootrain,lootest,'submit_no_LR','Ridge_all_loo')  
#quick_dirty_LM(linear_model.LogisticRegression(),lootrain[['char_38','group_1','char_2_x']],lootest[['char_38','group_1','char_2_x']],'submit_LR','LR')  
#quick_dirty_LM(linear_model.BayesianRidge(),lootrain[['char_38','group_1','char_2_x']],lootest[['char_38','group_1','char_2_x']],'submit_no_LR','Bayesian_ridge')  

#transformation woth random tree embedding
 