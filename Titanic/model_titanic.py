import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
import math

# For .read_csv, always use header=0 when you know row 0 is the header row
df = pd.read_csv('C:/Users/Ju/Desktop/Titanic/titanic_lab5-master/train.csv', header=0)
test_submit=pd.read_csv('C:/Users/Ju/Desktop/Titanic/titanic_lab5-master/test.csv', header=0)
#submit=pd.read_csv('C:/Users/J099055/Desktop/Titanic/titanic_lab5-master/submit_sas.csv', header=0)
#submit=submit.fillna(0)
#submit=submit.astype(int)
#submit.dtypes
df.dtypes
df.info()
test_submit.info()
df.describe()

import pylab as P
df['Age'].hist()
P.show()

df['Age'].dropna().hist(bins=16, range=(0,80), alpha = .5)
P.show()

df['Gender'] = df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
test_submit['Gender'] = test_submit['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

median_ages = np.zeros((2,3))
median_ages

for i in range(0, 2):
    for j in range(0, 3):
        median_ages[i,j] = df[(df['Gender'] == i) & \
                              (df['Pclass'] == j+1)]['Age'].dropna().median()
 
median_ages

df['AgeFill'] = df['Age']
test_submit['AgeFill']=test_submit['Age']
df.head()
test_submit.head()

df[ df['Age'].isnull() ][['Gender','Pclass','Age','AgeFill']].head(10)
test_submit[ test_submit['Age'].isnull() ][['Gender','Pclass','Age','AgeFill']].head(10)

for i in range(0, 2):
    for j in range(0, 3):
        df.loc[ (df.Age.isnull()) & (df.Gender == i) & (df.Pclass == j+1),\
                'AgeFill'] = median_ages[i,j]

for i in range(0, 2):
    for j in range(0, 3):
        test_submit.loc[ (test_submit.Age.isnull()) & (test_submit.Gender == i) & (test_submit.Pclass == j+1),\
                'AgeFill'] = median_ages[i,j]

df.loc[df.Embarked.isnull()].index.values
                
df[ df['Age'].isnull() ][['Gender','Pclass','Age','AgeFill']].head(10)
test_submit[ test_submit['Age'].isnull() ][['Gender','Pclass','Age','AgeFill']].head(10)
df['AgeIsNull'] = pd.isnull(df.Age).astype(int)
test_submit['AgeIsNull'] = pd.isnull(test_submit.Age).astype(int)

df['FamilySize'] = df['SibSp'] + df['Parch']
df['Age*Class'] = df.AgeFill * df.Pclass
test_submit['FamilySize'] = test_submit['SibSp'] + test_submit['Parch']
test_submit['Age*Class'] = test_submit.AgeFill * test_submit.Pclass

df.dtypes[df.dtypes.map(lambda x: x=='object')]

#Embarked
df.loc[df.Embarked.isnull()].index.values
df   = df.fillna('S')

#Name
df['Title']=''
test_submit['Title']=''

for i in range(0,df.shape[0]):
    df['Title'][i]=df['Name'][i].split(',')[1:][0].split('.')[0:1][0].strip()

for i in range(0,test_submit.shape[0]):
    test_submit['Title'][i]=test_submit['Name'][i].split(',')[1:][0].split('.')[0:1][0].strip()
    
df['Title'].unique()
df['Title'].value_counts()

test_submit['Title'].unique()
test_submit['Title'].value_counts()

for i in range(0,df.shape[0]):
    df['Title'][i]=df['Title'][i].replace('Jonkheer','Lady')
    df['Title'][i]=df['Title'][i].replace('Dona','Lady')
    df['Title'][i]=df['Title'][i].replace('the Countess','Lady')
    df['Title'][i]=df['Title'][i].replace('Mlle','Lady')
    
    #A regrouper avec un KNN
    df['Title'][i]=df['Title'][i].replace('Ms','Miss')
    df['Title'][i]=df['Title'][i].replace('Mme','Miss')
    
    df['Title'][i]=df['Title'][i].replace('Capt','Master')
    df['Title'][i]=df['Title'][i].replace('Don','Master')
    df['Title'][i]=df['Title'][i].replace('Major','Master')
    df['Title'][i]=df['Title'][i].replace('Sir','Master')

for i in range(0,test_submit.shape[0]):    
    test_submit['Title'][i]=test_submit['Title'][i].replace('Jonkheer','Lady')
    test_submit['Title'][i]=test_submit['Title'][i].replace('Dona','Lady')
    test_submit['Title'][i]=test_submit['Title'][i].replace('the Countess','Lady')
    #A regrouper avec un KNN
    test_submit['Title'][i]=test_submit['Title'][i].replace('Ms','Miss')
    test_submit['Title'][i]=test_submit['Title'][i].replace('Mme','Miss')
    
    test_submit['Title'][i]=test_submit['Title'][i].replace('Capt','Master')
    test_submit['Title'][i]=test_submit['Title'][i].replace('Don','Master')
    test_submit['Title'][i]=test_submit['Title'][i].replace('Major','Master')
    test_submit['Title'][i]=test_submit['Title'][i].replace('Sir','Master')
 
Title1 = pd.get_dummies(df["Title"], prefix="Title")
df=df.drop(['Title'], axis=1).join(Title1)

Title2 = pd.get_dummies(test_submit["Title"], prefix="Title")
test_submit=test_submit.drop(['Title'], axis=1).join(Title2)
   
#df = df.drop(['Name', 'Sex', 'Ticket', 'Cabin','Embarked'], axis=1) 
df = df.drop(['Name','Sex', 'Ticket', 'Cabin'], axis=1) 
df = df.drop(['Age'], axis=1)
Embarked1 = pd.get_dummies(df["Embarked"], prefix="Embarked")
df=df.drop(['Embarked'], axis=1).join(Embarked1)
#pclass1 = pd.get_dummies(df["Pclass"], prefix="Pclass")
#df=df.drop(['Pclass'], axis=1).join(pclass1)

test_submit.dtypes[test_submit.dtypes.map(lambda x: x=='object')]
#test_submit = test_submit.drop(['Name', 'Sex', 'Ticket', 'Cabin','Embarked'], axis=1) 
test_submit = test_submit.drop(['Name','Sex', 'Ticket', 'Cabin'], axis=1) 
test_submit = test_submit.drop(['Age'], axis=1)
Embarked = pd.get_dummies(test_submit["Embarked"], prefix="Embarked")
test_submit=test_submit.drop(['Embarked'], axis=1).join(Embarked)
#pclass = pd.get_dummies(test_submit["Pclass"], prefix="Pclass")
#test_submit=test_submit.drop(['Pclass'], axis=1).join(pclass)
test_submit.dtypes

#Top adulte
Data_test={'Top_adulte' : pd.Series(range(test_submit.shape[0]))}
Data_train={'Top_adulte' : pd.Series(range(df.shape[0]))}

for i in range(0,test_submit.shape[0]):
    if test_submit['AgeFill'][i]<18.0 :
        Data_test['Top_adulte'][i]=0
    else:
        Data_test['Top_adulte'][i]=1

for i in range(0,df.shape[0]):
    if df['AgeFill'][i]<18.0 :
        Data_train['Top_adulte'][i]=0
    else:
        Data_train['Top_adulte'][i]=1

top_tr=pd.DataFrame(Data_train,columns=['Top_adulte'])   
top_ts=pd.DataFrame(Data_test,columns=['Top_adulte']) 
#On merge sur les index
test_submit=pd.merge(test_submit,top_ts,left_index=True,right_index=True,sort=False)
df=pd.merge(df,top_tr,left_index=True,right_index=True,sort=False)

#Mother
Data_test={'Mother' : pd.Series(range(test_submit.shape[0]))}
Data_train={'Mother' : pd.Series(range(df.shape[0]))}

for i in range(0,test_submit.shape[0]):
    if test_submit['AgeFill'][i]>18.0 and test_submit['Parch'][i]>0 and test_submit['Gender'][i]<1 and test_submit['Title_Miss'][i]>0:
        Data_test['Mother'][i]=1
    else:
        Data_test['Mother'][i]=0

for i in range(0,df.shape[0]):
    if df['AgeFill'][i]>18.0 and df['Parch'][i]>0 and df['Gender'][i]<1 and df['Title_Miss'][i]>0:
        Data_train['Mother'][i]=1
    else:
        Data_train['Mother'][i]=0

top_tr=pd.DataFrame(Data_train,columns=['Mother'])   
top_ts=pd.DataFrame(Data_test,columns=['Mother']) 
#On merge sur les index
test_submit=pd.merge(test_submit,top_ts,left_index=True,right_index=True,sort=False)
df=pd.merge(df,top_tr,left_index=True,right_index=True,sort=False)
#test_submit= test_submit.drop(['Embarked_C','Embarked_Q','Embarked_S'], axis=1)
#df= df.drop(['Embarked_C','Embarked_Q','Embarked_S'], axis=1)
#Verification et localisation des NaN
#for j in ['PassengerId', 'Pclass', 'SibSp', 'Parch', 'Fare', 'Gender', 'AgeFill', 'AgeIsNull', 'FamilySize', 'Age*Class']:
#   print(test_submit[ test_submit[j].isnull() ].head(10))

test_submit.loc[test_submit.Fare.isnull()].index.values

#rempalcement des NaN de FARE
test_submit   = test_submit.fillna(test_submit['Fare'].mean())
PassengerId=test_submit['PassengerId']
test_submit = test_submit.drop(['PassengerId'], axis=1)
df = df.drop(['PassengerId'], axis=1)
targets_tr=df['Survived']
train = df.drop(['Survived'], axis=1).values

targets_tr=df['Survived']
df.drop(['Survived'], axis=1).values

#Echantillons
x_train, x_test, y_train, y_test = train_test_split(train, targets_tr, test_size=0.30,random_state=42)

#Modélisdation
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

#pca = PCA(n_components=6)
#selection=SelectKBest(k=6)
#combined_features = FeatureUnion([("pca", pca), ("univ_select", selection)])

#train=pd.DataFrame(train)
#train = train.join(pd.DataFrame(combined_features.fit(train, targets_tr).transform(train)),rsuffix='r')
#test_submit=pd.DataFrame(test_submit)
#test_submit=test_submit.join(pd.DataFrame(combined_features.transform(test_submit)),rsuffix='r')

#x_train, x_test, y_train, y_test = train_test_split(train, targets_tr, test_size=0.30,random_state=42)

def modele(clf,tr,y_tr,ts,y_ts):
    #cv=KFold(tr.shape[0], n_folds=2,shuffle=False)
    #cv=StratifiedKFold(y_tr,n_folds=2)                               
    #for k in enumerate(cv):
    #     clf.fit(tr, y_tr)

    clf.fit(tr, y_tr)
    pred=clf.predict(ts)
    clf_score = accuracy_score(y_ts, pred)
    clf_score2=zero_one_loss(y_ts, pred)
    print(clf_score)
    print(clf_score2)
    print(confusion_matrix(y_ts, pred))
    global pred_final1
    pred_final1=clf.predict(test_submit)  
    #print(clf.feature_importances_)

pred_final1 = pd.DataFrame()
modele(RandomForestClassifier(n_estimators=1000,max_depth=5,criterion='entropy',max_features=6,
random_state=42),x_train,y_train,x_test,y_test)
modele(AdaBoostClassifier(n_estimators=750,learning_rate=0.01),x_train,y_train,x_test,y_test)
modele(BaggingClassifier(n_estimators=750,bootstrap_features=True)
,x_train,y_train,x_test,y_test)
modele(GradientBoostingClassifier(n_estimators=750,learning_rate=0.01,max_depth=5)
,x_train,y_train,x_test,y_test)
modele(ExtraTreesClassifier(n_estimators=750,max_depth=6),x_train,y_train,x_test,y_test)
modele(LogisticRegression(C=5,intercept_scaling=1),x_train,y_train,x_test,y_test)
modele(RidgeClassifier(alpha=0.01),x_train,y_train,x_test,y_test)
modele(LinearSVC(C=0.8),x_train,y_train,x_test,y_test)
modele(CalibratedClassifierCV(RandomForestClassifier(n_estimators=200,max_depth=5,criterion='entropy',max_features=6,
random_state=42).fit(x_train, y_train), method="sigmoid", cv="prefit"),x_train,y_train,x_test,y_test)

modele(tree.DecisionTreeClassifier( max_depth=3,max_features=22,random_state=42,splitter='best'),
       x_train,y_train,x_test,y_test)
       
modele(OneVsRestClassifier(tree.DecisionTreeClassifier( max_depth=3,max_features=22,random_state=42,splitter='best')
),x_train,y_train,x_test,y_test)      

modele(RidgeClassifierCV(alphas=(0.01, 1.0, 10.0),cv=10),x_train,y_train,x_test,y_test)
modele(GaussianNB(),x_train,y_train,x_test,y_test)


#Modèle par type de sexe
#Decoupage par type de gender
#pca = PCA(n_components=6)
#selection=SelectKBest(k=6)
#combined_features0 = FeatureUnion([("pca0", pca), ("univ_select0", selection)])
#combined_features1 = FeatureUnion([("pca1", pca), ("univ_select1", selection)])

df_0=df[df['Gender']==0].drop(['Survived'], axis=1)
df_1=df[df['Gender']==1].drop(['Survived'], axis=1)
df_0_index=df[df['Gender']==0].index.values
df_1_index=df[df['Gender']==1].index.values

#df_0 = combined_features0.fit(df_0, targets_tr[df_0_index]).transform(df_0)
#df_1 = combined_features1.fit(df_1, targets_tr[df_1_index]).transform(df_1)


test_submit_0=test_submit[test_submit['Gender']==0]
test_submit_1=test_submit[test_submit['Gender']==1]
test_submit_0_index=test_submit_0.index.values
test_submit_1_index=test_submit_1.index.values

#test_submit_0=combined_features0.transform(test_submit_0)
#test_submit_1=combined_features1.transform(test_submit_1)

test_submit_0_indexdf=pd.DataFrame(test_submit_0_index,columns=['index'])
test_submit_1_indexdf=pd.DataFrame(test_submit_1_index,columns=['index'])


x_train0, x_test0, y_train0, y_test0 = train_test_split(df_0, targets_tr[df_0_index], 
                                                        test_size=0.30,random_state=42)
x_train1, x_test1, y_train1, y_test1 = train_test_split(df_1, targets_tr[df_1_index], 
                                                        test_size=0.30,random_state=42)

def model_gender(clf0,clf1,tr0,y_tr0,ts0,y_ts0,tr1,y_tr1,ts1,y_ts1):
    clf0.fit(tr0,y_tr0)
    pred0=clf0.predict(ts0)
    clf0_score = accuracy_score(y_ts0, pred0)
    print(clf0_score)

    clf1.fit(tr1,y_tr1)
    pred1=clf1.predict(ts1)
    clf1_score = accuracy_score(y_ts1, pred1)
    print(clf1_score)
    y=pd.concat((y_ts0,y_ts1))
    pr=pd.concat((pd.Series(pred0),pd.Series(pred1)))
    clf_score=accuracy_score(y,pr)
    print(clf_score)
    
    pred_sub_0=pd.DataFrame(clf0.predict(test_submit_0),index=test_submit_0_index,columns=['Survived'])
    pred_sub_00=pd.DataFrame(pred_sub_0).join(pd.DataFrame(index=test_submit_0_index))
    pred_sub_1=pd.DataFrame(clf1.predict(test_submit_1),index=test_submit_1_index,columns=['Survived'])
    pred_sub_11=pd.DataFrame(pred_sub_1).join(pd.DataFrame(index=test_submit_1_index))
    global pred_final
    pred_final=pd.concat([pred_sub_00,pred_sub_11])

pred_final = pd.DataFrame()

model_gender(RandomForestClassifier(n_estimators=100,max_depth=6,max_features=5,random_state=42),
             RandomForestClassifier(n_estimators=100,max_depth=6,max_features=5,random_state=42),
x_train0,y_train0,x_test0,y_test0,x_train1,y_train1,x_test1,y_test1)
model_gender(AdaBoostClassifier(n_estimators=200,learning_rate=0.01),
             AdaBoostClassifier(n_estimators=200,learning_rate=0.01),
x_train0,y_train0,x_test0,y_test0,x_train1,y_train1,x_test1,y_test1)
model_gender(BaggingClassifier(n_estimators=750,bootstrap_features=True),
             BaggingClassifier(n_estimators=750,bootstrap_features=True),
x_train0,y_train0,x_test0,y_test0,x_train1,y_train1,x_test1,y_test1)
model_gender(GradientBoostingClassifier(n_estimators=750,learning_rate=0.01,max_depth=6),
             GradientBoostingClassifier(n_estimators=750,learning_rate=0.01,max_depth=6),
x_train0,y_train0,x_test0,y_test0,x_train1,y_train1,x_test1,y_test1)
model_gender(ExtraTreesClassifier(n_estimators=750,max_depth=6),
             ExtraTreesClassifier(n_estimators=750,max_depth=6),
x_train0,y_train0,x_test0,y_test0,x_train1,y_train1,x_test1,y_test1)
model_gender(LogisticRegression(C=10,intercept_scaling=1),
             LogisticRegression(C=10,intercept_scaling=1),
x_train0,y_train0,x_test0,y_test0,x_train1,y_train1,x_test1,y_test1)

#Meilleur modele
model_gender(GradientBoostingClassifier(n_estimators=750,learning_rate=0.01,max_depth=6),
              AdaBoostClassifier(n_estimators=200,learning_rate=0.01),
x_train0,y_train0,x_test0,y_test0,x_train1,y_train1,x_test1,y_test1)

test_submit_fin=pd.DataFrame(pd.DataFrame(PassengerId).join(pred_final),columns=['PassengerId','Survived'])

test_1=pd.DataFrame(pd.DataFrame(PassengerId).join(pd.DataFrame(pred_final1,columns=['Survived'])),
                    columns=['PassengerId','Survived'])
test_submit_fin[['PassengerId','Survived']].to_csv("C:/Users/J099055/Desktop/Titanic/titanic_lab5-master/submit.csv",
encoding='utf-8',index=False,index_col = False,columns=['PassengerId','Survived'])

test_1[['PassengerId','Survived']].to_csv("C:/Users/J099055/Desktop/Titanic/titanic_lab5-master/submit1.csv",
encoding='utf-8',index=False,index_col = False,columns=['PassengerId','Survived'])

submit[['PassengerId','Survived']].to_csv("C:/Users/J099055/Desktop/Titanic/titanic_lab5-master/submit_sas_retrait.csv",
encoding='utf-8',index=False,index_col = False,columns=['PassengerId','Survived'])