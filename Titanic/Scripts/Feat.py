import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
import math

df = pd.read_csv('C:/Users/Ju/Desktop/Titanic/titanic_lab5-master/train.csv', header=0)
test_submit=pd.read_csv('C:/Users/Ju/Desktop/Titanic/titanic_lab5-master/test.csv', header=0)
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
df.describe()

