import pandas as pd
import numpy as np
path="C:/Users/Ju/Desktop/Kaggle/RedHat/"

def intersect(a, b):
    return list(set(a) & set(b))

def get_features(train, test):
    trainval = list(train.columns.values)
    testval = list(test.columns.values)
    output = intersect(trainval, testval)
    output.remove('people_id')
    output.remove('activity_id')
    return sorted(output)

def read_test_train():

    print("Read people.csv...")
    people = pd.read_csv(path+"people.csv",
                       dtype={'people_id': np.str,
                              'activity_id': np.str,
                              'char_38': np.int32},
                       parse_dates=['date'])

    print("Load train.csv...")
    train = pd.read_csv(path+"act_train.csv",
                        dtype={'people_id': np.str,
                               'activity_id': np.str,
                               'outcome': np.int8},
                        parse_dates=['date'])

    print("Load test.csv...")
    test = pd.read_csv(path+"act_test.csv",
                       dtype={'people_id': np.str,
                              'activity_id': np.str},
                       parse_dates=['date'])

    print("Process tables...")
    for table in [train, test]:
        table['year'] = table['date'].dt.year
        table['month'] = table['date'].dt.month
        table['day'] = table['date'].dt.day
        table.drop('date', axis=1, inplace=True)
        table['activity_category'] = table['activity_category'].str.lstrip('type ').astype(np.int32)
        for i in range(1, 11):
            table['char_' + str(i)].fillna('type -999', inplace=True)
            table['char_' + str(i)] = table['char_' + str(i)].str.lstrip('type ').astype(np.int32)

    people['year'] = people['date'].dt.year
    people['month'] = people['date'].dt.month
    people['day'] = people['date'].dt.day
    people.drop('date', axis=1, inplace=True)
    people['group_1'] = people['group_1'].str.lstrip('group ').astype(np.int32)
    for i in range(1, 10):
        people['char_' + str(i)] = people['char_' + str(i)].str.lstrip('type ').astype(np.int32)
    for i in range(10, 38):
        people['char_' + str(i)] = people['char_' + str(i)].astype(np.int32)

    print("Merge...")
    train = pd.merge(train, people, how='left', on='people_id', left_index=True)
    train.fillna(-999, inplace=True)
    test = pd.merge(test, people, how='left', on='people_id', left_index=True)
    test.fillna(-999, inplace=True)

    features = get_features(train, test)
    return train, test, features

train, test, features=read_test_train()
#print(features)

#Features V0
features0=features

train['char_10_x'].replace(to_replace=-999,value=1,inplace=True)
test['char_10_x'].replace(to_replace=-999,value=1,inplace=True)

#test['char_10_x'].value_counts()

#Features
#print(train.dtypes)

#3 first lines
#print(train.head())  

people_id_train=pd.DataFrame(train['people_id']).reset_index(drop=True)
people_id_test=pd.DataFrame(test['people_id']).reset_index(drop=True)

#Exclude features type 'object'
train_num=train.select_dtypes(exclude=['object'])
test_num=test.select_dtypes(exclude=['object'])
train_num.describe()
train_num.reset_index(drop=True,inplace=True)

to_predict=pd.DataFrame(train_num['outcome'],columns=['outcome']).reset_index(drop=True)

#rescale min-max features train & test
for i in features:
    if min(train_num[i])!=min(test_num[i]):
        print("min for i in train",i,min(train_num[i]),"min for i in test",min(test_num[i]))
    
for i in features:
    if max(train_num[i])!=max(test_num[i]):
        print("max for i in train",i,max(train_num[i]),"max for i in test",max(test_num[i]))

#replace min and max
for i in features:
    if min(train_num[i])!=min(test_num[i]):
        test_num[i].replace(to_replace=min(test_num[i]),value=min(train_num[i]),inplace=True)

for i in features:
    if max(train_num[i])!=max(test_num[i]):
        test_num[i].replace(to_replace=max(test_num[i]),value=max(train_num[i]),inplace=True)

#Percentage NA by features
features_na=[]
for i in (features):
    if (train_num[train_num[i]==-999][i].count()/train_num[i].count())*100 > 0:
        percent=(train_num[train_num[i]==-999][i].count()/train_num[i].count())*100
        print("Features:", i, percent, "% of NA")
        if percent>90:
                features_na.append(i)

#unique()  
for i in features:    
    (train_num.groupby(i).size()==1).astype(int)
    if np.std((train_num.groupby(i).size()==1).astype(int))>0:
        print("standard deviation of",i,"binarizer with unique labels",np.std((train_num.groupby(i).size()==1).astype(int)))
        right=pd.DataFrame((train_num.groupby(i).size()==1).astype(int),columns=[i+'_bin'])
        train_num = pd.merge(train_num, right, left_on=i, right_index=True,how='left', sort=False)
        test_num = pd.merge(test_num, right, left_on=i, right_index=True,how='left', sort=False)

test_num.fillna(1, inplace=True)
features=features+list(train_num[train_num.columns[57:]].columns.values)

#Univariate correlation with explivative
def corr_univ(train_num,var,name_corr):
    list_corr=list()
    for i in (features):
        if i!=var:
            corr=np.corrcoef(train_num[i], train_num[var])[1,0]
            list_corr.append(corr)

    feat=pd.DataFrame(features,columns=['feat'])
    corr=pd.DataFrame(list_corr,columns=[name_corr])
    feat_corr=feat.join(corr)
    return feat_corr

feat_corr=corr_univ(train_num,'outcome','corr')

feat_corr_abs=feat_corr.join(pd.DataFrame(list(feat_corr['corr'].abs()),columns=['corr_abs']))
feat_corr_abs.sort(columns=['corr_abs'],ascending=False,inplace=True)

#next step: binarizer + correlation univariate
from sklearn.preprocessing import Binarizer

for i in features:
    if i==features[0]:
        train_num_bin=pd.DataFrame(Binarizer(threshold=train_num[i].mean()).fit_transform(pd.DataFrame(train_num[i])),columns=[i])
    if i!=features[0]:
        train_num_bin=train_num_bin.join(pd.DataFrame(Binarizer(threshold=train_num[i].mean()).fit_transform(pd.DataFrame(train_num[i])),columns=[i]))

feat_corr_bin=corr_univ(train_num_bin.join(to_predict),'outcome','corr_bin')
feat_corr_abs_bin=feat_corr_bin.join(pd.DataFrame(list(feat_corr_bin['corr_bin'].abs()),columns=['corr_abs_bin']))
feat_corr_abs_bin.sort(columns=['corr_abs_bin'],ascending=False,inplace=True)

correlation=pd.merge(feat_corr_abs_bin,feat_corr_abs,on='feat')
correlation['comparaison']=correlation['corr_abs_bin']-correlation['corr_abs']

better_binarizer=list(correlation[correlation['comparaison']>0]['feat'])

#feature engineering: probabilities
def prob_freq(data,data2,var):
    right1=pd.DataFrame(data.groupby(var)['outcome'].mean())
    right1.columns=['class_prob_'+var]
    right2=pd.DataFrame(data.groupby(var)['outcome'].size(),columns=['freq_date_'+var])
    data=pd.merge(data, right1, left_on=var, right_index=True,how='left', sort=False)
    data=pd.merge(data, right2, left_on=var, right_index=True,how='left', sort=False)
    data2=pd.merge(data2, right1, left_on=var, right_index=True,how='left', sort=False)
    data2=pd.merge(data2, right2, left_on=var, right_index=True,how='left', sort=False)
    return data, data2

train_num,test_num=prob_freq(train_num,test_num,'day_x')
train_num,test_num=prob_freq(train_num,test_num,'day_y')
train_num,test_num=prob_freq(train_num,test_num,'month_x')
train_num,test_num=prob_freq(train_num,test_num,'month_y')
train_num,test_num=prob_freq(train_num,test_num,'year_x')
train_num,test_num=prob_freq(train_num,test_num,'year_y')

train_num,test_num=prob_freq(train_num,test_num,'char_38')
train_num,test_num=prob_freq(train_num,test_num,'group_1')
train_num,test_num=prob_freq(train_num,test_num,'char_2_y')

for i in range(61,len(train_num.columns.values)):
    features.append(train_num[[i]].columns.values[0])
    
to_predict=train_num['outcome']

id_test=pd.DataFrame(test['activity_id'])
id_test.reset_index(drop=True,inplace=True)    

def submit(name,pred):
    submit=id_test.join(pred)
    submit.to_csv(path+name+'.csv',index=False,header=['activity_id','outcome'],dtype=int)

train_num.fillna(-999, inplace=True)
test_num.fillna(-999, inplace=True)

#mean, min, max, std by people id
train_num=train_num.join(people_id_train)
test_num=test_num.join(people_id_train)

features_bis=['activity_category','char_10_x','day_x','month_x','year_x']

def min_max_std(data,feat):
    for i in feat:
        print(i)
        a=pd.DataFrame(data.groupby('people_id')[i].min())
        a.columns=[i+'_min']
        b=pd.DataFrame(data.groupby('people_id')[i].max())
        b.columns=[i+'_max']
        c=pd.DataFrame(data.groupby('people_id')[i].std())
        c.columns=[i+'_std']
        data=pd.merge(data, a, left_on='people_id', right_index=True,how='left', sort=False)
        data=pd.merge(data, b, left_on='people_id', right_index=True,how='left', sort=False)
        data=pd.merge(data, c, left_on='people_id', right_index=True,how='left', sort=False) 
    return data

#train_num = train_num.join(people_id_train)
#test_num = test_num.join(people_id_test)

freq_people_tr = pd.DataFrame(train_num.groupby('people_id').size(),columns=['freq_people'])
freq_people_ts = pd.DataFrame(test_num.groupby('people_id').size(),columns=['freq_people'])

train_num=pd.merge(train_num, freq_people_tr, left_on='people_id', right_index=True,how='left', sort=False)
test_num=pd.merge(test_num, freq_people_ts, left_on='people_id', right_index=True,how='left', sort=False)
  
#train_num.drop('peolple_id',axis=1,inplace=True)
#test_num.drop('peolple_id',axis=1,inplace=True)

train_num=min_max_std(train_num,features_bis)
train_num.drop('outcome',axis=1,inplace=True)
#train_num.drop('people_id',axis=1,inplace=True)
features3=list(train_num.drop('people_id',axis=1).columns)
train_num.fillna(-999, inplace=True)

test_num=min_max_std(test_num,features_bis)
#test_num.drop('people_id',axis=1,inplace=True)
test_num.fillna(-999, inplace=True)


#rescale V2 min-max features train & test
for i in features:
    if min(train_num[i])!=min(test_num[i]):
        print("min for", i ,"in train",i,min(train_num[i]),"min for i in test",min(test_num[i]))
    
for i in features:
    if max(train_num[i])!=max(test_num[i]):
        print("max for", i ,"in train",i,max(train_num[i]),"max for i in test",max(test_num[i]))

#replace min and max
#for i in features:
#    if min(train_num[i])!=min(test_num[i]):
#        test_num[i].replace(to_replace=min(test_num[i]),value=min(train_num[i]),inplace=True)

#for i in features:
#    if max(train_num[i])!=max(test_num[i]):
#        test_num[i].replace(to_replace=max(test_num[i]),value=max(train_num[i]),inplace=True)


def sample(train_num,data_key):
    stratify=data_key.join(to_predict).drop_duplicates('people_id')['outcome']
    people_id_dup=data_key.join(to_predict).drop_duplicates('people_id')['people_id']
    people_train, people_test, peopley_train, peopley_test = train_test_split(people_id_dup,stratify,random_state=42,test_size=0.2,stratify=stratify)

    x_train = pd.merge(pd.DataFrame(people_train), train_num.join(to_predict), how='left', on='people_id', left_index=False)
    x_test = pd.merge(pd.DataFrame(people_test), train_num.join(to_predict), how='left', on='people_id', left_index=False)
    y_train = x_train['outcome']
    y_test = x_test['outcome']
    
    x_train2 = x_train.drop_duplicates()
    y_train2 = x_train2['outcome']
    
    x_train.drop('outcome',axis=1,inplace=True)
    x_test.drop('outcome',axis=1,inplace=True)
    return x_train, x_test, y_train, y_test, x_train2, y_train2

#x_train, x_test, y_train, y_test, x_train2, y_train2 = sample(train_num,people_id_train)
