
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