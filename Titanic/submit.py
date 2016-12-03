test_submit_fin=pd.DataFrame(pd.DataFrame(PassengerId).join(pred_final),columns=['PassengerId','Survived'])

test_1=pd.DataFrame(pd.DataFrame(PassengerId).join(pd.DataFrame(pred_final1,columns=['Survived'])),
                    columns=['PassengerId','Survived'])
test_submit_fin[['PassengerId','Survived']].to_csv("C:/Users/J099055/Desktop/Titanic/titanic_lab5-master/submit.csv",
encoding='utf-8',index=False,index_col = False,columns=['PassengerId','Survived'])

test_1[['PassengerId','Survived']].to_csv("C:/Users/J099055/Desktop/Titanic/titanic_lab5-master/submit1.csv",
encoding='utf-8',index=False,index_col = False,columns=['PassengerId','Survived'])

submit[['PassengerId','Survived']].to_csv("C:/Users/J099055/Desktop/Titanic/titanic_lab5-master/submit_sas_retrait.csv",
encoding='utf-8',index=False,index_col = False,columns=['PassengerId','Survived'])
