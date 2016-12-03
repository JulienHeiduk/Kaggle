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
import xgboost

def modele(clf,tr,y_tr,ts,y_ts):
    clf.fit(tr, y_tr)
    pred=clf.predict(ts)
    clf_score = accuracy_score(y_ts, pred)
    #↓clf_score2=zero_one_loss(y_ts, pred)
    print(clf_score)
    print(clf_score2)
    print(confusion_matrix(y_ts, pred))
    global pred_final1
    pred_final1=clf.predict(test_submit)  
    #print(clf.feature_importances_)

pred_final1 = pd.DataFrame()
modele(RandomForestClassifier(n_estimators=1000,max_depth=5,criterion='entropy',max_features=6,
random_state=42),x_train,y_train,x_test,y_test)

modele(AdaBoostClassifier(RandomForestClassifier(n_estimators=1000,max_depth=5,criterion='entropy',max_features=6,
random_state=42),learning_rate=0.01),x_train,y_train,x_test,y_test)

modele(BaggingClassifier(RandomForestClassifier(n_estimators=1000,max_depth=5,criterion='entropy',max_features=6,
random_state=42),bootstrap_features=True),x_train,y_train,x_test,y_test)

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

modele(xgboost.XGBClassifier(n_estimators=750,learning_rate=0.01,max_depth=4)
,x_train,y_train,x_test,y_test)

import os
import sys
my_dir = os.getcwd()
sys.path.append(my_dir+'/Desktop/Titanic')
import gen_clf

#Regression logistique sur les variables categoeilles pour les transformer en meta feature
#Puis une autre méthode par au dessus pour finir la prédiction
clf1 = LogisticRegression(C=5,verbose=1, n_jobs=-1)
my_clf = gen_clf.my_clf(num_class=2, n_fold=35, seed=0)
meta_feat1 = my_clf.predict(clf1, train, targets_tr, x_test, 'base') 
meta_feat1_1 = np.reshape(np.apply_along_axis(np.argmax, 1, meta_feat1), (-1, 1))





