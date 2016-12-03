import matplotlib.pyplot as plt

import numpy as np

from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score
from sklearn.ensemble import  AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier

np.random.seed(0)

# Generate data
X, y = make_blobs(n_samples=100000, n_features=10, random_state=42,cluster_std=10)
X_train, y_train = X[:6000], y[:6000]
X_valid, y_valid = X[6000:8000], y[6000:8000]

X_train_valid, y_train_valid = X[:8000], y[:8000]
X_test, y_test = X[8000:], y[8000:]

# Train uncalibrated random forest classifier on whole train and validation
# data and evaluate on test data
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train_valid, y_train_valid)
clf_rf_probs = clf.predict_proba(X_test)
score_rf = accuracy_score(y_test, clf_rf_probs.argmax(1))

clf_ext = ExtraTreesClassifier(n_estimators=100)
clf_ext.fit(X_train_valid, y_train_valid)
clf_ext_probs = clf_ext.predict_proba(X_test)
score_ext = accuracy_score(y_test, clf_ext_probs.argmax(1))

clf_grad=GradientBoostingClassifier()
clf_grad.fit(X_train_valid, y_train_valid)
clf_grad_probs = clf_grad.predict_proba(X_test)
score_grad = accuracy_score(y_test, clf_grad_probs.argmax(1))

clf_bag=BaggingClassifier()
clf_bag.fit(X_train_valid, y_train_valid)
clf_bag_probs = clf_bag.predict_proba(X_test)
score_bag = accuracy_score(y_test, clf_bag_probs.argmax(1))

clf_ada = AdaBoostClassifier()
clf_ada.fit(X_train_valid, y_train_valid)
clf_ada_probs = clf_ada.predict_proba(X_test)
score_ada = accuracy_score(y_test, clf_ada_probs.argmax(1))

sig_clf = CalibratedClassifierCV(clf, method="sigmoid", cv="prefit")
sig_clf.fit(X_valid, y_valid)
sig_clf_probs = sig_clf.predict_proba(X_test)
sig_score = accuracy_score(y_test, sig_clf_probs.argmax(1))

#for w in zip(range(1,100,1),range(99,0,-1)):
poids_score={'Poids a' : pd.Series(range(1,100)),
'Poids b':pd.Series(range(1,100)),
'Score':pd.Series(range(1,100))}

weight_a=0.5
weight_b=0.5
score_ini=0.0
#model_a=clf_grad_probs
#model_b=clf_ada_probs

modela=[clf_rf_probs,clf_ext_probs,clf_grad_probs,clf_bag_probs,clf_ada_probs,sig_clf_probs]
model_char=['Random Forest','Extra Trees','Gradient Boost','Bagging','AdaBoost','Sigmoid']
Indice=[0,1,2,3,4,5]


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
            if w[0]==1:
                df_score=pd.DataFrame(poids_score)
                #print("Poids du modèle 1")
                #print(df_score['Poids a'][df_score['Score'].argmax()])
                #print("Poids du modèle 2")
                #print(df_score['Poids b'][df_score['Score'].argmax()])
                print(model_char[i], model_char[j], df_score['Poids a'][df_score['Score'].argmax()] , 
                                 df_score['Poids b'][df_score['Score'].argmax()],df_score['Score'][df_score['Score'].argmax()]) 
 #VERIF#############################################################################  
    poids_score={'Poids a' : pd.Series(range(1,100)),
                 'Poids b':pd.Series(range(1,100)),
                 'Score':pd.Series(range(1,100))}
model_a=clf_ext_probs
model_b=clf_bag_probs

        for w in zip(range(1,100,1),range(99,0,-1)):
            score = accuracy_score(y_test,(model_a*w[0]/100.0+model_b*w[1]/100.0).argmax(1))
            poids_score['Poids a'][w[1]-1]=w[0]
            poids_score['Poids b'][w[1]-1]=w[1]
            poids_score['Score'][w[1]-1]=score*1000000000000
            if w[0]==1:
                df_score=pd.DataFrame(poids_score)
                print(df_score['Poids a'][df_score['Score'].argmax()],df_score['Poids b'][df_score['Score'].argmax()],df_score['Score'][df_score['Score'].argmax()]) 


score_ext_bagg = accuracy_score(y_test, (clf_ext_probs*89/100+clf_bag_probs*11/100).argmax(1))
#####################################################################################

df_score=pd.DataFrame(poids_score)
print("Poids du modèle 1")
print(df_score['Poids a'][df_score['Score'].argmax()])
print("Poids du modèle 2")
print(df_score['Poids b'][df_score['Score'].argmax()])
print("Score associé")
print(df_score['Score'][df_score['Score'].argmax()])
print("Score modéle 1")
print(score_grad*1000000000000)
print("Score modele 2")
print(score_ada*1000000000000)
