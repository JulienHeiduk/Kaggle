import pandas as pd
import numpy as np
import xgboost as xgb
from scipy.stats import skew
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
from sklearn.cross_validation import KFold, StratifiedKFold
#from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Ridge, LassoCV, BayesianRidge, PassiveAggressiveRegressor
from sklearn.linear_model import LinearRegression

#import data
ID = 'id'
TARGET = 'loss'

TRAIN_FILE = "C:/Users/Ju/Desktop/Kaggle/Allsafe/train.csv"
TEST_FILE = "C:/Users/Ju/Desktop/Kaggle/Allsafe/test.csv"
#SUBMISSION_FILE = "../input/sample_submission.csv"

train = pd.read_csv(TRAIN_FILE)
train.drop_duplicates(inplace=True)
test = pd.read_csv(TEST_FILE)

id_test = test.id

y_train = np.log(train[TARGET]).ravel()

train.drop([ID, TARGET], axis=1, inplace=True)
test.drop([ID], axis=1, inplace=True)

ntrain = train.shape[0]
ntest = test.shape[0]
train_test = pd.concat((train, test)).reset_index(drop=True)

features = train.columns

cats = [feat for feat in features if 'cat' in feat]

for cf1 in cats:
    le = LabelEncoder()
    le.fit(train_test[cf1].unique())
    train_test[cf1] = le.transform(train_test[cf1])

numeric_feats = train_test.dtypes[train_test.dtypes != "object"].index
skewed_feats = train_test[numeric_feats].apply(lambda x: skew(x.dropna()))
skewed_feats = skewed_feats[skewed_feats > 0.5]
skewed_feats = skewed_feats.index

train_test[skewed_feats] = np.log1p(train_test[skewed_feats])

list_cont = ['cont14','cont7','cont2']

New_feat = pd.DataFrame(PolynomialFeatures(degree=3,interaction_only=False,include_bias=False).fit_transform(train[list_cont]))
New_feat2 = pd.DataFrame(PolynomialFeatures(degree=3,interaction_only=False,include_bias=False).fit_transform(test[list_cont]))

test2 = train_test.iloc[ntrain:,:].reset_index(drop=True)

train_new = pd.merge(left=train_test.iloc[:ntrain,:], right=New_feat, how='left',left_index=True, right_index=True)
test_new = pd.merge(left=test2, right=New_feat2, how='left',left_index=True, right_index=True)

x_train = np.array(train_new)
x_test = np.array(test_new)

print("Importation OK")