
#train=train.merge(right1, left_on='Breed', right_index=True,how='left').merge(right2, left_on='Breed',right_index=True,how='left').merge(right3,left_on='Breed', right_index=True,how='left').merge(right4,left_on='Breed',right_index=True,how='left').merge(right5,left_on='Breed', right_index=True,how='left').merge(right6, left_on='Breed',right_index=True,how='left').merge(right7,left_on='Breed', right_index=True,how='left').merge(right8, left_on='Breed',right_index=True,how='left').merge(right9,left_on='Breed', right_index=True,how='left').merge(right10, left_on='Breed',right_index=True,how='left').merge(right11,left_on='Breed', right_index=True,how='left').merge(right12, left_on='Breed',right_index=True,how='left').merge(right13,left_on='Breed', right_index=True,how='left').merge(right14, left_on='Breed',right_index=True,how='left').merge(right15,left_on='Breed', right_index=True,how='left').merge(right16, left_on='Breed',right_index=True,how='left')
print('ok')

#train.drop(['Aggressive_x','At Vet_x','Barn_x','Behavior_x','Court/Investigation_x','Enroute_x','Foster_x','In Foster_x',
#            'In Kennel_x','In Surgery_x','Medical_x','Offsite_x','Partner_x','Rabies Risk_x','SCRP_x','Suffering_x'],axis=1,inplace=True)
print('ok')

from nltk.corpus.reader.wordnet import NOUN
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import re
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer

#train['Breed_vec'] = [''.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line)) for line in lists]).strip() for lists in train['Breed11']]
#train['Breed_vec2'] = [''.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line)) for line in lists]).strip() for lists in train['Breed2']]
#train['Color_vec'] = [''.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line)) for line in lists]).strip() for lists in train['Color']]
#train['Color1_vec'] = [''.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line)) for line in lists]).strip() for lists in train['Color1']]
#train['Color2_vec'] = [''.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line)) for line in lists]).strip() for lists in train['Color2']]
#train['Shade1_vec'] = [''.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line)) for line in lists]).strip() for lists in train['Shade1']]
#train['Shade2_vec'] = [''.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line)) for line in lists]).strip() for lists in train['Shade2']]

#corpustr_br = train['Breed_vec']
#corpustr_br2 = train['Breed_vec2']
#corpustr_col = train['Color_vec']
#corpustr_col1 = train['Color1_vec']
#corpustr_col2 = train['Color2_vec']
#corpustr_sh1 = train['Shade1_vec']
#corpustr_sh2 = train['Shade2_vec']

vectorizertr = CountVectorizer(stop_words='english',
                             ngram_range = ( 1 , 1 ),analyzer="word", 
                             binary=True , token_pattern=r'\w+' )

#tfidftr_br=vectorizertr.fit_transform(corpustr_br).todense()
#tfidftr_br2=vectorizertr.fit_transform(corpustr_br2).todense()
#tfidftr_col=vectorizertr.fit_transform(corpustr_col).todense()
#tfidftr_col1=vectorizertr.fit_transform(corpustr_col1).todense()
#tfidftr_col2=vectorizertr.fit_transform(corpustr_col2).todense()
#tfidftr_sh1=vectorizertr.fit_transform(corpustr_sh1).todense()
#tfidftr_sh2=vectorizertr.fit_transform(corpustr_sh2).todense()

#dt1=pd.DataFrame(tfidftr_br,dtype=float).add_suffix('_br1')
#dt2=pd.DataFrame(tfidftr_col,dtype=float).add_suffix('_col')
#dt3=pd.DataFrame(tfidftr_col1,dtype=float).add_suffix('_col1')
#dt4=pd.DataFrame(tfidftr_col2,dtype=float).add_suffix('_col2')
#dt5=pd.DataFrame(tfidftr_sh1,dtype=float).add_suffix('_sh1')
#dt6=pd.DataFrame(tfidftr_sh2,dtype=float).add_suffix('_sh2')
#dt7=pd.DataFrame(tfidftr_br2,dtype=float).add_suffix('_br2')

#train2=train.join(dt1).join(dt2)#.join(dt3).join(dt4).join(dt5).join(dt6).join(dt7)
train2=train
print('ok')
train2.shape