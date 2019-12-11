from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.corpus import stopwords
import pandas as pd
import re

 
df_idf=pd.read_json("file_1.json",lines=True)     #54MB file after splitting the 3GB preprocessed file
 
# print schema
print("Schema:\n\n",df_idf.dtypes)
print("Number of queries,columns=",df_idf.shape)
#print("Text:",df_idf)
def pre_process(text):
    
    
    #remove tags
    text=re.sub("<!--?.*?-->","",text)
    
    # remove special characters and digits
    text=re.sub("(\\d|\\W)+"," ",text)
    
    return text


df_idf['text'] = df_idf['train'] + df_idf['test']
df_idf['text'] = df_idf['text'].apply(lambda x:pre_process(x))
#df_idf['text'][1]
 
print("Text:",df_idf['text'][1])

def get_stop_words(stop_file_path):
    
    with open(stop_file_path, 'r', encoding="utf-8") as f:
        stop_words = f.readlines()
        stop_set = set(m.strip() for m in stop_words)
        return frozenset(stop_set)
 
#load a set of stop words
stop_words=stopwords.words('english')
 
docs=df_idf['text'].tolist()
 
#create a vocabulary of words
cv=CountVectorizer(max_df=0.85,sw=stop_words,max_features=10000)
word_count_vector=cv.fit_transform(docs)
list(cv.vocabulary_.keys())[:10]
print("List:",list(cv.vocabulary_.keys())[:10])

tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
tfidf_transformer.fit(word_count_vector)

df_test=pd.read_json("cleaned_data_1.json",lines=True)    #this file will be changed
df_test['text'] = df_test['train'] + df_test['test']
df_test['text'] =df_test['text'].apply(lambda x:pre_process(x))
 
# get test docs into a list
docs_test=df_test['text'].tolist()

 you only needs to do this once, this is a mapping of index to 
feature_names=cv.get_feature_names()
 
# get the document that we want to extract keywords from
doc=docs_test[0]
 
#generate tf-idf for the given document
tf_idf_vector=tfidf_transformer.transform(cv.transform([doc]))
 
#sort the tf-idf vectors by descending order of scores
sorted_items=sort_keywords(tf_idf_vector.tocoo())
 
#extract only the top n; n here is 10
keywords=extract_topn_from_vector(feature_names,sorted_items,10)
 
print("\n Doc:")
print(doc)
print("\n Keywords:")
for k in keywords:
    print(k,keywords[k])
 
def sort_keywords(kw_matrix):
    tuples = zip(kw_matrix.col, kw_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)
 
def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    """get the feature names and tf-idf score of top n items"""
    
    #use only topn items from vector
    sorted_items = sorted_items[:topn]
 
    score_vals = []
    feature_vals = []
    
    # word index and corresponding tf-idf score
    for idx, score in sorted_items:
        
        #keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])
 
    #create a tuples of feature,score
    #results = zip(feature_vals,score_vals)
    results= {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]]=score_vals[idx]
    
    return results 
