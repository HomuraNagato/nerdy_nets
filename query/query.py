from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import pandas as pd
import re
batch_size = 100 
# read json into a dataframe
df_idf = pd.read_json("cleaned_data_1.json",precise_float=True, dtype=False, lines=True, chunksize=batch_size)
 
#for section in df_idf:
 #print schema
 #print("Schema:\n\n",df_idf.dtypes)
 #print("Number of queries,columns=",df_idf.shape)
def pre_process(text):
    #remove tags
    text=re.sub("<!--?.*?-->","",text)
    
    # remove special characters and digits
    text=re.sub("(\\d|\\W)+"," ",text)
    
    return text
def get_stop_words(stop_file_path):
    """load stop words """
    #print("File path:",stop_file_path)
    with open(stop_file_path, 'r') as f:
        stopwords = f.readlines()
        #print("words:",stopwords)
        stop_set = set(m.strip() for m in stopwords)
        #print("Frozen:",frozenset(stop_set))
        return frozenset(stop_set)
stopwords=get_stop_words("stopwords.txt")
#print("Stop words:",stozpwords)
for section in df_idf: 
 section['text'] = section['train'] + section['test']
 section['text'] = section['text'].apply(lambda x:pre_process(x))
 docs=section['text'].tolist()
 cv=CountVectorizer(max_df=0.85,stop_words=stopwords)
 word_count_vector=cv.fit_transform(docs)
 print("List:",list(cv.vocabulary_.keys())[:10])

tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
tfidf_transformer.fit(word_count_vector)

# read test docs into a dataframe 
df_test=pd.read_json("cleaned_test.jsonl",precise_float=True, dtype=False, lines=True, chunksize=batch_size)
for sec_test in df_test:
 sec_test['text'] = sec_test['train'] + sec_test['test']
 sec_test['text'] =sec_test['text'].apply(lambda x:pre_process(x))
 docs_test=sec_test['text'].tolist()

feature_names=cv.get_feature_names()
 
# get the document that we want to extract keywords from
for i in range(len(docs)):
 doc=docs_test[i]
 #print("Doc:",doc) 

def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)
#generate tf-idf for the given document
tf_idf_vector=tfidf_transformer.transform(cv.transform([doc]))
 
#sort the tf-idf vectors by descending order of scores
sorted_items=sort_coo(tf_idf_vector.tocoo())
 
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
 
#extract only the top n; n here is 10
keywords=extract_topn_from_vector(feature_names,sorted_items,10)
 
# now print the results
print("\n=====Doc=====")
print(doc)
print("\n===Keywords===")
for k in keywords:
    print(k,keywords[k])
