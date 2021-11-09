import pandas as pd
import numpy as np
from nltk.stem.porter import *
import re
from gensim.models import Word2Vec
from tqdm import tqdm
import pickle
import struct
data_path='../used_data/'
    
def stem_tokens(tokens, stemmer):
    """
    进行对已经分完词进行词干抽取
    """
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed
stemmer = PorterStemmer()

business_df=pd.read_pickle('gnn_business_df.pkl')


business_cates=business_df['business_cate'].tolist()
stem_words=[]
splitter = re.compile('[^a-zA-Z0-9]')
for idx,cate in enumerate(business_cates):
    #print(idx)
    words=[]
    #print(cate)
    for phrase in cate.split(','):
        phrase=phrase.strip()
        if phrase!='Health & Medical':
            for curr_word in splitter.split(phrase):
                curr_word=curr_word.strip().lower()
                if curr_word !='':
                    words.append(curr_word)
    stemmed = stem_tokens(words,stemmer)
    stem_words.append(stemmed)
    

model = Word2Vec(stem_words, min_count=1,workers=3, size=128)
cate_vec=[]
delete_row=[]
for idx,query in enumerate(tqdm(stem_words)):
    if len(query):
        x=model[query]
        x=np.mean(x,axis=0)
        alg=np.linalg.norm(x)
        x=x/alg
        cate_vec.append(x)

    else:
        delete_row.append(idx)

cate_vec=np.array(cate_vec)
dim=cate_vec.shape[1]  #128
num=cate_vec.shape[0]  #16117
with open('save_utils.pkl','wb')as f:
    pickle.dump(num,f)
    pickle.dump(delete_row,f)
outfile=open('business_cate_vec',mode='wb')
for i in range(num):
    elem=struct.pack('<i',dim)
    outfile.write(elem)
    for j in range(dim):
        elem=struct.pack('<f',cate_vec[i][j])
        outfile.write(elem)
outfile.close()
print(cate_vec)





business_num=business_df.shape[0]
business_idx_list=[i for i in range(business_num)]
business_df['business_idx']=business_idx_list


business_city_list=[]  #285
business_state_list=[] #19
business_postal_list=[] #2078
business_feature=[]
for idx,row in business_df.iterrows():
    if row['business_city'] not in business_city_list:
        business_city_list.append(row['business_city'])
    if row['business_state'] not in business_state_list:
        business_state_list.append(row['business_state'])
    if row['business_postal'] not in business_postal_list:
        business_postal_list.append(row['business_postal'])
    tmp_list=[]
    tmp_list.append(row['business_latitude'])
    tmp_list.append(row['business_longitude'])
    tmp_list.append(row['business_stars'])
    tmp_list.append(row['business_reviewcount'])
    tmp_list.append(row['business_isopen'])
    business_feature.append(tmp_list)

city_num=len(business_city_list)
state_num=len(business_state_list)
postal_num=len(business_postal_list)
with open(data_path+'business_cat_feature_num.pkl','wb')as f:
    pickle.dump(city_num,f)
    pickle.dump(state_num,f)
    pickle.dump(postal_num,f)

#num类
business_feature=np.array(business_feature,dtype=np.float32)
with open('business_feature.pkl','wb')as f:
    pickle.dump(business_feature,f)
    

business_city_idx=[i for i in range(city_num)]
business_state_idx=[i for i in range(state_num)]
business_postal_idx=[i for i in range(postal_num)]
city_df=pd.DataFrame({'business_city':business_city_list,'business_city_idx':business_city_idx})
state_df=pd.DataFrame({'business_state':business_state_list,'business_state_idx':business_state_idx})
postal_df=pd.DataFrame({'business_postal':business_postal_list,'business_postal_idx':business_postal_idx})
business_df=pd.merge(business_df,city_df,on='business_city',how='left')
business_df=pd.merge(business_df,state_df,on='business_state',how='left')
business_df=pd.merge(business_df,postal_df,on='business_postal',how='left')
#cat类
city_idx_list=business_df['business_city_idx'].tolist()
state_idx_list=business_df['business_state_idx'].tolist()
postal_idx_list=business_df['business_postal_idx'].tolist()
with open(data_path+'business_cate_feature_idx.pkl','wb')as f:
    pickle.dump(city_idx_list,f)
    pickle.dump(state_idx_list,f)
    pickle.dump(postal_idx_list,f)

