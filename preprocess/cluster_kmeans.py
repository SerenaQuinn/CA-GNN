import pandas as pd
import numpy as np
from nltk.stem.porter import *
import re
from gensim.models import Word2Vec
from tqdm import tqdm
import pickle
import struct
from sklearn.cluster import KMeans

business_df=pd.read_pickle('gnn_business_df.pkl')

cate_dict={}
review_cate=[]
tmp_idx=0
business_cates=business_df['business_cate'].tolist()
business_num=len(business_cates)
for idx,cate in enumerate(business_cates):
    tmp=[]
    for phrase in cate.split(','):
        phrase=phrase.strip()
        if phrase!='Health & Medical':
            if not cate_dict.__contains__(phrase):
                cate_dict[phrase]=tmp_idx
                tmp_idx+=1

            tmp.append(phrase)
    review_cate.append(tmp)
cate_mat=np.zeros(shape=[business_num,tmp_idx])
print(tmp_idx)
for i in range(business_num):
    for cate in review_cate[i]:
        cate_mat[i][cate_dict[cate]]=1
print(cate_mat.shape)
with open('cate_mat_kmeans.pkl','wb')as f:
    pickle.dump(cate_mat,f)
