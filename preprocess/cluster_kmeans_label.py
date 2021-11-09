import pandas as pd
import numpy as np
from nltk.stem.porter import *
import re
from gensim.models import Word2Vec
from tqdm import tqdm
import pickle
import struct
from sklearn.cluster import KMeans
CLUSTER_NUM=8
with open('cate_mat_kmeans.pkl','rb')as f:
    cate_mat=pickle.load(f)
print(cate_mat.shape)
y_pred=KMeans(n_clusters=CLUSTER_NUM).fit_predict(cate_mat)
with open('kmeans_label.pkl','wb')as f:
    pickle.dump(y_pred,f)
  
result=np.zeros(CLUSTER_NUM)
for i in y_pred:
    result[i]+=1
for i in result:
    print(i)
