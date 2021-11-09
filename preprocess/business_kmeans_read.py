import numpy as np 
import struct 
import pandas as pd
import networkx as nx 
import pickle
CLUSTER_NUM=8
business_df=pd.read_pickle('gnn_business_df.pkl')
business_num=business_df.shape[0]


with open('kmeans_label.pkl','rb')as f:
    y_pred=pickle.load(f)
result=np.zeros(CLUSTER_NUM)
for i in y_pred:
    result[i]+=1
for i in result:
    print(i)

business_df['cluster_id']=y_pred
business_num=business_df.shape[0]
business_idx_list=[i for i in range(business_num)]
business_df['business_idx']=business_idx_list

business_df=business_df[['business_id','business_idx','cluster_id']]
#print(business_df.columns.values.tolist())
business_df.to_pickle('business_idx_df.pkl')
print(business_df)

