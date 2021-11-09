import pandas as pd 
import numpy as np 
import pickle
user_df=pd.read_pickle('gnn_user_df.pkl')

user_feature=[]

for idx,row in user_df.iterrows():
    tmp_list=[]
    tmp_list.append(row['user_reviewcount'])
    tmp_list.append(row['user_useful'])
    tmp_list.append(row['user_funny'])
    tmp_list.append(row['user_cool'])
    tmp_list.append(row['user_fans'])
    tmp_list.append(row['user_averstars'])
    tmp_list.append(row['user_compli_hot'])
    tmp_list.append(row['user_compli_more'])
    tmp_list.append(row['user_compli_profile'])
    tmp_list.append(row['user_compli_cute'])
    tmp_list.append(row['user_compli_list'])
    tmp_list.append(row['user_compli_note'])
    tmp_list.append(row['user_compli_plain'])
    tmp_list.append(row['user_compli_cool'])
    tmp_list.append(row['user_compli_funny'])
    tmp_list.append(row['user_compli_writer'])
    tmp_list.append(row['user_compli_photos'])
    user_feature.append(tmp_list)
user_feature=np.array(user_feature,dtype=np.float32)
with open('user_feature.pkl','wb')as f:
    pickle.dump(user_feature,f)
user_num=user_df.shape[0]
user_idx_list=[i for i in range(user_num)]
user_df['user_idx']=user_idx_list
user_df=user_df[['user_id','user_idx']]
user_df.to_pickle('user_idx_df.pkl')
print(user_df)
