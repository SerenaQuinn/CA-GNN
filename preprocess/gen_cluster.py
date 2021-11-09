import pandas as pd 
import numpy as np 
import pickle
from tqdm import tqdm
data_path='../used_data/'
    
CLUSTER_NUM=8
business_idx_df=pd.read_pickle('business_idx_df.pkl')#business_id,business_idx,cluster_id
user_idx_df=pd.read_pickle('user_idx_df.pkl')#user_id,user_idx
review_idx_df=pd.read_pickle('review_idx_df.pkl')#review_id,review_idx,business_id,user_id,mask,label
merged_user=pd.merge(review_idx_df,user_idx_df,on='user_id')
merged_business=pd.merge(merged_user,business_idx_df,on='business_id')
total_idx=merged_business[['review_idx','user_idx','business_idx','cluster_id']]
review_num=review_idx_df.shape[0]
user_num=user_idx_df.shape[0]  #155944
business_num=business_idx_df.shape[0]  #16157
related_user_idx=[-1]*review_num
related_business_idx=[-1]*review_num
user_tree={}
business_tree={}
for idx,row in total_idx.iterrows():
    related_user_idx[row['review_idx']]=row['user_idx']
    related_business_idx[row['review_idx']]=row['business_idx']
    if business_tree.__contains__(row['business_idx']):
        business_tree[row['business_idx']].append(row['review_idx'])
    else:
        business_tree[row['business_idx']]=[row['review_idx']]
    if user_tree.__contains__(row['user_idx']):
        user_tree[row['user_idx']].append(row['review_idx'])
    else:
        user_tree[row['user_idx']]=[row['review_idx']]


#对应于review从0到217243的user_idx
related_user_idx_arr=np.array(related_user_idx)
#对应于review从0到217243的business_idx
related_business_idx_arr=np.array(related_business_idx)

##更新user所用(user更新邻居以后不需要重排序)：
user_segment_id=[]
#对应于user从0到155943的review_idx,business_idx
user_related_review_idx=[]
user_related_business_idx=[]
for idx in tqdm(range(user_num)):
    tmp_user_related_review_idx=user_tree[idx]
    user_segment_id+=[idx]*len(tmp_user_related_review_idx)
    user_related_review_idx+=tmp_user_related_review_idx
    tmp_user_related_business_idx=related_business_idx_arr[tmp_user_related_review_idx].tolist()
    user_related_business_idx+=tmp_user_related_business_idx
with open(data_path+'user_indices.pkl','wb')as f:
    pickle.dump(user_related_review_idx,f)
    pickle.dump(user_related_business_idx,f)
    pickle.dump(user_segment_id,f)
##更新business global neighbor所用(更新邻居后不用重排序)：
global_business_segment_id=[]
global_business_related_review_idx=[]
global_business_related_user_idx=[]
for idx in tqdm(range(business_num)):
    tmp_business_related_review_idx=business_tree[idx]
    global_business_segment_id+=[idx]*len(tmp_business_related_review_idx)
    global_business_related_review_idx+=tmp_business_related_review_idx
    tmp_business_related_user_idx=related_user_idx_arr[tmp_business_related_review_idx].tolist()
    global_business_related_user_idx+=tmp_business_related_user_idx
##更新business所用(business更新邻居后需要重排序)：
business_idx_cluster=[]
business_related_review_idx=[]
business_related_user_idx=[]
business_segment_id=[]
for i in tqdm(range(CLUSTER_NUM)):
    cluster_business_df=business_idx_df[business_idx_df['cluster_id']==i]
    tmp_business_idx_list=cluster_business_df['business_idx'].tolist()
    
    business_idx_cluster.append(tmp_business_idx_list)
    tmp_review_idx_list=[]
    tmp_segment_id=[]
    for idx,business_idx in enumerate(tmp_business_idx_list):
        tmp_review_idx_list+=business_tree[business_idx]
        tmp_segment_id+=[idx]*len(business_tree[business_idx])
    business_related_review_idx.append(tmp_review_idx_list)
    business_related_user_idx.append(related_user_idx_arr[tmp_review_idx_list].tolist())
    business_segment_id.append(tmp_segment_id)

with open(data_path+'business_indices.pkl','wb')as f:
    pickle.dump(business_idx_cluster,f)
    pickle.dump(business_related_review_idx,f)
    pickle.dump(business_related_user_idx,f)
    pickle.dump(business_segment_id,f)
    pickle.dump(global_business_segment_id,f)
    pickle.dump(global_business_related_review_idx,f)
    pickle.dump(global_business_related_user_idx,f)




with open(data_path+'review_indices.pkl','wb')as f:
    pickle.dump(related_user_idx,f)
    pickle.dump(related_business_idx,f)
    


