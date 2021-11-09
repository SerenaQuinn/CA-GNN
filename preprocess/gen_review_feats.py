import pandas as pd 
import numpy as np 
from gensim.models import Doc2Vec
import pickle
from sklearn import preprocessing
data_path='../used_data/'
    
review_df=pd.read_pickle('gnn_review_df.pkl')

review_len=review_df.shape[0]

model=Doc2Vec.load('review_d2v.model')
review_emb_list=[]
for i in range(review_len):
    review_emb_list.append(model.docvecs[str(i)])
review_emb=np.array(review_emb_list)
print(review_emb.shape)
with open('review_emb.pkl','wb')as f:
    pickle.dump(review_emb,f)
review_feature=np.array(review_df['review_stars'].tolist(),dtype=np.float32)
review_feature=review_feature[:,np.newaxis]
with open('review_feature.pkl','wb')as f:
    pickle.dump(review_feature,f)
lb = preprocessing.LabelBinarizer()
enc=preprocessing.OneHotEncoder(sparse=False)
label=review_df['label'].tolist()
mask_list=review_df['mask'].tolist()
idx=0
train_helpful_idx=[]  #58384
train_unhelpful_idx=[]#32010
review_len=len(label)
for i in range(review_len):
    if mask_list[i]==0:
        if label[i]==1:
            train_helpful_idx.append(idx)
        else:
            train_unhelpful_idx.append(idx)
        idx+=1
helpful_count=0
unhelpful_count=0
for i in range(review_len):
    if mask_list[i]!=3:
        if label[i]==1:
            helpful_count+=1
        else:
            unhelpful_count+=1
print("helpful_rate: ",helpful_count/217244)
print("unhelpful_rate: ",unhelpful_count/217244)

with open(data_path+'train_idx.pkl','wb')as f:
    pickle.dump(train_helpful_idx,f)
    pickle.dump(train_unhelpful_idx,f)

label = lb.fit_transform(label)  
label=enc.fit_transform(label).astype(np.float32)
with open(data_path+'review_label.pkl','wb')as f:
    pickle.dump(label,f)
with open(data_path+'review_mask.pkl','wb')as f:
    pickle.dump(mask_list,f)
review_idx_list=[i for i in range(review_len)]
review_df['review_idx']=review_idx_list
review_idx_df=review_df[['review_id','review_idx','business_id','user_id','mask','label']]
review_idx_df.to_pickle('review_idx_df.pkl')
print(review_idx_df.columns.values.tolist())