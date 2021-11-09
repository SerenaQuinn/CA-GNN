import pickle 
from sklearn import preprocessing
import numpy as np 
data_path='../used_data/'
    
with open('business_feature.pkl','rb') as f:
    business_feature=pickle.load(f)
with open('user_feature.pkl','rb') as f:
    user_feature=pickle.load(f)
with open('review_feature.pkl','rb') as f:
    review_feature=pickle.load(f)
with open('review_emb.pkl','rb') as f:
    review_emb=pickle.load(f)

business_feature=preprocessing.scale(business_feature)
user_feature=preprocessing.scale(user_feature)
review_feature=preprocessing.scale(review_feature)
review_feature=np.hstack((review_feature,review_emb))
print(business_feature.shape)
print(user_feature.shape)
print(review_feature.shape)
with open(data_path+'scaled_business_feat.pkl','wb')as f:
    pickle.dump(business_feature,f)
with open(data_path+'scaled_user_feat.pkl','wb') as f:
    pickle.dump(user_feature,f)
with open(data_path+'scaled_review_feat.pkl','wb')as f:
    pickle.dump(review_feature,f)

