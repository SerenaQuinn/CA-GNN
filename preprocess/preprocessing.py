import json 
import pandas as pd
import argparse
import pickle
import random
#business_id_set=set()
def parse_args():
    parser = argparse.ArgumentParser(description="Run my code.")
    parser.add_argument('--dataset', nargs='?', default='yelp',
                        help='Choose a dataset: yelp or amazon')
    parser.add_argument('--category',nargs='?', default='health',
                        help='Choose a category')
    return parser.parse_args()
args=parse_args()
if args.dataset=='yelp':
    business_id_list=[]
    #business_name_list=[]
    business_cate_list=[]
    business_city_list=[]
    business_state_list=[]
    business_postal_list=[]
    business_latitude_list=[]
    business_longitude_list=[]
    business_stars_list=[]
    business_reviewcount_list=[]
    business_isopen_list=[]
    with open('../yelp_academic_dataset_business.json','r',encoding='utf-8') as f:
        for jsonstr in f.readlines():
            jsonstr=json.loads(jsonstr)  #jsonstr是dict类型
            #b_id = jsonstr['business_id']
            if not jsonstr['categories']:
                continue

            for cate in jsonstr['categories'].split(','):
                cate=cate.strip()
                if(cate=='Health & Medical'):
                    #business_id_set.add(b_id)
                    business_id_list.append(jsonstr['business_id'])
                    #business_name_list.append(jsonstr['name'])
                    business_cate_list.append(jsonstr['categories'])
                    business_city_list.append(jsonstr['city'])
                    business_state_list.append(jsonstr['state'])
                    business_postal_list.append(jsonstr['postal_code'])
                    business_latitude_list.append(jsonstr['latitude'])
                    business_longitude_list.append(jsonstr['longitude'])
                    business_stars_list.append(jsonstr['stars'])
                    business_reviewcount_list.append(jsonstr['review_count'])
                    business_isopen_list.append(jsonstr['is_open'])
                    break
    business_d={'business_id':business_id_list,'business_cate':business_cate_list,'business_city':business_city_list,
    'business_state':business_state_list,'business_postal':business_postal_list,'business_latitude':business_latitude_list,
    'business_longitude':business_longitude_list,'business_stars':business_stars_list,'business_reviewcount':business_reviewcount_list,
    'business_isopen':business_isopen_list}
    business_df=pd.DataFrame(data=business_d)
    print(business_df.columns.values.tolist())
    user_id_list=[]
    user_reviewcount_list=[]
    user_useful_list=[]
    user_funny_list=[]
    user_cool_list=[]
    user_fans_list=[]
    user_averstars=[]
    user_compli_hot_list=[]
    user_compli_more_list=[]
    user_compli_profile_list=[]
    user_compli_cute_list=[]
    user_compli_list_list=[]
    user_compli_note_list=[]
    user_compli_plain_list=[]
    user_compli_cool_list=[]
    user_compli_funny_list=[]
    user_compli_writer_list=[]
    user_compli_photos_list=[]
    with open('../yelp_academic_dataset_user.json','r',encoding='utf-8')as f:
        for jsonstr in f.readlines():
            jsonstr=json.loads(jsonstr)  #jsonstr是dict类型
            #b_id = jsonstr['business_id']
            user_id_list.append(jsonstr['user_id'])
            user_reviewcount_list.append(jsonstr['review_count'])
            user_useful_list.append(jsonstr['useful'])
            user_funny_list.append(jsonstr['funny'])
            user_cool_list.append(jsonstr['cool'])
            user_fans_list.append(jsonstr['fans'])
            user_averstars.append(jsonstr['average_stars'])
            user_compli_hot_list.append(jsonstr['compliment_hot'])
            user_compli_more_list.append(jsonstr['compliment_more'])
            user_compli_profile_list.append(jsonstr['compliment_profile'])
            user_compli_cute_list.append(jsonstr['compliment_cute'])
            user_compli_list_list.append(jsonstr['compliment_list'])
            user_compli_note_list.append(jsonstr['compliment_note'])
            user_compli_plain_list.append(jsonstr['compliment_plain'])
            user_compli_cool_list.append(jsonstr['compliment_cool'])
            user_compli_funny_list.append(jsonstr['compliment_funny'])
            user_compli_writer_list.append(jsonstr['compliment_writer'])
            user_compli_photos_list.append(jsonstr['compliment_photos'])
    user_d={'user_id':user_id_list,'user_reviewcount':user_reviewcount_list,'user_useful':user_useful_list,
    'user_funny':user_funny_list,'user_cool':user_cool_list,'user_fans':user_fans_list,
    'user_averstars':user_averstars,'user_compli_hot':user_compli_hot_list,'user_compli_more':user_compli_hot_list,
    'user_compli_profile':user_compli_profile_list,'user_compli_cute':user_compli_cute_list,
    'user_compli_list':user_compli_list_list,
    'user_compli_note':user_compli_note_list,'user_compli_plain':user_compli_plain_list,
    'user_compli_cool':user_compli_cool_list,'user_compli_funny':user_compli_funny_list,
    'user_compli_writer':user_compli_writer_list,'user_compli_photos':user_compli_photos_list}
    user_df=pd.DataFrame(user_d)
    review_id_list=[]
    review_user_id_list=[]
    review_business_id_list=[]
    review_stars_list=[]
    review_text_list=[]
    review_useful_list=[]
    review_funny_list=[]
    review_cool_list=[]
    with open('../yelp_academic_dataset_review.json','r',encoding='utf-8')as f:
        for jsonstr in f.readlines():
            jsonstr=json.loads(jsonstr)  #jsonstr是dict类型
            review_id_list.append(jsonstr['review_id'])
            review_user_id_list.append(jsonstr['user_id'])
            review_business_id_list.append(jsonstr['business_id'])
            review_stars_list.append(jsonstr['stars'])
            review_text_list.append(jsonstr['text'])
            review_useful_list.append(jsonstr['useful'])
            review_funny_list.append(jsonstr['funny'])
            review_cool_list.append(jsonstr['cool'])

            
    review_d={'review_id':review_id_list,'user_id':review_user_id_list,'business_id':review_business_id_list,
    'review_stars':review_stars_list,'text':review_text_list,'useful':review_useful_list,'funny':review_funny_list,
    'cool':review_cool_list}
    review_df=pd.DataFrame(review_d)
    business_uni_df=pd.DataFrame({'business_id':business_df['business_id'].tolist()})
    review_df=pd.merge(review_df,business_uni_df,on='business_id')
    print(review_df.columns.values.tolist())
    review_user_uni=list(set(review_df['user_id'].tolist()))
    user_uni_df=pd.DataFrame({'user_id':review_user_uni})
    user_df=pd.merge(user_df,user_uni_df,on='user_id')
    review_total_id=review_df['review_id'].tolist()
    review_no_label_df=review_df[review_df['useful']+review_df['funny']+review_df['cool']<1]
    review_no_label_id=review_no_label_df['review_id'].tolist()
    review_label_id=list(set(review_total_id).difference(set(review_no_label_id)))

    review_label_df=review_df[review_df['useful']+review_df['funny']+review_df['cool']>=1]
    review_helpful_df=review_label_df[review_label_df['useful']>=0.75*(review_label_df['useful']+review_label_df['funny']+review_label_df['cool'])]
    review_helpful_id=review_helpful_df['review_id'].tolist()
    review_unhelpful_id=list(set(review_label_id).difference(set(review_helpful_id)))
    tmp_nolabel_labeldf=pd.DataFrame({'review_id':review_no_label_id,'label':[0]*len(review_no_label_id)})
    tmp_helpful_labeldf=pd.DataFrame({'review_id':review_helpful_id,'label':[1]*len(review_helpful_id)})
    tmp_unhelpful_labeldf=pd.DataFrame({'review_id':review_unhelpful_id,'label':[0]*len(review_unhelpful_id)})
    total_label_df=pd.concat([tmp_nolabel_labeldf,tmp_helpful_labeldf,tmp_unhelpful_labeldf])
    tmp_nolabel_maskdf=pd.DataFrame({'review_id':review_no_label_id,'mask':[3]*len(review_no_label_id)})
    random.shuffle(review_helpful_id)
    random.shuffle(review_unhelpful_id)
    train_helpful_id=review_helpful_id[:int(0.7*len(review_helpful_id))]
    valid_helpful_id=review_helpful_id[int(0.7*len(review_helpful_id)):int(0.8*len(review_helpful_id))]
    test_helpful_id=review_helpful_id[int(0.8*len(review_helpful_id)):]
    train_unhelpful_id=review_unhelpful_id[:int(0.7*len(review_unhelpful_id))]
    valid_unhelpful_id=review_unhelpful_id[int(0.7*len(review_unhelpful_id)):int(0.8*len(review_unhelpful_id))]
    test_unhelpful_id=review_unhelpful_id[int(0.8*len(review_unhelpful_id)):]
    train_id=train_helpful_id+train_unhelpful_id
    valid_id=valid_helpful_id+valid_unhelpful_id
    test_id=test_helpful_id+test_unhelpful_id
    train_df=pd.DataFrame({'review_id':train_id,'mask':[0]*len(train_id)})
    valid_df=pd.DataFrame({'review_id':valid_id,'mask':[1]*len(valid_id)})
    test_df=pd.DataFrame({'review_id':test_id,'mask':[2]*len(test_id)})
    total_mask_df=pd.concat([tmp_nolabel_maskdf,train_df,valid_df,test_df])
    review_df=pd.merge(review_df,total_label_df,on='review_id',how='left')
    review_df=pd.merge(review_df,total_mask_df,on='review_id',how='left')
    
    
    
    
    review_df.to_pickle('gnn_review_df.pkl')
    business_df.to_pickle('gnn_business_df.pkl')
    user_df.to_pickle('gnn_user_df.pkl')
    print(review_df)
    
