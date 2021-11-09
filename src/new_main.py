from __future__ import division
from __future__ import print_function
from time import strftime, localtime
import tensorflow as tf
import argparse
import numpy as np
import pickle
from new_models import BaseModel
import os
import random
from sklearn.metrics import roc_auc_score,f1_score
# Set random seed
seed = 2020
#random.seed(seed)
np.random.seed(seed)
tf.set_random_seed(seed)
CITY_EMB_SIZE=4
STATE_EMB_SIZE=4
POSTAL_STATE_SIZE=4



IF_MULTI_TASK=True
IF_GNN=False



REVIEW_HIDDEN_DIM=[17,64,64]
BUSINESS_HIDDEN_DIM=[8,20]
USER_HIDDEN_DIM=[8,20]
BUSINESS_NEIGHBOR_HIDDEN_DIM=[56,44]
USER_NEIGHBOR_HIDDEN_DIM=[56,44]

def parse_args():
    parser = argparse.ArgumentParser(description="Run the DEMO-Net.")
    parser.add_argument('--dataset', nargs='?', default='brazil',
                        help='Choose a dataset: brazil, europe or usa')
    parser.add_argument('--epochs', type=int, default=400,
                        help='Number of epochs.')
    parser.add_argument('--dropout', type=int, default=0.05,
                        help='dropout rate (1 - keep probability).')
    parser.add_argument('--patience', type=int, default=1000,
                        help='patience to update the parameters.')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='Learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0.0005,
                        help='weight for l2 loss on embedding matrix')
    parser.add_argument('--n_layers', type=int, default=3,
                        help='Number of hidden layers')
    return parser.parse_args()

def load_data():
    data_path='../used_data/'
    #int
    with open(data_path+'business_cat_feature_num.pkl','rb')as f:
        business_city_num=pickle.load(f)
        business_state_num=pickle.load(f)
        business_postal_num=pickle.load(f)
    #ndarray
    with open(data_path+'scaled_business_feat.pkl','rb')as f:
        business_feat=pickle.load(f)
    with open(data_path+'scaled_user_feat.pkl','rb')as f:
        user_feat=pickle.load(f)
    with open(data_path+'scaled_review_feat.pkl','rb')as f:
        review_feat=pickle.load(f)
    with open(data_path+'review_label.pkl','rb')as f:
        label=pickle.load(f) #(217244,2)
    #list
    with open(data_path+'business_cate_feature_idx.pkl','rb')as f:
        business_city_idx=pickle.load(f)
        business_state_idx=pickle.load(f)
        business_postal_idx=pickle.load(f)
    with open(data_path+'review_mask.pkl','rb')as f:
        mask=pickle.load(f)  #217244
    with open(data_path+'train_idx.pkl','rb')as f:
        train_helpful_idx=pickle.load(f)  #58384
        train_unhelpful_idx=pickle.load(f) #32010
    with open(data_path+'user_indices.pkl','rb')as f:
        user_related_review_idx=pickle.load(f)   #长度都是217244
        user_related_business_idx=pickle.load(f)
        user_segment_id=pickle.load(f)
    with open(data_path+'business_indices.pkl','rb')as f:
        business_idx_cluster=pickle.load(f)
        business_related_review_idx=pickle.load(f)
        business_related_user_idx=pickle.load(f)
        business_segment_id=pickle.load(f)
        global_business_segment_id=pickle.load(f)
        global_business_related_review_idx=pickle.load(f)
        global_business_related_user_idx=pickle.load(f)
    with open(data_path+'review_indices.pkl','rb')as f:
        related_user_idx=pickle.load(f)
        related_business_idx=pickle.load(f)
    
    return business_feat,user_feat,review_feat,label,mask,\
    train_helpful_idx,train_unhelpful_idx,\
    business_city_idx,business_state_idx,business_postal_idx,\
    business_city_num,business_state_num,business_postal_num,\
    user_related_review_idx,user_related_business_idx,user_segment_id,\
    business_idx_cluster,business_related_review_idx,business_related_user_idx,business_segment_id,\
    global_business_segment_id,global_business_related_review_idx,global_business_related_user_idx,\
    related_user_idx,related_business_idx





def construct_placeholder(review_len,business_len,user_len,\
        review_feat_size,business_feat_size, user_feat_size, num_classes):
    with tf.name_scope('input'):
        placeholders = {
            'label': tf.placeholder(tf.float32, shape=(review_len, num_classes), name='label'),
            'mask': tf.placeholder(tf.int32,shape=(review_len,),name='mask'),
            'business_feat': tf.placeholder(tf.float32,shape=(business_len,business_feat_size),name='business_feat'),
            'user_feat': tf.placeholder(tf.float32,shape=(user_len,user_feat_size),name='user_feat'),
            'review_feat':tf.placeholder(tf.float32,shape=(review_len,review_feat_size),name='review_feat'),
            'train_shuffled_idx':tf.placeholder(tf.int32,shape=None,name='train_shuffled_idx'),
            'business_city_idx':tf.placeholder(tf.int32,shape=(business_len,),name='business_city_idx'),
            'business_state_idx':tf.placeholder(tf.int32,shape=(business_len,),name='business_state_idx'),
            'business_postal_idx':tf.placeholder(tf.int32,shape=(business_len,),name='business_postal_idx'),
            'dropout': tf.placeholder_with_default(0., shape=(), name='dropout'),
            
        }
        return placeholders


def train(args, data):
    business_feat,user_feat,review_feat,label,mask,\
    train_helpful_idx,train_unhelpful_idx,\
    business_city_idx,business_state_idx,business_postal_idx,\
    business_city_num,business_state_num,business_postal_num,\
    user_related_review_idx,user_related_business_idx,user_segment_id,\
    business_idx_cluster,business_related_review_idx,business_related_user_idx,business_segment_id,\
    global_business_segment_id,global_business_related_review_idx,global_business_related_user_idx,\
    related_user_idx,related_business_idx =load_data()

    
    business_feat_size=business_feat.shape[1]  #5
    user_feat_size=user_feat.shape[1]          #17
    review_feat_size=review_feat.shape[1]      #301
    num_classes=label.shape[1]                 #2
    business_len=business_feat.shape[0]        #16157
    user_len=user_feat.shape[0]                #155944
    review_len=review_feat.shape[0]            #217244
    
    placeholders = construct_placeholder(review_len,business_len,user_len,\
        review_feat_size,business_feat_size, user_feat_size, num_classes)

    model = BaseModel(placeholders, num_classes, user_related_review_idx,user_related_business_idx,user_segment_id,\
        business_idx_cluster,business_related_review_idx,business_related_user_idx,business_segment_id,\
        global_business_segment_id,global_business_related_review_idx,global_business_related_user_idx,\
        related_user_idx,related_business_idx,\
        business_city_num,business_state_num,business_postal_num,\
        city_emb_size=CITY_EMB_SIZE,state_emb_size=STATE_EMB_SIZE,postal_emb_size=POSTAL_STATE_SIZE,\
        review_hidden_dim=REVIEW_HIDDEN_DIM,business_hidden_dim=BUSINESS_HIDDEN_DIM,user_hidden_dim=USER_HIDDEN_DIM, \
        business_neighbor_hidden_dim=BUSINESS_NEIGHBOR_HIDDEN_DIM,user_neighbor_hidden_dim=USER_NEIGHBOR_HIDDEN_DIM,\
        num_layers=args.n_layers)
    


    if IF_MULTI_TASK:
        logits=model.xiaomao_inference()#business multi-task, review single network
        
    elif IF_GNN:
        logits=model.xiaomao_single_inference()
    
    segmented_logits=tf.dynamic_partition(logits,placeholders['mask'],4)
    train_logits=segmented_logits[0]
    valid_logits=segmented_logits[1]
    test_logits=segmented_logits[2]
    selected_train_logits=tf.gather(train_logits,placeholders['train_shuffled_idx'])
    segmented_labels=tf.dynamic_partition(placeholders['label'],placeholders['mask'],4)
    train_labels=segmented_labels[0]
    valid_labels=segmented_labels[1]
    test_labels=segmented_labels[2]
    selected_train_labels=tf.gather(train_labels,placeholders['train_shuffled_idx'])
    train_loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=selected_train_labels,logits=selected_train_logits))
    valid_loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=valid_labels,logits=valid_logits))
    train_correct_prediction = tf.equal(tf.argmax(selected_train_logits, 1), tf.argmax(selected_train_labels, 1))
    train_accuracy=tf.reduce_mean(tf.cast(train_correct_prediction, tf.float32))
    valid_correct_prediction = tf.equal(tf.argmax(valid_logits, 1), tf.argmax(valid_labels, 1))
    valid_accuracy=tf.reduce_mean(tf.cast(valid_correct_prediction, tf.float32))
    
    
    train_op = model.training(train_loss, lr=args.lr, l2_coef=args.weight_decay)
    #train_op = model.training(train_loss,lr=args.lr,weight_decay=args.weight_decay)
    
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer(), tf.tables_initializer())
    
    # vloss_min = np.inf
    # vacc_max = 0.0
    # curr_step = 0
    saver=tf.train.Saver()
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    
    train_shuffled_idx=[]
    help_len=len(train_helpful_idx)
    unhelp_len=len(train_unhelpful_idx)
    if help_len>unhelp_len:
        train_shuffled_idx+=train_unhelpful_idx
        random.seed(seed)
        train_shuffled_idx+=random.sample(train_helpful_idx,unhelp_len)
    else:
        train_shuffled_idx+=train_helpful_idx
        random.seed(seed)
        train_shuffled_idx+=random.sample(train_unhelpful_idx,help_len)
    random.seed(seed)
    random.shuffle(train_shuffled_idx)
    
            
    
    with tf.Session(config=config) as sess:
        sess.run(init_op)
        # vacc_early_model = 0.0
        # vlss_early_model = 0.0
        max_auc=0
        curr_step=0
        step=0
        for epoch in range(args.epochs):
            print("Epoch: ",epoch)
            step+=1
            cluster_feed_dict={}
            cluster_feed_dict.update({placeholders['label']:label})
            cluster_feed_dict.update({placeholders['mask']:mask})
            cluster_feed_dict.update({placeholders['business_feat']:business_feat})
            cluster_feed_dict.update({placeholders['user_feat']:user_feat})
            cluster_feed_dict.update({placeholders['review_feat']:review_feat})
            cluster_feed_dict.update({placeholders['train_shuffled_idx']:train_shuffled_idx})
            cluster_feed_dict.update({placeholders['business_city_idx']:business_city_idx})
            cluster_feed_dict.update({placeholders['business_state_idx']:business_state_idx})
            cluster_feed_dict.update({placeholders['business_postal_idx']:business_postal_idx})
            cluster_feed_dict.update({placeholders['dropout']:args.dropout})
            _,tmp_train_loss,tmp_train_acc=sess.run([train_op,train_loss,train_accuracy],feed_dict=cluster_feed_dict)
            cluster_feed_dict.update({placeholders['dropout']:0})
            tmp_valid_loss,tmp_valid_acc,tmp_valid_logits,tmp_valid_labels=sess.run([valid_loss,valid_accuracy,valid_logits,valid_labels],feed_dict=cluster_feed_dict)
            valid_auc=roc_auc_score(tmp_valid_labels,tmp_valid_logits)
            print("Epoch: ",epoch," train_loss: ",tmp_train_loss," valid_loss: ",tmp_valid_loss," train_acc: ",tmp_train_acc," valid_acc: ",tmp_valid_acc," valid_auc: ",valid_auc)
            if valid_auc>max_auc:
                max_auc=valid_auc
                curr_step=0
                save_path=saver.save(sess,'./checkpoint/cluster',global_step=step)
                print(save_path)
            else:
                curr_step+=1
                if curr_step==args.patience:
                    print("Early stop! Max AUC: ",max_auc,"Epoch: ",epoch)
                    break
    with tf.Session(config=config) as sess:
        saver.restore(sess,save_path)
        cluster_feed_dict={}
        cluster_feed_dict.update({placeholders['label']:label})
        cluster_feed_dict.update({placeholders['mask']:mask})
        cluster_feed_dict.update({placeholders['business_feat']:business_feat})
        cluster_feed_dict.update({placeholders['user_feat']:user_feat})
        cluster_feed_dict.update({placeholders['review_feat']:review_feat})
        cluster_feed_dict.update({placeholders['train_shuffled_idx']:train_shuffled_idx})
        cluster_feed_dict.update({placeholders['business_city_idx']:business_city_idx})
        cluster_feed_dict.update({placeholders['business_state_idx']:business_state_idx})
        cluster_feed_dict.update({placeholders['business_postal_idx']:business_postal_idx})
        cluster_feed_dict.update({placeholders['dropout']:0})
        tmp_test_logits,tmp_test_labels=sess.run([test_logits,test_labels],feed_dict=cluster_feed_dict)
        test_auc=roc_auc_score(tmp_test_labels,tmp_test_logits)
        print("AUC: ",test_auc)
        y_true=np.argmax(tmp_test_labels,1)
        y_pred=np.argmax(tmp_test_logits,1)
        test_f1=f1_score(y_true,y_pred)
        print("F1: ",test_f1)
        
                

        

if __name__ == '__main__':
    time_stamp = strftime('%Y_%m_%d_%H_%M_%S', localtime())
    print("The time of running the codes: ", time_stamp)
    args = parse_args()
    data = load_data()
    train(args, data)
