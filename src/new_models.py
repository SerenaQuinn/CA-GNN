import tensorflow as tf
import numpy as np
from tqdm import tqdm


class BaseModel(object):
    def __init__(self, placeholders, num_class, user_related_review_idx,user_related_business_idx,user_segment_id,\
        business_idx_cluster,business_related_review_idx,business_related_user_idx,business_segment_id,\
        global_business_segment_id,global_business_related_review_idx,global_business_related_user_idx,\
        related_user_idx,related_business_idx,\
        business_city_num,business_state_num,business_postal_num,\
        city_emb_size,state_emb_size,postal_emb_size,\
        review_hidden_dim, business_hidden_dim,user_hidden_dim,\
        business_neighbor_hidden_dim,user_neighbor_hidden_dim,num_layers, activation=tf.nn.relu):
        
        self.label=placeholders['label']
        self.mask=placeholders['mask']
        self.business_feat=placeholders['business_feat']
        self.user_feat=placeholders['user_feat']
        self.review_feat=placeholders['review_feat']
        self.train_shuffled_idx=placeholders['train_shuffled_idx']
        self.business_city_idx=placeholders['business_city_idx']
        self.business_state_idx=placeholders['business_state_idx']
        self.business_postal_idx=placeholders['business_postal_idx']
        self.business_city_num=business_city_num
        self.business_state_num=business_state_num
        self.business_postal_num=business_postal_num
        self.dropout = placeholders['dropout']

        self.num_class=num_class
        self.review_hidden_dim=review_hidden_dim
        self.business_hidden_dim=business_hidden_dim
        self.user_hidden_dim=user_hidden_dim
        self.business_neighbor_hidden_dim=business_neighbor_hidden_dim
        self.user_neighbor_hidden_dim=user_neighbor_hidden_dim
        
        self.num_layers=num_layers
        self.act=activation

        self.user_related_review_idx=user_related_review_idx
        self.user_related_business_idx=user_related_business_idx
        self.user_segment_id=user_segment_id
        self.business_idx_cluster=business_idx_cluster
        self.business_related_review_idx=business_related_review_idx
        self.business_related_user_idx=business_related_user_idx
        self.business_segment_id=business_segment_id
        self.global_business_segment_id=global_business_segment_id
        self.global_business_related_review_idx=global_business_related_review_idx
        self.global_business_related_user_idx=global_business_related_user_idx
        
        
        self.related_user_idx=related_user_idx
        self.related_business_idx=related_business_idx
        self.city_embedding_table=tf.get_variable("city_embedding_table",shape=[self.business_city_num,city_emb_size],initializer=tf.contrib.layers.xavier_initializer())
        self.state_embedding_table=tf.get_variable("state_embedding_table",shape=[self.business_state_num,state_emb_size],initializer=tf.contrib.layers.xavier_initializer())
        self.postal_embedding_table=tf.get_variable("postal_embedding_table",shape=[self.business_postal_num,postal_emb_size],initializer=tf.contrib.layers.xavier_initializer())
        
        self.business_num=self.business_feat.shape[0]
        
    
    
    def xiaomao_user_forward(self,id_layer,business_inputs,user_inputs,review_inputs,user_out_sz,user_neighbor_out_sz,act=tf.nn.elu):
        with tf.name_scope('xiaomao_user_layer' + str(id_layer)):
            user_related_review=tf.gather(review_inputs,self.user_related_review_idx)
            user_related_business=tf.gather(business_inputs,self.user_related_business_idx)
            user_neigh=tf.concat([user_related_review,user_related_business],axis=1)
            user_neigh=tf.layers.conv1d(tf.expand_dims(user_neigh,axis=0),user_neighbor_out_sz,1,padding='valid',
                                                use_bias=False, kernel_initializer=tf.contrib.layers.xavier_initializer())
            
            
            hidden_user_neigh=tf.segment_mean(tf.squeeze(user_neigh),self.user_segment_id)
            
            user_self=tf.layers.conv1d(tf.expand_dims(user_inputs,axis=0),user_out_sz,1,padding='valid',
                                                use_bias=False, kernel_initializer=tf.contrib.layers.xavier_initializer())
            
            hidden_user_self=tf.squeeze(user_self)
            hidden_user_neigh = tf.nn.dropout(hidden_user_neigh,1-self.dropout)
            hidden_user_self = tf.nn.dropout(hidden_user_self,1-self.dropout)
            hidden_user=tf.concat([hidden_user_self,hidden_user_neigh],axis=1)
            ret=tf.contrib.layers.bias_add(hidden_user)

        return self.act(ret)
    
    
    
    
    def xiaomao_business_forward(self,id_layer,business_inputs,user_inputs,review_inputs,business_out_sz,business_neighbor_out_sz,act=tf.nn.elu):
        with tf.name_scope('business_layer' + str(id_layer)):
            #为了做global的kernel，还需要把item和review做成和user一样的list
            ##global business neighbor
            business_related_review=tf.gather(review_inputs,self.global_business_related_review_idx)
            business_related_user=tf.gather(user_inputs,self.global_business_related_user_idx)
            business_neigh=tf.concat([business_related_review,business_related_user],axis=1)
            business_neigh=tf.layers.conv1d(tf.expand_dims(business_neigh,axis=0),business_neighbor_out_sz,1,padding='valid',
                                                use_bias=False, kernel_initializer=tf.contrib.layers.xavier_initializer())
            
            business_neighbor_global_map=tf.segment_mean(tf.squeeze(business_neigh),self.global_business_segment_id)
            
            business_hidden=[]
            business_neighbor_hidden=[]
            all_business_idx_list=[]
            for i,business_idx in enumerate(tqdm(self.business_idx_cluster)):
                all_business_idx_list+=business_idx
                local_business_neighbor_review=tf.gather(review_inputs,self.business_related_review_idx[i])
                local_business_neighbor_user=tf.gather(user_inputs,self.business_related_user_idx[i])
                local_business_neigh=tf.concat([local_business_neighbor_review,local_business_neighbor_user],axis=1)
                local_business_neigh=tf.layers.conv1d(tf.expand_dims(local_business_neigh,axis=0),business_neighbor_out_sz,1,padding='valid',
                                                use_bias=False, kernel_initializer=tf.contrib.layers.xavier_initializer())
            
                
                
                local_business_neigh=tf.segment_mean(tf.squeeze(local_business_neigh),self.business_segment_id[i])
            
                
                business_neighbor_global=tf.gather(business_neighbor_global_map,business_idx)
                business_neighbor_local=tf.nn.dropout(local_business_neigh,1-self.dropout)
                business_neighbor_global=tf.nn.dropout(business_neighbor_global,1-self.dropout)
                business_neighbor=tf.add(business_neighbor_local,business_neighbor_global)
                business_neighbor_hidden.append(business_neighbor)
                
                
                
            from_neighs=tf.concat(business_neighbor_hidden,axis=0)

            business_id_list = np.argsort(all_business_idx_list)  #np.argsort:将元素从小到大排列后，提取索引(节点0对应在all_list当中的位置，节点1对应在all_list中的位置……)
            businessTFID = tf.Variable(tf.constant(business_id_list), trainable=False)
            hidden_neigh=tf.nn.embedding_lookup(from_neighs,businessTFID)
            hidden_neigh=tf.nn.dropout(hidden_neigh,1-self.dropout)
            
            
            hidden_self = tf.layers.conv1d(tf.expand_dims(business_inputs, axis=0), business_out_sz, 1, padding='valid',
                                           use_bias=False, kernel_initializer=tf.contrib.layers.xavier_initializer())
            hidden_self = tf.squeeze(hidden_self)
            hidden_self = tf.nn.dropout(hidden_self,1-self.dropout)

            business_hidden=tf.concat([hidden_self,hidden_neigh],axis=1)
            ret=tf.contrib.layers.bias_add(business_hidden)


        return self.act(ret)
    
    
    

    
    def xiaomao_single_business_forward(self,id_layer,business_inputs,user_inputs,review_inputs,business_out_sz,business_neighbor_out_sz,act=tf.nn.elu):
        with tf.name_scope('single_business_layer' + str(id_layer)):
            business_related_review=tf.gather(review_inputs,self.global_business_related_review_idx)
            business_related_user=tf.gather(user_inputs,self.global_business_related_user_idx)
            business_neigh=tf.concat([business_related_review,business_related_user],axis=1)
            business_neigh=tf.layers.conv1d(tf.expand_dims(business_neigh,axis=0),business_neighbor_out_sz,1,padding='valid',
                                                use_bias=False, kernel_initializer=tf.contrib.layers.xavier_initializer())
            
            
            hidden_business_neigh=tf.segment_mean(tf.squeeze(business_neigh),self.global_business_segment_id)
            
            business_self=tf.layers.conv1d(tf.expand_dims(business_inputs,axis=0),business_out_sz,1,padding='valid',
                                                use_bias=False, kernel_initializer=tf.contrib.layers.xavier_initializer())
            
            hidden_business_self=tf.squeeze(business_self)
            hidden_business_neigh = tf.nn.dropout(hidden_business_neigh,1-self.dropout)
            hidden_business_self = tf.nn.dropout(hidden_business_self,1-self.dropout)
            hidden_business=tf.concat([hidden_business_self,hidden_business_neigh],axis=1)
            ret=tf.contrib.layers.bias_add(hidden_business)
        return self.act(ret)
    
    
    
    def xiaomao_review_forward(self,id_layer,business_inputs,user_inputs,review_inputs,review_out_sz,act=tf.nn.elu):
        with tf.name_scope('xiaomao_review_layer' + str(id_layer)):
            review_related_business=tf.gather(business_inputs,self.related_business_idx)
            review_related_user=tf.gather(user_inputs,self.related_user_idx)
            review_concat=tf.concat([review_related_business,review_related_user],axis=1)
            
            hidden_review=tf.layers.dense(review_concat,review_out_sz,activation=act,
                                            kernel_initializer=tf.contrib.layers.xavier_initializer())
            
            hidden_review=tf.nn.dropout(hidden_review,1-self.dropout)
            ret=tf.concat([review_inputs,hidden_review],axis=1)
        return ret
    
    def single_review_forward(self,id_layer,business_inputs,user_inputs,review_inputs,review_out_sz,act=tf.nn.elu):
        with tf.name_scope('single_review_layer' + str(id_layer)):
            review_related_business=tf.gather(business_inputs,self.related_business_idx)
            review_related_user=tf.gather(user_inputs,self.related_user_idx)
            review_concat=tf.concat([review_inputs,review_related_business,review_related_user],axis=1)
            #print(review_concat.shape)
            hidden_review=tf.layers.dense(review_concat,review_out_sz,activation=act,
                                            kernel_initializer=tf.contrib.layers.xavier_initializer())
            #print(hidden_review.shape)
            hidden_review=tf.nn.dropout(hidden_review,1-self.dropout)
        return hidden_review
    
            
    def xiaomao_inference(self):
        """Create DEMO-Net With Weight-based Multi-task Function"""
        with tf.name_scope('multi-task_xiaomao'):
            business_city_emb=tf.gather(self.city_embedding_table,self.business_city_idx)
            business_state_emb=tf.gather(self.state_embedding_table,self.business_state_idx)
            business_postal_emb=tf.gather(self.postal_embedding_table,self.business_postal_idx)
            business_inputs=tf.concat([self.business_feat,business_city_emb,business_state_emb,business_postal_emb],axis=1)
            user_inputs=self.user_feat
            review_inputs=self.review_feat
            
            for i in range(self.num_layers-1):
                new_business_inputs=self.xiaomao_business_forward(i,business_inputs,user_inputs,review_inputs,\
                    business_out_sz=self.business_hidden_dim[i],business_neighbor_out_sz=self.business_neighbor_hidden_dim[i],act=self.act)
                
                new_user_inputs=self.xiaomao_user_forward(i,business_inputs,user_inputs,review_inputs,\
                    user_out_sz=self.user_hidden_dim[i],user_neighbor_out_sz=self.user_neighbor_hidden_dim[i],act=self.act)
                
                new_review_inputs=self.xiaomao_review_forward(i,business_inputs,user_inputs,review_inputs,review_out_sz=self.review_hidden_dim[i],act=self.act)
                
                business_inputs=new_business_inputs
                user_inputs=new_user_inputs
                review_inputs=new_review_inputs
            review_inputs=self.single_review_forward(self.num_layers-1,business_inputs,user_inputs,review_inputs,review_out_sz=self.review_hidden_dim[self.num_layers-1],act=self.act)
            
            review_logits=tf.layers.dense(review_inputs,16,kernel_initializer=tf.contrib.layers.xavier_initializer(),activation=self.act)
            
            review_logits=tf.nn.dropout(review_logits,1-self.dropout)
            review_logits=tf.layers.dense(review_logits,self.num_class,kernel_initializer=tf.contrib.layers.xavier_initializer(),activation=None)
        return review_logits
    

    
    
    def xiaomao_single_inference(self):
        """Create DEMO-Net With Weight-based Multi-task Function"""
        with tf.name_scope('gnn_xiaomao'):
            business_city_emb=tf.gather(self.city_embedding_table,self.business_city_idx)
            business_state_emb=tf.gather(self.state_embedding_table,self.business_state_idx)
            business_postal_emb=tf.gather(self.postal_embedding_table,self.business_postal_idx)
            business_inputs=tf.concat([self.business_feat,business_city_emb,business_state_emb,business_postal_emb],axis=1)
            user_inputs=self.user_feat
            review_inputs=self.review_feat
            
            for i in range(self.num_layers-1):
                new_business_inputs=self.xiaomao_single_business_forward(i,business_inputs,user_inputs,review_inputs,\
                    business_out_sz=self.business_hidden_dim[i],business_neighbor_out_sz=self.business_neighbor_hidden_dim[i],act=self.act)
                
                new_user_inputs=self.xiaomao_user_forward(i,business_inputs,user_inputs,review_inputs,\
                    user_out_sz=self.user_hidden_dim[i],user_neighbor_out_sz=self.user_neighbor_hidden_dim[i],act=self.act)
                
                new_review_inputs=self.xiaomao_review_forward(i,business_inputs,user_inputs,review_inputs,review_out_sz=self.review_hidden_dim[i],act=self.act)
                
                business_inputs=new_business_inputs
                user_inputs=new_user_inputs
                review_inputs=new_review_inputs
            review_inputs=self.single_review_forward(self.num_layers-1,business_inputs,user_inputs,review_inputs,review_out_sz=self.review_hidden_dim[self.num_layers-1],act=self.act)
            
            
            review_logits=tf.layers.dense(review_inputs,16,kernel_initializer=tf.contrib.layers.xavier_initializer(),activation=self.act)
            
            review_logits=tf.nn.dropout(review_logits,1-self.dropout)
            review_logits=tf.layers.dense(review_logits,self.num_class,kernel_initializer=tf.contrib.layers.xavier_initializer(),activation=None)
        return review_logits
    
    
            


    def training(self, loss, lr, l2_coef):
        vars = tf.trainable_variables()
        lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in vars if v.name not
                           in ['bias', 'gamma', 'b', 'g', 'beta']]) * l2_coef

        opt = tf.train.AdamOptimizer(learning_rate=lr)
        train_op = opt.minimize(loss + lossL2)
        return train_op





