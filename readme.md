#CA-GNN

This repository is a tensorflow implementation of the CIKM'2020 paper: 

Category-aware Graph Neural Networks for Improving E-commerce Review Helpfulness Prediction


##Requirements:

Tensorflow == 1.13


##Dataset:

You can download the Yelp dataset (Round 13) from [here](https://www.yelp.com/dataset), and put the json files along the side of the floders **preprocess** and **src**:

```
|--- yelp_academic_dataset_business.json
|--- yelp_academic_dataset_user.json
|--- yelp_academic_dataset_review.json
|--- preprocess
|--- |--- preprocessing.py
|--- |--- gen_business_feats.py
|--- |--- cluster_kmeans.py
|--- |--- cluster_kmeans_label.py
|--- |--- business_kmeans_read.py
|--- |--- gen_user_feats.py
|--- |--- gen_doc2vec_model.py
|--- |--- gen_review_feats.py
|--- |--- gen_cluster.py
|--- |--- feat_scale.py
|--- src
|--- |--- new_models.py
|--- |--- new_main.py
```


##Data preprocessing:

Run the preprocessing code files in the following order:

```
1.  preprocessing.py
2.  gen_business_feats.py     
3.  cluster_kmeans.py
4.  cluster_kmeans_label.py
5.  business_kmeans_read.py
6.  gen_user_feats.py
7.  gen_doc2vec_model.py       
8.  gen_review_feats.py        
9.  gen_cluster.py 
10. feat_scale.py
```


##Model training

Run new_main.py


## Changing the number of clusters:

If you want to change the number of clusters, modify and run the files in the following order:

```
1. cluster_kmeans_label.py
2. business_kmeans_read.py
3. gen_cluster.py

```


##Notice:

This is a code implementation on Yelp Dataset. The implementation on Amazon Dataset is similar to the code in this repository. If you want to test CA-GNN on Amazon Dataset, modify the data preprocessing code. And the node embeddings of the amazon users and items are trainable id embeddings.

Besides, we just release the full-batch version of our code. If the dataset you use is larger and your memory is not enough, split the full batch of reviews into mini-batches of reviews, and sample neighbors as [GraphSAGE](https://github.com/williamleif/GraphSAGE) do.



##Citation

```
@inproceedings{DBLP:conf/cikm/QuLWZZJHXZG20,
  author    = {Xiaoru Qu and
               Zhao Li and
               Jialin Wang and
               Zhipeng Zhang and
               Pengcheng Zou and
               Junxiao Jiang and
               Jiaming Huang and
               Rong Xiao and
               Ji Zhang and
               Jun Gao},
  title     = {Category-aware Graph Neural Networks for Improving E-commerce Review
               Helpfulness Prediction},
  booktitle = {{CIKM} '20: The 29th {ACM} International Conference on Information
               and Knowledge Management, Virtual Event, Ireland, October 19-23, 2020},
  pages     = {2693--2700},
  publisher = {{ACM}},
  year      = {2020},
}
```






