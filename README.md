# Twitter100k: A Real-world Dataset for Weakly Supervised Cross-Media Retrieval

Yuting Hu, Liang Zheng, Yi Yang, and Yongfeng Huang

## Introduction

This paper contributes a new large-scale dataset for weakly supervised cross-media retrieval, named Twitter100k.
It is characterized by two aspects: 1) it has 100,000 image-text pairs randomly crawled from Twitter and thus has no constraint in the image categories; 2) text in Twitter100k is written in informal language by the users.

Since strongly supervised methods leverage the class labels that may be missing in practice, this paper focuses on weakly supervised learning for cross-media retrieval, in which only text-image pairs are exploited during training. We extensively benchmark the performance of four subspace learning methods and three variants of the Correspondence AutoEncoder, along with various text features on Wikipedia, Flickr30k and Twitter100k.

As a minor contribution, inspired by the characteristic of Twitter100k, we propose an OCR-based cross-media retrieval method. In experiment, we show that the proposed OCR-based method improves the baseline performance.

Detailed description is provided in our paper.

## Requirements

- This software is both tested on Windows 10 and CentOS Linux release 7.3.1611.
- Matlab (tested with R2016a both on Windows and CentOS).
- Python (tested with 2.7.5 both on Windows and CentOS).
- <a href="https://github.com/FangxiangFeng/deepnet"> Deepnet</a> and its dependencies. (Copyright is held by the author.)

## How to use the code

### For subspace learning methods (CCA, PLS, BLM, GMMFA)

1. download the data of the three benckmark datasets from my <a href="http://ngn.ee.tsinghua.edu.cn/members/yuting-hu/"> homepage</a> and put them into the folders *feature/* or other folders convenient to you.
2. modify the dataset name and the data path variables of the script file *run_baseline.m* in *code/GMA-CVPR2012/*.
3. run the matlab script file *run_baseline.m*.
4. run *retrieve.py* for a specific dataset and the results of the rank of ground truth will be saved in *result/rank/*. 

### For Corr-AE methods

1. download the data of the three benckmark datasets from my <a href="http://ngn.ee.tsinghua.edu.cn/members/yuting-hu/"> homepage</a> and put them into the folders *feature/* or other folders convenient to you.
2. run the python script file *genNPYdata.py* in *code/deepnet-master/deepnet/examples/yutinghu/* to generate the input data for Corr-AE methods.
3. install deepnet and its dependencies with patience following the instruction *INSTALL.TXT* in *code/deepnet-master/*.
4. run *runall\_all.sh* in *code/deepnet-master/deepnet/examples/yutinghu/wikipedia/* or *flickr30k/*, *twitter100k*.
5. run *retrieve\_corr\_ae.py* for a specific dataset and the results of the rank of ground truth will be saved in *result/rank/*. 

## Result Files

You can download the results of CMC saved in MAT-file format for direct comparison. 

## Data
<a href="http://203.91.121.76/wp-content/uploads/twitter100k.tar"> The Twitter100k dataset </a> (10G)

<a href="http://203.91.121.76/wp-content/uploads/samples.tar"> Samples of Twitter100k </a> (8.4M)

<a href="http://203.91.121.76/wp-content/uploads/dataset_split.tar"> Dataset_split </a> (2M)

## Homepage
I'm quite sorry that my <a href="http://ngn.ee.tsinghua.edu.cn/members/yuting-hu"> homepage </a> is not accessible now. I'll try to fix it as soon as possible. If you have any questions, please contact me by email (huyt16@mails.tsinghua.edu.cn).
