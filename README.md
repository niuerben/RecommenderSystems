# 项目介绍
介绍推荐系统基本知识，相关算法以及实现。


# 目录规划
* data 测试用数据集合 
* manual 相关资料集合  
	* paper阅读分享
	* 基础知识分享
* outputs 
    * itemcf_recommendation.csv 基于物品协同过滤的推荐表
    * metrics.csv 算法指标（精确率、召回率）
    * usercf_recommendation.csv 基于用户协同过滤的推荐表
* py3x 基于python的协同过滤算法实现
* references 引用信息
* RS-torch 基于PyTorch实现的推荐系统，包含DCN_torch和XSimGCL
* main.py 运行协同过滤算法的主脚本


# 内容导航
## python 实现（主要用于原理理解）
* ItemCF 基于物品的协同过滤算法
* UserCF 基于用户的协同过滤算法
* LFM 没看
* Graph—Based 没看

# 计划项(原作挖坑，下面都没看，后续再考虑要不要删了或者实现)
## 推荐算实现
### 基于用户行为数据的推荐算法
* 关联规则  
* LFM   
* Graph   
* ALS  

### 利用用户标签数据推荐算法
* LDA  
* TF-IDF  
* TagCF  

### 探索性研究（各个paper的实现）
* Markov Chain  
* 社交网络  
* 基于深度学习的推荐算法
....


## 评价系统实现

## 推荐系统架构实现
### 外围架构
#### 用户行为日志存储系统
#### 日志系统
#### UI

### 功能模块
#### 数据录入模块
#### 用户特征生成模块
#### 推荐模块
#### 过滤模块
#### 排名模块




