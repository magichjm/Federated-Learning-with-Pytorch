# Federated-Learning-framework-with-Pytorch

基于fedavg联邦学习算法与smote过采样对联邦信用卡欺诈交易检测任务进行了优化，其中fedavg算法通过创建Server与Clients对象来模拟真实场景下的服务器<-->节点的双向参数传递。

### 任务说明

信用欺诈每年给银行带来数十亿美元的损失，然而现有的欺诈检测存在以下两个问题：

- 由于数据隐私和安全问题，数据集通常不允许在不同银行之间共享
- 在已有欺诈检测数据集中，正负比例严重失调

因此，本项目提出联邦学习+过采样的思路来解决数据隐私与数据不平衡的问题。

### 数据集

该数据集包含欧洲持卡人于 2013 年 9 月通过信用卡进行的交易信息。在 284807 笔交易中，存在 492 起欺诈，数据集高度不平衡，正类（欺诈）仅占所有交易的 0.172%。原数据集已做脱敏处理和PCA处理，匿名变量V1， V2， ...V28 是 PCA 获得的主成分，唯一未经过 PCA 处理的变量是 Time 和 Amount。Time 是每笔交易与数据集中第一笔交易之间的间隔，单位为秒；Amount 是交易金额。Class 是分类变量，在发生欺诈时为1，否则为0。

*来源：[Credit Card Fraud Detection (kaggle.com)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data)*

### 算法描述

![image-20231128154608579](https://github.com/magichjm/Federated-Learning-Credit-Card-Fraud-Detection-with-Pytorch/blob/master/process.png)

### 上手指南

###### 开发前的配置要求

1. python==3.10
2. torch=1.13.1

### 文件目录说明

```
filetree 
├── README.md
├── client.py                #联邦客户端
├── creditcard.csv           #信用卡欺诈检测数据集，负样本远远超过正样本
├── load_data.py             #数据预处理及数据集划分
├── main.py                  #运行入口
├── model.py                 #联邦模型架构
├── server.py                #联邦服务器
```

### 运行

```shell
run main.py
```

