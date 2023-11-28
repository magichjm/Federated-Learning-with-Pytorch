# Federated-Learning-framework-with-Pytorch

基于fedavg联邦学习算法和smote过采样对信用卡欺诈交易检测任务进行了优化，其中fedavg算法使用torch1.13.1实现了服务器<--->节点的双向参数传递。

### 上手指南

###### 开发前的配置要求

1. python==3.10
2. torch=1.13.1

### 文件目录说明

eg:

```
filetree 
├── README.md
├── cilent.py                #联邦客户端
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

