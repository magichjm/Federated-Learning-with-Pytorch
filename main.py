# main.py
from server import Server
from client import Client
from model import Model
from load_data import *
import copy


def Initialization(num, model):
    # 初始化服务器
    server = Server(model=copy.deepcopy(model))
    test_data = Test_Data(datas)
    server.set_test_data(test_data)
    # 初始化客户端
    clients = [Client(id=i, model=copy.deepcopy(model)) for i in range(0, num)]
    # 获取数据并分配
    train_data = Train_Data(datas)
    subset_len = len(train_data) // num
    # 创建 Subset 对象的列表
    subsets = [Subset(train_data, list(range(i * subset_len, (i + 1) * subset_len))) for i in range(num)]
    # 将 Subset 分配给每个 Client
    for client, subset in zip(clients, subsets):
        client.set_trian_data(subset)
    return server, clients


datas = Datas()
#节点数量
node_num = 5
model = Model()
#全局迭代次数
global_steps = 100

if __name__ == '__main__':
    server, clients = Initialization(node_num, model)
    for global_step in range(0, global_steps):
        for client in clients:
            client.train(epochs=1)
        server.average(clients)
        if global_step % 10 == 0:
            loss, accuracy = server.test()
            print(f'global_step:{global_step}\tloss:{loss.item()}\taccuracy:{accuracy}')
