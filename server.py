# server.py
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F


class Server:
    def __init__(self, model):
        # 初始化全局模型
        self.dataloader = None
        self.dataset = None
        self.global_model = model
        self.lossfn = F.binary_cross_entropy_with_logits

    def average(self, clients):
        client_models = [client.get_model() for client in clients]
        # 聚合客户端模型权重的平均值
        avg_weights = {}
        for key in client_models[0]:
            avg_weights[key] = torch.stack([client_model[key] for client_model in client_models]).float().mean(0)
        # 更新全局模型权重
        self.global_model.load_state_dict(avg_weights)
        for client in clients:
            client.update_global_model(self.global_model)

    def set_test_data(self, dataset, num_workers=4):
        self.dataset = dataset
        self.dataloader = DataLoader(self.dataset, batch_size=len(dataset), shuffle=True, num_workers=num_workers)

    def get_model(self):
        # 返回本地模型的权重
        return self.global_model.state_dict()

    def test(self):
        total_samples = 0
        correct_predictions = 0

        for (x, y) in self.dataloader:
            # 获取模型预测结果
            logits = self.global_model(x)
            # 计算损失
            loss = self.lossfn(logits, y)
            # 对 logits 应用 sigmoid 激活函数
            probabilities = torch.sigmoid(logits)
            # 将概率值转换为预测标签
            predicted_labels = (probabilities > 0.5).float()
            # 计算准确率
            total_samples += y.size(0)
            correct_predictions += (predicted_labels == y).sum().item()
            accuracy = correct_predictions / total_samples
            return loss, accuracy
