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
        client_lens = [client.get_len() for client in clients]
        # 计算每个客户端的权重
        weights = [length / sum(client_lens) for length in client_lens]
        # 初始化平均权重字典
        avg_weights = {key: torch.zeros_like(client_models[0][key]).float() for key in client_models[0]}
        # 加权平均每个客户端的模型参数
        for i, client_model in enumerate(client_models):
            for key in avg_weights:
                avg_weights[key] += weights[i] * client_model[key]
        # 更新全局模型权重
        self.global_model.load_state_dict(avg_weights)
        # 将全局模型更新到每个客户端
        for client in clients:
            client.update_global_model(self.global_model)

    def set_test_data(self, dataset, num_workers=4):
        self.dataset = dataset
        self.dataloader = DataLoader(self.dataset, batch_size=len(dataset), shuffle=True, num_workers=num_workers)

    def get_model(self):
        # 返回全局模型参数
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
