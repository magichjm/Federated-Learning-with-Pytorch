# client.py
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import functional as F
from tqdm import tqdm


class Client:
    def __init__(self, model, id):
        self.id = id
        self.local_model = model
        self.optimizer = optim.SGD(self.local_model.parameters(), lr=0.001)
        self.lossfn = F.binary_cross_entropy_with_logits
        self.dataset = None
        self.dataloader = None

    def set_trian_data(self, dataset, batchsize=64, num_workers=4):
        self.dataset = dataset
        self.dataloader = DataLoader(self.dataset, batch_size=batchsize, shuffle=True, num_workers=num_workers)

    def train(self, epochs):
        """
        :param epochs:在本地训练时迭代的轮数
        """
        total_steps = len(self.dataloader) * epochs
        progress_bar = tqdm(total=total_steps, position=0, leave=True)  # 设置 leave=True 保留进度条在终端
        for epoch in range(epochs):
            for step, (x, y) in enumerate(self.dataloader):
                self.optimizer.zero_grad()
                outputs = self.local_model(x)
                loss = self.lossfn(outputs, y)
                loss.backward()
                self.optimizer.step()
                # 更新进度条
                progress_bar.update(1)
                progress_bar.set_postfix({'id': self.id, 'Epoch': epoch, 'Step': step, 'Loss': loss.item()})
        progress_bar.close()

    def get_len(self):
        return len(self.dataset)

    def get_model(self):
        # 返回本地模型的权重
        return self.local_model.state_dict()

    def update_global_model(self, global_model):
        # 更新本地模型的权重
        self.local_model.load_state_dict(global_model.state_dict())
