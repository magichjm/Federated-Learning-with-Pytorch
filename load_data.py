from imblearn.over_sampling import SMOTE
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data.dataset import Subset
from torch.utils.data import DataLoader


class Datas:
    def __init__(self):
        super().__init__()
        pd_data = pd.read_csv("creditcard.csv")
        data = np.array(pd_data, dtype=np.float32)
        # print(pd_data)
        x = data[::, 0:-1:]
        y = data[::, -1::]
        # 对第一列进行Sigmoid操作并替换
        x[:, 0] = 1 / (1 + np.exp(-x[:, 0]))
        # 对最后一列进行Sigmoid操作并替换
        x[:, -1] = 1 / (1 + np.exp(-x[:, -1]))
        smote = SMOTE()
        x, y = smote.fit_resample(x, y)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.2)

    def get_data(self):
        return self.x_train, self.x_test, self.y_train, self.y_test


class Train_Data(Dataset):
    def __init__(self, datas):
        super().__init__()
        self.x_train, x_test, self.y_train, y_test = datas.get_data()

    def __len__(self):
        return len(self.x_train)

    def __getitem__(self, item):
        feature = torch.rand([1, 256])
        for index, data in enumerate(self.x_train[item]):
            feature[0][3 * index] = torch.tensor(data)
        return feature.reshape((1, 16, 16)), torch.tensor(np.expand_dims([self.y_train[item]], axis=0))


class Test_Data(Dataset):
    def __init__(self, datas):
        super().__init__()
        x_train, self.x_test, y_train, self.y_test = datas.get_data()

    def __len__(self):
        return len(self.x_test)

    def __getitem__(self, item):
        feature = torch.rand([1, 256])
        for index, data in enumerate(self.x_test[item]):
            feature[0][3 * index] = torch.tensor(data)
        return feature.reshape((1, 16, 16)), torch.tensor(np.expand_dims([self.y_test[item]], axis=0))


# if __name__ == "__main__":
#     datas = Datas()
#     train_data = Train_Data(datas)
#     # 将数据集分为 n 个部分
#     n = 5
#     data_len = len(train_data)
#     subset_len = data_len // n
#     # 创建 Subset 对象的列表
#     subsets = [Subset(train_data, list(range(i * subset_len, (i + 1) * subset_len))) for i in range(n)]
#     data = [subset for subset in subsets]
#     dataloader = DataLoader(data[0], batch_size=64, shuffle=True, num_workers=1)
#     for index, (x, y) in enumerate(dataloader):
#         print(index, x.shape[0])
