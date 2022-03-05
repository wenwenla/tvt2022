import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as opt
from torch.utils.data import DataLoader


PATH = 'new_version/sl_samples_32.pkl'
MODEL_PATH = 'new_version/sl_models_32'


class PredictModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(48, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class MyDataSet(torch.utils.data.Dataset):

    def __init__(self):
        with open(PATH, 'rb') as f:
            self._data = pickle.load(f)
        self._len = len(self._data)

    def __getitem__(self, item):
        x = self._data[item][:48]
        y = self._data[item][48:]
        return x, y

    def __len__(self):
        return self._len


def main():
    dataset = MyDataSet()

    train_size = int(len(dataset) * 0.8)
    test_size = len(dataset) - train_size
    train_ds, test_ds = torch.utils.data.random_split(dataset, [train_size, test_size])

    print(len(train_ds))
    print(len(test_ds))

    train_dl= DataLoader(train_ds, batch_size=1024, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=1024, shuffle=True)

    model = PredictModel()
    optimizer = opt.Adam(model.parameters(), lr=1e-3)

    rmse = []
    for i in range(200):
        print(f'{i}...')
        loss_item = []
        for x, y in train_dl:
            y_hat = model(x.float())
            loss_fn = nn.MSELoss()
            loss = loss_fn(y.float(), y_hat)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_item.append(loss.detach().item())
        loss_item_test = []
        for x, y in test_dl:
            loss_fn = nn.MSELoss()
            with torch.no_grad():
                y_hat = model(x.float())
                loss = torch.sqrt(loss_fn(y.float(), y_hat))
            loss_item_test.append(loss.detach().item())

        print(f'Loss: {np.mean(loss_item)}, Test Loss RMSE: {np.mean(loss_item_test)}')
        torch.save(model.state_dict(), f'{MODEL_PATH}/model_{i}.pkl')
        rmse.append(np.mean(loss_item_test))
    torch.save(rmse, f'{MODEL_PATH}/rmse.pkl')


if __name__ == '__main__':
    main()
    # sl_model_comp()