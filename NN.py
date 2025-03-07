# %%
from pathlib import Path
from typing import OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
import tqdm
from torch import nn
from torch.amp import GradScaler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# %%
class NN(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.rl = nn.ReLU()
        self.fc1 = nn.Linear(in_features, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 2048)
        self.fc4 = nn.Linear(2048, 2048)
        self.fc5 = nn.Linear(2048, 1024)
        self.fc6 = nn.Linear(1024, 1024)
        self.fc7 = nn.Linear(1024, 512)
        self.fc8 = nn.Linear(512, 512)
        self.fc9 = nn.Linear(512, 256)
        self.fc10 = nn.Linear(256, 256)
        self.fc11 = nn.Linear(256, 128)
        self.fc12 = nn.Linear(128, 128)
        self.fc13 = nn.Linear(128, 64)
        self.fc14 = nn.Linear(64, 64)
        self.fc15 = nn.Linear(64, out_features)
        self.dropout = nn.Dropout(0.5)
        self.batchnorm1 = nn.BatchNorm1d(4096)
        self.batchnorm2 = nn.BatchNorm1d(2048)
        self.batchnorm3 = nn.BatchNorm1d(1024)
        self.batchnorm4 = nn.BatchNorm1d(512)
        self.batchnorm5 = nn.BatchNorm1d(256)
        self.batchnorm6 = nn.BatchNorm1d(128)

    def forward(self, x):
        x = self.fc1(x)
        x = self.rl(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.rl(x)
        x = self.batchnorm1(x)
        x = self.fc3(x)
        x = self.rl(x)
        x = self.dropout(x)
        x = self.fc4(x)
        x = self.rl(x)
        x = self.batchnorm2(x)
        x = self.fc5(x)
        x = self.rl(x)
        x = self.dropout(x)
        x = self.fc6(x)
        x = self.rl(x)
        x = self.batchnorm3(x)
        x = self.fc7(x)
        x = self.rl(x)
        x = self.dropout(x)
        x = self.fc8(x)
        x = self.rl(x)
        x = self.batchnorm4(x)
        x = self.fc9(x)
        x = self.rl(x)
        x = self.dropout(x)
        x = self.fc10(x)
        x = self.rl(x)
        x = self.batchnorm5(x)
        x = self.fc11(x)
        x = self.rl(x)
        x = self.dropout(x)
        x = self.fc12(x)
        x = self.rl(x)
        x = self.batchnorm6(x)
        x = self.fc13(x)
        x = self.rl(x)
        x = self.dropout(x)
        x = self.fc14(x)
        x = self.rl(x)
        x = self.fc15(x)
        x = F.softmax(x, dim=0)
        return x


class MyDataset(Dataset):
    def __init__(self, feature, target, transforms=None):
        super().__init__()
        self.transforms = transforms
        self.feature = feature
        self.target = target

    # ここで取り出すデータを指定している
    def __getitem__(self, idx):
        temp_feature = self.feature[idx]
        temp_target = self.target[idx]

        # データの変形 (transforms)
        if self.transforms is not None:
            temp_feature = self.transforms(temp_feature)
        return temp_feature, temp_target

    # この method がないと DataLoader を呼び出す際にエラーを吐かれる
    def __len__(self):
        return len(self.feature)


def train(
    model, total_epoch, train_dataloader, val_dataloader, loss_func, optimizer, device
):

    scaler = GradScaler(device=device)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, mode="min", factor=0.1, patience=10, verbose=True
    # )
    for epoch in range(total_epoch):
        with tqdm(train_dataloader) as pbar:
            pbar.set_description(f"[Epoch {epoch + 1}/{total_epoch}]")
            model.train()
            tloss = 0
            # ミニバッチごとにループさせる,train_loaderの中身を出し切ったら1エポックとなる
            for i in pbar:
                input = i[0].float().to(device)
                target = i[1].float().to(device)
                output = model(input)  # 順伝播
                optimizer.zero_grad()  # 勾配を初期化（前回のループ時の勾配を削除）
                train_loss = loss_func(output, target)  # 損失を計算
                tloss += train_loss.item()
                scaler.scale(train_loss).backward()  # 逆伝播で勾配を計算

                scaler.step(optimizer)  # 最適化
                scaler.update()

                pbar.set_postfix(
                    OrderedDict(
                        Loss=train_loss.item(),
                    )
                )
            tloss /= len(train_dataloader)

            model.eval()
            val_loss = 0
            with torch.inference_mode():
                for i in val_dataloader:
                    input = i[0].float().to(device)
                    target = i[1].float().to(device)

                    preds = model(input)

                    val_loss += loss_func(preds, target).item()

            val_loss /= len(val_dataloader)
            print(f"Train_Loss={tloss:.4f}  Val_Loss={val_loss:.4f}")
            # scheduler.step(val_loss)


# %%
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device)

    data = np.load("Preprocessed_data.npz", allow_pickle=True)
    feature = data["feature"].astype(float)
    result = data["result"]
    id_date = pd.DataFrame(data["id_date"][:, [16, 8]], columns=["ID", "date"])
    train_idx = id_date["date"].dt.year < 2025
    test_idx = id_date["date"].dt.year >= 2025
    train_val_dataset = MyDataset(feature[train_idx], result[train_idx])
    test_dataset = MyDataset(feature[test_idx], result[test_idx])

    train_ratio = 0.8
    n_samples = len(train_val_dataset)
    train_size = int(n_samples * train_ratio)
    val_size = n_samples - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_val_dataset, [train_size, val_size]
    )

    train_Dataloader = DataLoader(
        train_dataset,
        batch_size=512,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    val_Dataloader = DataLoader(
        val_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    model = NN(feature.shape[1], result.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters())
    loss = torch.nn.CrossEntropyLoss()

    epoch = 1000
    train(
        model,
        epoch,
        train_Dataloader,
        val_Dataloader,
        loss_func=loss,
        optimizer=optimizer,
        device=device,
    )
    # %%
    model_scripted = torch.jit.script(model)
    model_scripted.save("model/KeibaYosou.pth")


# %%
