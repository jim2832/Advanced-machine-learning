import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# 建立 X, Y 軸數值陣列
X = np.empty((0))
Y = np.empty((0))

# 讀取 csv 檔案
fileValue = np.genfromtxt('HW1-1.csv', delimiter=',', skip_header=1)

# 分類成 X, Y 軸數值
for r in fileValue:
    X = np.append(X, float(r[0]))
    Y = np.append(Y, float(r[1]))

# 將數值轉換成 PyTorch 二維張量
X_t = torch.tensor(X, dtype=torch.float32)
Y_t = torch.tensor(Y, dtype=torch.float32)
X_t = X_t.view(X_t.shape[0],1)
Y_t = Y_t.view(Y_t.shape[0],1)

# 建立模型
class simulator(nn.Module):
    def __init__(self):
        super(simulator,self).__init__()
        
        # 全連接層
        self.l1 = nn.Linear(1, 32)
        self.l2 = nn.Linear(32, 128)
        self.l3 = nn.Linear(128, 64)
        self.l4 = nn.Linear(64, 1)

        # dropout
        self.dropout = nn.Dropout(0.25)
    
    # 設置模型中的各層順序
    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.dropout(x)
        x = F.relu(self.l2(x))
        x = self.dropout(x)
        x = F.relu(self.l3(x))
        x = self.dropout(x)
        x = self.l4(x)
        return x

# 訓練模型
def train(X_t, model, optimizer, criterion):
    # 總共有幾回合
    for epoch in range(total_epochs):
        train_loss = 0
        train_time = tqdm(range(one_epoch))
        # 一回合有幾次
        for i in train_time:
            # 透過模型的 forward propagation 取得資料的輸出結果
            y_pred = model.forward(X_t)
            # 透過輸出結果和 label 來計算損失量
            loss = criterion(y_pred,Y_t)
            # 清空前一次的梯度
            optimizer.zero_grad()
            # 根據loss進行 back propagation，來計算梯度
            loss.backward()
            # 做梯度下降
            optimizer.step()
            
            train_loss += loss.item()
            train_time.set_description(f'Train Epoch {epoch+1}')
            train_time.set_postfix({'loss':float(train_loss)/(i+1)})

# 設置梯度的學習率
learning_rate = 0.001
# 設定有幾個 epoch
total_epochs = 5
# 設定每次 epoch 有幾輪
one_epoch = 10000
# 設定使用模型
model = simulator()
# 設定損失函數
criterion = nn.MSELoss()
# 設定梯度
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 開始訓練
train(X_t, model, optimizer, criterion)

# 顯示訓練結果
predicted = model(X_t).detach().numpy()
plt.plot(X_t,Y_t,'ro')
plt.plot(X_t,predicted,'b')
plt.show()