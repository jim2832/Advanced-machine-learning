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

# 設置模型
def forward(X_t):
    return W1 + W2 * X_t + W3 * (X_t * X_t) + W4 * (X_t * X_t * X_t)

# 設置 X 的係數
W1 = torch.tensor(-5.0, requires_grad=True)
W2 = torch.tensor(-5.0, requires_grad=True)
W3 = torch.tensor(-5.0, requires_grad=True)
W4 = torch.tensor(-5.0, requires_grad=True)

# 訓練模型
def train(X_t, criterion):
    # 總共有幾回合
    for epoch in range(total_epochs):
        train_loss = 0
        train_time = tqdm(range(one_epoch))
        # 一回合有幾次
        for i in train_time:
            # 透過模型的 forward propagation 取得資料的輸出結果
            y_pred = forward(X_t)
            # 透過輸出結果和 label 來計算損失量
            loss = criterion(y_pred,Y_t)
            # 根據loss進行 back propagation，來計算梯度
            loss.backward()
            train_loss += loss.item()
            train_time.set_description(f'Train Epoch {epoch+1}')
            train_time.set_postfix({'loss':float(train_loss)/(i+1)})
            
            # 更新 X 的係數
            W1.data = W1.data - step_size * W1.grad.data
            W2.data = W2.data - step_size * W2.grad.data
            W3.data = W3.data - step_size * W3.grad.data
            W4.data = W4.data - step_size * W4.grad.data
            # 迭代後歸零 X 係數的梯度
            W1.grad.data.zero_()
            W2.grad.data.zero_()
            W3.grad.data.zero_()
            W4.grad.data.zero_()

# 設置相關參數
total_epochs = 10
one_epoch = 100
# 設定梯度移動率
step_size = 0.1
# 設定損失函數
criterion = nn.MSELoss()

# 開始訓練
print('------------------------------------開始訓練------------------------------------')
train(X_t, criterion)
print('------------------------------------方程式係數------------------------------------')
print('W1 = %f' % (W1))
print('W2 = %f' % (W2))
print('W3 = %f' % (W3))
print('W4 = %f' % (W4))

# 顯示訓練結果
predicted = forward(X_t).detach().numpy()
plt.plot(X_t,Y_t,'ro')
plt.plot(X_t,predicted,'b')
plt.show()