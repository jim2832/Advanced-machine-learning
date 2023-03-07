from __future__ import division
from sympy import *
import torch
import torch.nn as nn
from torch.optim import optimizer
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 0) prepare data
x = np.empty((0))
y = np.empty((0))
csv_file = np.genfromtxt('HW1-1.csv', delimiter=',', skip_header = 1)
csv_file = csv_file.tolist()
csv_file.sort(key = lambda l:l[0])
csv_file = np.array(csv_file)

for data in csv_file: # 逐筆放入陣列中
    x = np.append(x, float(data[0]))
    y = np.append(y, float(data[1]))

tensor_x = torch.tensor(x, dtype=torch.float32) # 將 numpy array 轉化成 tensor
tensor_y = torch.tensor(y, dtype=torch.float32)
tensor_x = tensor_x.view(tensor_x.shape[0], 1)
tensor_y = tensor_y.view(tensor_y.shape[0], 1)


# 1.model
w0 = torch.tensor(-5.0, requires_grad=True)
w1 = torch.tensor(-5.0, requires_grad=True)
w2 = torch.tensor(-5.0, requires_grad=True)
w3 = torch.tensor(-5.0, requires_grad=True)

def forward(X):
    return w0 + (w1 * X) + (w2 * X**2) + (w3 * X**3)


# 2) 設定相關參數
learning_rate = 0.1 # learning rate
criterion = nn.MSELoss() # loss function


# 3) training loop
epochs = 100
training_time = 500
for epoch in range(epochs):
    running_loss = 0.0

    for time in range(training_time):
        # forward pass and loss
        y_predicted = forward(tensor_x)
        loss = criterion(y_predicted, tensor_y)

        # backward pass
        loss.backward()
        
        # 計算loss
        running_loss += loss.item()

        if(time % 100 == 99):
            print("Epoch: [%d/%d], Iteration: [%s/%s], loss: %.3f" % (epoch+1, epochs, time+1, training_time, running_loss/100))
            running_loss = 0

        # 藉由每次訓練去更新 X 的係數
        w0.data = w0.data - learning_rate * w0.grad.data
        w1.data = w1.data - learning_rate * w1.grad.data
        w2.data = w2.data - learning_rate * w2.grad.data
        w3.data = w3.data - learning_rate * w3.grad.data

        # 歸零 X 係數的梯度
        w0.grad.data.zero_()
        w1.grad.data.zero_()
        w2.grad.data.zero_()
        w3.grad.data.zero_()


# 印出結果
print("w0 = ", float(w0))
print("w1 = ", float(w1))
print("w2 = ", float(w2))
print("w3 = ", float(w3))

# def diff(X):
#     return w1 + 2 *(w2 * X) + 3 *(w3 * x**2)

# print("f'(3.0) = ", diff(3.0))
# print("f'(0.1) = ", diff(0.1))
# print("f'(-0.5) = ", diff(-0.5))

predicted = forward(tensor_x).detach().numpy()
plt.plot(tensor_x, tensor_y, 'ro')
plt.plot(tensor_x, predicted, 'b')
plt.show()