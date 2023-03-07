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
class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()

        #定義多層神經網路
        self.layer1 = nn.Linear(1, 1024)
        self.layer2 = nn.Linear(1024, 1)

        #設定dropout
        self.dropout = nn.Dropout(0.25)
        
    def forward(self, x):
        x = self.layer1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.layer2(x)
        return x

model = LinearRegression() #建立model


# 2) loss and optimizer
learning_rate = 0.001 # learning rate
criterion = nn.MSELoss() # loss function
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) # gradient descent


# 3) training loop
epochs = 5
training_time = 10000
for epoch in range(epochs):
    running_loss = 0.0

    for time in range(training_time):
        # forward pass and loss
        y_predicted = model.forward(tensor_x)
        loss = criterion(y_predicted, tensor_y)

        #將梯度歸0
        optimizer.zero_grad()

        # backward pass
        loss.backward()

        # update
        optimizer.step()
        
        # 計算loss
        running_loss += loss.item()

        if(time % 100 == 99):
            print("Epoch: [%d/%d], Iteration: [%s/%s], loss: %.3f" % (epoch+1, epochs, time+1, training_time, running_loss))
            running_loss = 0


# 印出結果
predicted = model(tensor_x).detach().numpy()
plt.plot(tensor_x, tensor_y, 'ro')
plt.plot(tensor_x, predicted, 'b')
plt.show()