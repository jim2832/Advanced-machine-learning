import torch
import torch.nn as nn
from torch.optim import optimizer
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# 0) prepare data
x, y = np.empty((0))
csv_file = np.genfromtxt('HW1-1.csv', delimiter=',', skip_header=1)
for data in csv_file: # 逐筆放入陣列中
    x = np.append(x, float(data[0]))
    y = np.append(y, float(data[1]))
tensor_x, tensor_y = torch.Tensor(x), torch.Tensor(y) # 將 numpy array 轉化成 tensor
tensor_x, tensor_y = tensor_x.view(tensor_x.shape[0], 1), tensor_y.view(tensor_y.shape[0], 1)


# 1) model
class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()

        # define layers
        self.layer1 = nn.Linear(1, 32)
        self.layer2 = nn.Linear(32, 128)
        self.layer3 = nn.Linear(128, 64)
        self.layer4 = nn.Linear(64, 1)

        #dropout
        self.dropout = nn.Dropout(0.25)
        
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.dropout(x)
        x = F.relu(self.layer2(x))
        x = self.dropout(x)
        x = F.relu(self.layer3(x))
        x = self.dropout(x)
        x = F.relu(self.layer4(x))
        return x

model = LinearRegression() #建立model


# 2) loss and optimizer
learning_rate = 0.01
criterion = nn.MSELoss() # loss function
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) # gradient descent


# 3) training loop
epochs = 5
training_times = 8000
for epoch in range(epochs):
    running_loss = 0.0

    for time in range(training_times):
        # forward pass and loss
        y_predicted = model.forward(tensor_x)
        loss = criterion(y_predicted, tensor_y)

        # backward pass
        loss.backward()

        # update
        optimizer.step()
        
        # init optimizer
        optimizer.zero_grad()

        running_loss += loss.item()

        if(time % 100 == 99):
            print("Epoch: [%d/%d], Iteration: [%s/%s], loss: %.3f" % (epoch+1, epochs, time+1, training_times, running_loss))

# show in image
predicted = model(tensor_x).detach().numpy()
plt.plot(tensor_x, tensor_y, 'ro')
plt.plot(tensor_x, predicted, 'b')
plt.show()