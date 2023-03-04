import torch
import torch.nn as nn
from torch.optim import optimizer
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# 0) prepare data

# 建立x, y陣列存csv檔之資料
x, y = np.empty()
csv_file = np.genfromtxt('HW1-1.csv', dtype=float, unpack=True, skip_header=1, usecols=(1, 2))

# 1) model
class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()

        # define layers
        self.layer1 = nn.Linear(, )
        self.layer2 = nn.Linear(, )
        self.layer3 = nn.Linear(, )
        self.layer4 = nn.Linear(, )
        
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        return self.linear(x)

model = LinearRegression() #建立model

# 2) loss and optimizer
learning_rate = 0.01
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 3) training loop
epochs = 100
for epoch in range(epochs):
    # forward pass and loss
    y_predicted = model(feature)
    loss = criterion(y_predicted, target)

    # backward pass
    loss.backward()

    # update
    optimizer.step()
    
    # init optimizer
    optimizer.zero_grad()

    if (epoch + 1) % 10 == 0:
        print(f'epoch: {epoch+1}, loss = {loss.item(): .4f}')

# show in image
predicted = model(feature).detach().numpy()
plt.plot(feature_numpy, target_numpy, 'ro')
plt.plot(feature_numpy, predicted, 'b')
plt.show()