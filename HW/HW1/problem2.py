# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchsummary import summary


# GPU
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('GPU state:', device)


# Cifar-10 data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


# Data
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
trainLoader = torch.utils.data.DataLoader(trainset, batch_size=8, shuffle=True, num_workers=0)
testLoader = torch.utils.data.DataLoader(testset, batch_size=8, shuffle=False, num_workers=0)


# Data classes
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# Model structure
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1))
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = (3, 3), stride = (1, 2))
        
        self.max_pooling = nn.MaxPool2d(kernel_size = (2, 2),stride = (2, 2))
        self.dropout = nn.Dropout(p = 0.25)
        
        self.padding1 = nn.ZeroPad2d(8)
        self.padding2 = nn.ZeroPad2d((9, 8, 1, 1))

        self.fc1 = nn.Linear(64*32*32, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 10)
        #self.Softmax = nn.Softmax(dim = 1)


    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.max_pooling(x)
        x = self.padding1(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.padding2(x)
        x = F.relu(x)
        x = self.max_pooling(x)
        x = self.padding1(x)
        x = self.dropout(x)

        x = x.view(-1, 64*32*32) # -1 -> 不指定flatten的大小

        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        #x = self.Softmax(x)

        return x

net = Model().to(device)
print(net) # 印出神經網路的相關資訊


# Parameters
criterion = nn.CrossEntropyLoss() # 使用 cross entropy作為loss function之方式
lr = 0.001 # learning rate
epochs = 3 # 訓練3次
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)



# Train
for epoch in range(epochs):
    running_loss = 0.0

    for times, data in enumerate(trainLoader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()

        if times % 100 == 99 or times+1 == len(trainLoader):
            print('[%d/%d, %d/%d] loss: %.3f' % (epoch+1, epochs, times+1, len(trainLoader), running_loss/100))
            running_loss = 0

print('Finished Training\n')


# Test
# top 1 和 top 5的結果
top1_correct = 0
top5_correct = 0
total = 0
with torch.no_grad():
    for data in testLoader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = net(inputs)
        labels_resize = labels.view(-1, 1)

        _, top1_predicted = torch.max(outputs.data, 1)
        _, top5_predicted = outputs.topk(5, 1, True, True)

        total += labels.size(0)
        top1_correct += (top1_predicted == labels).sum().item()
        top5_correct += (top5_predicted == labels_resize).sum().item()

print('Accuracy of the network on the 10000 test inputs: %d %%' % (100 * top1_correct / total))
print('Accuracy of the network on the 10000 test inputs: %d %%' % (100 * top5_correct / total))

# 每個 class 的準確度
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testLoader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(8):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

for i in range(10):
    print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))

summary((net), (3, 32, 32))