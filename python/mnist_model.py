#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

#%%
#class Net(nn.Module):
#    def __init__(self):
#        super(Net, self).__init__()
#        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
#        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
#        self.conv2_drop = nn.Dropout2d()
#        self.fc1 = nn.Linear(320, 50)
#        self.fc2 = nn.Linear(50, 10)
#
#    def forward(self, x):
#        x = F.relu(F.max_pool2d(self.conv1(x), 2))
#        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
#        x = x.view(-1, 320)
#        x = F.relu(self.fc1(x))
#        x = F.dropout(x, training=self.training)
#        x = self.fc2(x)
#        return F.log_softmax(x)
#%%

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, kernel_size=5)


    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        return F.log_softmax(x)

#%%
network = Net()

# Create empty data
data = torch.empty([1,1,28,28])
data = torch.ones([1,1,28,28])
picture = torch.ones([28,28])
data = picture.unsqueeze(0).unsqueeze(0)

# Perform inference for a single data object
result = network(data[0:1])
print(result)

#%%
network.conv1.weight.shape