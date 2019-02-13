#%% Imports
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt

#%% Data loader
batch_size_test = 1000

test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('/files/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_test, shuffle=True)

examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)

#%% Reference Pytorch model

i = 6
results = []

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=3)
        self.fc1 = nn.Linear(1690, 10)

    def forward(self, x):
        x = self.conv1(x)
        results.append(x.detach().numpy())

        x = F.max_pool2d(x, kernel_size=2)

        x = F.relu(x)
        results.append(x.detach().numpy())

        x = x.view(-1, 1690)
        results.append(x.detach().numpy())

        x = self.fc1(x)
        results.append(x)

        x = F.log_softmax(x)
        results.append(x)
        
        return x

net = Net()

# Load weights and biases
trained_model = torch.load('new_model.pth')
net.conv1.weight.data = trained_model['conv1.weight']
net.conv1.bias.data = trained_model['conv1.bias']
net.fc1.weight.data = trained_model['fc3.weight']
net.fc1.bias.data = trained_model['fc3.bias']

mnist_example = example_data[i][0].unsqueeze(0).unsqueeze(0)
result = net(mnist_example)
plt.pcolor(np.flip(mnist_example[0][0], axis=0))
plt.title("Pytorch network prediction\n{}".format(np.where(result[0].detach().numpy() == float(min(result[0], key=abs)))))

# Define data input dimensions
CO1_B = 1
CO1_F = 1
CO1_R = 28
CO1_C = 28

# Define kernel dimensions
CO1_K_R = 3
CO1_K_C = 3

# Define convolution output dimensions
CO1_OB = 1
CO1_OF = 10
CO1_OR = 26
CO1_OC = 26

# Define maxpool / ReLU dimensions
MR1_OB = 1
MR1_OF = 10
MR1_OR = 13
MR1_OC = 13

# Set up memory buffers
CO1_OUTPUT = np.zeros([CO1_OB, CO1_OF, CO1_OR, CO1_OC])
MR1_OUTPUT = np.zeros([MR1_OB, MR1_OF, MR1_OR, MR1_OC])
FL1_OUTPUT = np.zeros([1, 1690])
FC1_OUTPUT = np.zeros([1, 10])

# Load model parameters
CO1_KERNEL = trained_model['conv1.weight'].numpy()      # (10, 1, 3, 3)
CO1_BIAS = trained_model['conv1.bias'].numpy()          # (10,)
FC1_WEIGHT = trained_model['fc3.weight'].numpy()        # (10, 1690)
FC1_BIAS = trained_model['fc3.bias'].numpy()            # (10,)

# Correct input data dimensions
data = mnist_example.numpy()

def CO1():
    """Convolution layer"""
    global data, CO1_OUTPUT, CO1_KERNEL

    for b in range(CO1_OB):
        for f in range(CO1_OF):
            # Convolution begins here
            for r in range(CO1_OR):
                for c in range(CO1_OC):
                    # Calculate kernel and receptive field dot product
                    convoluted = 0
                    for k_r in range(CO1_K_R):
                        for k_c in range(CO1_K_C):
                            convoluted += data[b][0][r+k_r][c+k_c] * CO1_KERNEL[f][0][k_r][k_c]

                    CO1_OUTPUT[b][f][r][c] = convoluted + CO1_BIAS[f]

def MR1():
    """Maxpools and ReLUs convolution output product"""
    global CO1_OUTPUT, MR1_OUTPUT

    for b in range(MR1_OB):
        for f in range(MR1_OF):
            # Maxpool begins here
            for r in range(MR1_OR):
                for c in range(MR1_OC):
                    frame = np.zeros(4)
                    frame[0] = CO1_OUTPUT[b][f][r*2  ][c*2  ]
                    frame[1] = CO1_OUTPUT[b][f][r*2+1][c*2  ]
                    frame[2] = CO1_OUTPUT[b][f][r*2  ][c*2+1]
                    frame[3] = CO1_OUTPUT[b][f][r*2+1][c*2+1]

                    # ReLU begins here
                    maximum = 0
                    for i in range(4):
                        if frame[i] > maximum:
                            maximum = frame[i]

                    MR1_OUTPUT[b][f][r][c] = maximum

def FL1():
    """Flattens data to: (1, 1690)"""
    global FL1_OUTPUT

    i = 0
    for b in range(MR1_OB):
        for f in range(MR1_OF):
            for r in range(MR1_OR):
                for c in range(MR1_OC):
                    FL1_OUTPUT[b][i] = MR1_OUTPUT[b][f][r][c]
                    i = i + 1

def FC1():
    """Fully connected layer output dimensions: (1, 10)"""
    global FC1_OUTPUT

    for i in range(10):
        total = np.zeros(1)
        for j in range(1690):
            total += FL1_OUTPUT[0][j] * FC1_WEIGHT[i][j]

        FC1_OUTPUT[0][i] = total + FC1_BIAS[i]


#data = input_data
CO1()
MR1()
FL1()
FC1()

#%% Calculate total pixelwise error
test = CO1_OUTPUT - results[0]
error = 0
for b in range(test.shape[0]):
    for f in range(test.shape[1]):
        for r in range(test.shape[2]):
            for c in range(test.shape[3]):
                error += test[b][f][r][c]
print("Total error in CO1 test: {:>14.10f}".format(error))

test = MR1_OUTPUT - results[1]
error = 0
for b in range(test.shape[0]):
    for f in range(test.shape[1]):
        for r in range(test.shape[2]):
            for c in range(test.shape[3]):
                error += test[b][f][r][c]
print("Total error in MR1 test: {:>14.10f}".format(error))

test = FL1_OUTPUT - results[2]
error = 0
for i in range(test.shape[0]):
    for j in range(test.shape[1]):
        error += test[i][j]
print("Total error in FL1 test: {:>14.10f}".format(error))

test = FC1_OUTPUT - results[3].detach().numpy()
error = 0
for i in range(test.shape[0]):
    for j in range(test.shape[1]):
        error += test[i][j]
print("Total error in FC1 test: {:>14.10f}".format(error))

error = 0
for i in range(10):
    error += F.log_softmax(torch.tensor(FC1_OUTPUT[0][i])).detach().numpy() - F.log_softmax(results[3][0][1]).detach().numpy()
print("Total error in FC1 test: {:>14.10f}".format(error))

#%% Compare results visually
plt.subplot(131)
plt.pcolor(data[0][0])
avg = round(np.mean(data[0][0]), 4)
plt.title("Original\nMean: {:.6f}".format(avg))

plt.subplot(132)
plt.pcolor(results[0][0][0])
avg = round(np.mean(results[0][0][0]), 4)
plt.title("Ref network\nMean: {:.6f}".format(avg))

plt.subplot(133)
plt.pcolor(CO1_OUTPUT[0][0])
avg = round(np.mean(CO1_OUTPUT[0][0]), 4)
plt.title("New network\nMean: {:.6f}".format(avg))

#%% Compare results visually
plt.subplot(121)
plt.pcolor(results[1][0][0])
avg = round(np.mean(results[1][0][0]), 4)
plt.title("Ref network\nMean: {:.6f}".format(avg))

plt.subplot(122)
plt.pcolor(MR1_OUTPUT[0][0])
avg = round(np.mean(MR1_OUTPUT[0][0]), 4)
plt.title("New network\nMean: {:.6f}".format(avg))

#%% End results
r = F.log_softmax(torch.tensor(FC1_OUTPUT[0])).detach().numpy()

plt.subplot(121)
plt.pcolor([results[4][0].detach().numpy()])
avg = round(np.mean(results[4][0].detach().numpy()), 4)
plt.title("Ref network\nMean: {:.6f}".format(avg))

plt.subplot(122)
plt.pcolor([r])
avg = round(np.mean(r), 4)
plt.title("New network\nMean: {:.6f}".format(avg))
#%%
np.where(r == float(min(r, key=abs)))

#%% Evaluate pixel value differences
for i in range(26):
#    a = net_results[0][0][0][i].detach().numpy()
    a = results[0][0][0][i]
    b = CO1_OUTPUT[0][0][i]
    print("{:>10.4f}".format(sum(a - b)))

