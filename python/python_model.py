#%%
import torch
from torch.nn import Module

import numpy as np
#%% Load sample data
trained_model_parameters = torch.load('mnist_model.pth')
torch.set_printoptions(precision=4)
trained_model_parameters['conv1.weight']
trained_model_parameters['conv1.bias']


mnist_example = torch.load('mnist_example.pt')
conv_1_kernels = trained_model_parameters['conv1.weight']
conv_1_biases = trained_model_parameters['conv1.bias']
#conv_1_biases = torch.zeros(5)

#np.savetxt('conv_1_kernels.csv', conv_1_kernels.detach().numpy())
# (5, 1, 3, 3)
def np_to_c(kernels):
    return_string = ""
    for kernel in kernels:
        for channel in kernel:
            print("{", end='')
            for kernel_row in channel:
                print("{", end='')
                for kernel_value in kernel_row:
                    print(kernel_value, end=',')
                print("}")
            print("},")
            

#%%
a = check_data(convolute(mnist_example[0][0], conv_1_kernels[0][0]))

b = "-0.99 -0.99 -0.99 -0.99 -0.99 0.41 3.83 5.05 5.60 1.26 -0.49 -0.99 -0.99 -0.04 3.18 4.09 3.53 4.37 -0.01 -1.07 -0.99 -0.99 -0.99 -0.99 -0.99 -0.99 "

result[0][2][3][4]


#%% Convolution

def convolute(image, kernel):
    kernel_width = int((kernel.shape[0] - 1) / 2)

    output = torch.zeros([image.shape[0] - kernel_width*2, image.shape[1] - kernel_width*2])
    
    for row in range(output.shape[0]):
        for column in range(output.shape[1]):
            total = torch.zeros(1)
            for kernel_row in range(kernel.shape[0]):
                for kernel_column in range(kernel.shape[1]):
                    total += image[row + kernel_row][column + kernel_column] * kernel[kernel_row][kernel_column]

            output[row][column] = total
            
    return output

def sconv(images, n_kernels, kernel_size, kernels, biases):
    # n_kernels: Kernels per input image
    # Does the same as a pytorch convolution layer
    kernel_width = int((kernel_size - 1) / 2)
    output = torch.zeros([len(images), n_kernels, images[0][0].shape[0] - kernel_width*2, images[0][0].shape[1] - kernel_width*2])

    for i in range(len(images)):
        for j in range(n_kernels):
            result = 0
            for kernel in kernels[j]:
                result += convolute(images[i][0], kernel) # First channel in image_j
            output[i][j] = result + biases[j]
    
    return output

#%% Max Pool
def maxpool(image, kernel_size=2):

    rows = image.shape[0]//2
    columns = image.shape[1]//2

    # Create smaller output tensor
    output = torch.zeros([rows, columns])

    for row in range(rows):
        for column in range(columns):
 
            # Check for largest value
            temp = torch.zeros([kernel_size*2])
            temp[0] = image[row*2][column*2]
            temp[1] = image[row*2+1][column*2]
            temp[2] = image[row*2][column*2+1]
            temp[3] = image[row*2+1][column*2+1]
            output[row][column] = max(temp)
#            print(max(temp))
    
    return output

def spool(images):
    rows = images[0][0].shape[0]//2
    columns = images[0][0].shape[1]//2

    output = torch.zeros([1, 5, rows, columns])

    for i in range(len(images[0])):
        output[0][i] = maxpool(images[0][i]).unsqueeze(0).unsqueeze(0)

    return output

def nrelu(image):
    for row in range(image.shape[0]):
        for column in range(image.shape[1]):
            if image[row][column] < 0:
                image[row][column] = 0
    return image

def srelu(images):
    rows = images[0][0].shape[0]
    columns = images[0][0].shape[1]
    n_images = len(images[0])
    output = torch.zeros([1, n_images, rows, columns])

    for i in range(n_images):

        output[0][i] = nrelu(images[0][i]).unsqueeze(0).unsqueeze(0)
    return output


#%% Test network
class Net(Module):
    def __init__(self):
        super(Net, self).__init__()

    def forward(self, x):
        x = sconv(x, n_kernels=5, kernel_size=3, kernels=conv_1_kernels, biases=conv_1_biases)
        x = spool(x)
        x = srelu(x)
#        
#        x = sconv(x, n_kernels=5, kernel_size=3, kernels=kernels2, biases=biases2)
#        x = spool(x)
#        x = srelu(x)
#        
#        x = x.view(-1, 125)
#        x = fc(x, weights=fc_weights, biases=biases3)

#        return log_softmax(x)
        return x

network = Net()
result = network(mnist_example)
#for item in result[0]:
#    check_data(item)
#%%    
result[0][2][9][9]