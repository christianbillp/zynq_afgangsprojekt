#%%
import torch
from torch.nn import Conv2d, Module
from torch.nn.functional import relu, max_pool2d, log_softmax
import numpy as np

k_dim = 3
kernel = torch.ones([k_dim, k_dim])

class Net1(Module):
    """Network structure:
        Convolution -> Maxpooling -> log_softmax
        1 kernel, 5x5    2x2
    
    
    """
    def __init__(self):
        super(Net1, self).__init__()
        self.conv1 = Conv2d(1, 1, kernel_size=3)

    def forward(self, x):
#        x = relu(max_pool2d(self.conv1(x), kernel_size=2))

        x = self.conv1(x)
        print("Net1 conv: {}".format(x.shape))
        x = max_pool2d(x, kernel_size=2)
        print("Net1 pool: {}".format(x.shape))
        x = relu(x)

#        return log_softmax(x)
        return x
    
class Net2(Module):
    """Network structure:
        Convolution -> Maxpooling -> log_softmax
        1 kernel, 5x5    2x2
    
    
    """
    def __init__(self):
        super(Net2, self).__init__()

    def forward(self, x):
        x = sconv(x, 0)
        print("Net2 conv: {}".format(x.shape))
        x = spool(x)
        print("Net2 pool: {}".format(x.shape))
        x = srelu(x)
        
#        return log_softmax(x)
        return x

network1 = Net1()
network2 = Net2()

# Create picture of ones
picture = torch.ones([28,28])

# Fit picture to data format
data = picture.unsqueeze(0).unsqueeze(0)

# Load mnist_example
mnist_example = torch.load('mnist_example.pt')

# Set weights to 1
network1.conv1.weight.data.fill_(1)

# Set bias to 0
network1.conv1.bias.data.fill_(0)

# Perform inference for a single data object
#result = network(data)
result1 = network1(mnist_example)
result2 = network2(mnist_example)
diff = result1-result2
non_zero = np.count_nonzero(diff.detach().numpy())

print("""Results:
    Net1: {}
    Net2: {}
    Non_zero: {}""".format(result1.shape, result2.shape, non_zero))

#%%
def nsoftmax_log(image):
    for row in range(image.shape[0]):
        for column in range(image.shape[1]):
            if image[row][column] < 0:
                image[row][column] = 0
    return image
    np.log( np.exp(x_i) / np.exp(x).sum() )


#%%
def nrelu(image):
    for row in range(image.shape[0]):
        for column in range(image.shape[1]):
            if image[row][column] < 0:
                image[row][column] = 0
    return image

def srelu(image):
    output = nrelu(image[0][0]).unsqueeze(0).unsqueeze(0)
    return output
                
    
    
#%% Low-level convolution without bias
kernels = {'sharpen': torch.tensor([[ 0,-1, 0],
                                    [-1, 5,-1],
                                    [ 0,-1, 0]], dtype=torch.float32),
            'edge_detect' :  torch.tensor([[-1,-1,-1],
                                           [-1, 8,-1],
                                           [-1,-1,-1]], dtype=torch.float32),
            'emboss' :  torch.tensor([[-2,-1, 0],
                                      [-1, 1, 1],
                                      [ 0, 1, 2]], dtype=torch.float32),
}

#kernel = torch.tensor([[1.0, 0, 0],
#                       [0, 1.0, 0],
#                       [0, 0 ,1.0]])
    
#k_dim = 7
#kernel = torch.ones([k_dim, k_dim])


def convolute(image, kernel, bias):
    # Performs a single convolution
    
    
    kernel_size = int((kernel.shape[0] - 1) / 2)

    # Create smaller output tensor
    output = torch.zeros([image.shape[0] - kernel_size*2, image.shape[1] - kernel_size*2])
    
    for row in range(output.shape[0]):
        for column in range(output.shape[1]):
            total = 0
            for kernel_row in range(kernel.shape[0]):
                for kernel_column in range(kernel.shape[1]):
                    total += image[row + kernel_row][column + kernel_column]* kernel[kernel_row, kernel_column]

            # Activation function goes here
            output[row][column] = total+bias
            
    return output

def sconv(image, bias):
    output = convolute(image[0][0], kernel, bias).unsqueeze(0).unsqueeze(0)

    return output

def maxpool(image, kernel_size=2):
    
    # Create smaller output tensor
    output = torch.zeros([image.shape[0]//2, image.shape[1]//2])
    
    for row in range(image.shape[0]//2):
        for column in range(image.shape[1]//2):
 
            # Check for largest value
            temp = torch.zeros([kernel_size*2])
            temp[0] = image[row*2][column*2]
            temp[1] = image[row*2+1][column*2]
            temp[2] = image[row*2][column*2+1]
            temp[3] = image[row*2+1][column*2+1]
            output[row][column] = max(temp)
    
    return output

def spool(image):
    output = maxpool(image[0][0]).unsqueeze(0).unsqueeze(0)
    return output



            
#%%


import matplotlib.pyplot as plt

def compare(a, b):
    plt.subplot(2,2,1)
    plt.imshow(a, cmap='gray', interpolation='none')
    plt.title('original')
    plt.subplot(2,2,2)
    plt.imshow(b, cmap='gray', interpolation='none')
    plt.title('convoluted')


#compare(mnist_example[0][0], output)
compare(result1.detach().numpy()[0][0],  result2.detach().numpy())

def check_data(data):
    print("Diagonal:")
    for row in range(data.shape[0]):
        for column in range(data.shape[1]):
            if row == column:
                print("{:.2f}".format(float(data[row][column])), end=' ')

def check_output(data):
    print("Diagonal:")
    for row in range(data.shape[0]):
        for column in range(data.shape[1]):
            if row == column:
                print("{:.2f}".format(float(data[row][column])), end=' ')

check_data(mnist_example[0][0])
check_output(output)

def show(data):
    for row in data:
        for value in row:
            print(round(float(value),2),end=' ')
        print("")
