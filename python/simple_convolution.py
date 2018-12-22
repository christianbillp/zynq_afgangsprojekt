#%%
import torch
from torch.nn import Conv2d, Module
from torch.nn.functional import relu, max_pool2d, log_softmax

class Net(Module):
    """Network structure:
        Convolution -> Maxpooling -> log_softmax
        1 kernel, 5x5    2x2
    
    
    """
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = Conv2d(1, 1, kernel_size=3)

    def forward(self, x):
#        x = relu(max_pool2d(self.conv1(x), kernel_size=2))
        x = self.conv1(x)
        
#        return log_softmax(x)
        return x

network = Net()

# Create picture of ones
picture = torch.ones([28,28])

# Fit picture to data format
data = picture.unsqueeze(0).unsqueeze(0)

# Load mnist_example
mnist_example = torch.load('mnist_example.pt')

# Set weights to 1
network.conv1.weight.data.fill_(1)

# Set bias to 0
#network.conv1.bias.data.fill_(0)

# Perform inference for a single data object
#result = network(data)
result = network(mnist_example)
print(result)

#%%
print(network.conv1.weight.shape)
print(network.conv1.weight)
#%%
def show(data):
    for row in data:
        for value in row:
            print(round(float(value),2),end=' ')
        print("")


#%% Low-level convolution without bias

kernel = torch.tensor([[1.0, 0, 0],
                       [0, 1.0, 0],
                       [0, 0 ,1.0]])

kernel = torch.tensor([[2, -10, 0],
                       [2, -10, 0],
                       [2, -10, 0]], dtype=torch.float32)

kernel = torch.ones([3,3])
kernel_size = int((kernel.shape[0] - 1) / 2)

def convolute(image, kernel):
    # Prototype for C function
    
    # Create smaller output tensor
    output = torch.zeros([image.shape[0] - kernel_size*2, image.shape[1] - kernel_size*2])
    
    for row in range(output.shape[0]):
        for column in range(output.shape[1]):
            total = 0
            for kernel_row in range(kernel.shape[0]):
                for kernel_column in range(kernel.shape[1]):
                    total += image[row + kernel_row][column + kernel_column]* kernel[kernel_row, kernel_column]
    
            output[row][column] = total
            
    return output


#output = convolute(picture, kernel)
output = convolute(mnist_example[0][0], kernel)
#print(output)
#print(output.shape)
#show(output)



import matplotlib.pyplot as plt

def compare(a, b):
    plt.subplot(2,2,1)
    plt.imshow(a, cmap='gray', interpolation='none')
    plt.title('original')
    plt.subplot(2,2,2)
    plt.imshow(b, cmap='gray', interpolation='none')
    plt.title('convoluted')


compare(mnist_example[0][0], output)


#%%
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
