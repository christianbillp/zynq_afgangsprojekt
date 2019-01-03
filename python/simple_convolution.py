#%%
import torch
from torch.nn import Conv2d, Module
from torch.nn.functional import relu, max_pool2d, log_softmax
import numpy as np

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
    

#%%
def convolute(image, kernel):
    # Performs a single convolution
    
    
    kernel_width = int((kernel.shape[0] - 1) / 2)

    # Create smaller output tensor
    output = torch.zeros([image.shape[0] - kernel_width*2, image.shape[1] - kernel_width*2])
    
    for row in range(output.shape[0]):
        for column in range(output.shape[1]):
            total = 0
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
        print(f"Image: {i}")
        for j in range(n_kernels):
            result = 0
            for kernel in kernels[j]:
                print(kernel)
                result += convolute(images[i][0], kernel) # First channel in image_j
            print(result)
            output[i][j] = result + biases[j]
    
    return output

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


k_dim = 3
kernel = torch.ones([k_dim, k_dim])
trained_model = torch.load('mnist_model.pth')
kernels1 = trained_model['conv1.weight']
kernels2 = trained_model['conv2.weight']
biases1 = trained_model['conv1.bias']
biases2 = trained_model['conv2.bias']

#k_dim = 3
#n_kernels1 = 5
#n_kernels2 = 5
#kernels1 = torch.ones([n_kernels1, k_dim, k_dim])
#kernels2 = torch.ones([n_kernels2, k_dim, k_dim])




class Net1(Module):

    def __init__(self):
        super(Net1, self).__init__()
        self.conv1 = Conv2d(1, 5, kernel_size=3)
        self.conv2 = Conv2d(5, 5, kernel_size=3)

    def forward(self, x):
        print(x.shape)
        x = self.conv1(x)
        print("Net1 conv1: {}".format(x.shape))
        x = max_pool2d(x, kernel_size=2)
        print("Net1 pool2: {}".format(x.shape))
        x = relu(x)
        
        x = self.conv2(x)
        print("Net1 conv2: {}".format(x.shape))
#        x = max_pool2d(x, kernel_size=2)
#        print("Net1 pool2: {}".format(x.shape))
#        x = relu(x)

#        return log_softmax(x)
        return x
    
class Net2(Module):
    def __init__(self):
        super(Net2, self).__init__()

    def forward(self, x):
        x = sconv(x, n_kernels=5, kernel_size=3, kernels=kernels1, biases=biases1)
        print("Net2 conv1: {}".format(x.shape))
        x = spool(x)
        print("Net2 pool1: {}".format(x.shape))
        x = srelu(x)
        
        x = sconv(x, n_kernels=5, kernel_size=3, kernels=kernels2, biases=biases2)
        print("Net2 conv2: {}".format(x.shape))
#        x = spool(x)
#        print("Net2 pool2: {}".format(x.shape))
#        x = srelu(x)

#        return log_softmax(x)
        return x


network1 = Net1()
network2 = Net2()

network1.conv1.weight.data = trained_model['conv1.weight']
network1.conv1.bias.data = trained_model['conv1.bias']
network1.conv2.weight.data = trained_model['conv2.weight']
network1.conv2.bias.data = trained_model['conv2.bias']


# Create picture of ones
#picture = torch.ones([28,28])

# Fit picture to data format
#data = picture.unsqueeze(0).unsqueeze(0)

# Load mnist_example
mnist_example = torch.load('mnist_example.pt')

# Set weights to 1
#network1.conv1.weight.data.fill_(1)
#network1.conv2.weight.data.fill_(1)

# Set bias to 0
#network1.conv1.bias.data.fill_(1)
#network1.conv2.bias.data.fill_(1)

# Perform inference for a single data object
#result = network(data)
result1 = network1(mnist_example)
result2 = network2(mnist_example)
diff = result1 - result2
non_zero = np.count_nonzero(diff.detach().numpy())

print("""Results:
    Net1: {}
    Net2: {}
    Non_zero: {}
    Diff: {}""".format(result1.shape, result2.shape, non_zero, diff.sum()))

i = 2
compare(result1.detach().numpy()[0][i],  result2.detach().numpy()[0][i])
#%%
for i in range(5):
    d = result1.detach().numpy()[0][i] -  result2.detach().numpy()[0][i]
    for a in np.histogram(d):
        print(a)
#%%
def nsoftmax_log(image):
    for row in range(image.shape[0]):
        for column in range(image.shape[1]):
            if image[row][column] < 0:
                image[row][column] = 0
    return image
    np.log( np.exp(x_i) / np.exp(x).sum() )
            
#%%


import matplotlib.pyplot as plt

def compare(a, b):
    plt.subplot(2,2,1)
    plt.imshow(a, cmap='gray', interpolation='none')
    plt.title('original')
    plt.subplot(2,2,2)
    plt.imshow(b, cmap='gray', interpolation='none')
    plt.title('convoluted')

compare(result1.detach().numpy()[0][0],  result2.detach().numpy()[0][0])
#%%
#compare(mnist_example[0][0], output)

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
