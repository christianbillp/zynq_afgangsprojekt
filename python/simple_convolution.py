#%%
import torch
from torch.nn import Conv2d, Module, Linear
from torch.nn.functional import relu, max_pool2d, log_softmax
import numpy as np

#%%
def convolute(image, kernel):
    # Performs a single convolution
    kernel_width = int((kernel.shape[0] - 1) / 2)

    # Create smaller output tensor
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

def fc(x, weights, biases):
    print(x.shape)

    output_len = 10
    output = torch.ones(output_len)

    for i in range(output_len):
        total = torch.zeros(1)
        for j in range(125):
            total += x[0][j] * weights[i][j]
        output[i] = total + biases[i]
    
    return output.unsqueeze(0)

trained_model = torch.load('mnist_model.pth')

class Net1(Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.conv1 = Conv2d(1, 5, kernel_size=3)
        self.conv2 = Conv2d(5, 5, kernel_size=3)
        self.fc3 = Linear(125, 10)

    def forward(self, x):
        print(x.shape)
        x = self.conv1(x)
        print(x.shape)
        x = max_pool2d(x, kernel_size=2)
        print(x.shape)
        x = relu(x)
        print(x.shape)
        
        x = self.conv2(x)
        print(x.shape)
#        x = max_pool2d(x, kernel_size=2)
#        print(x.shape)
#        x = relu(x)
#        print(x.shape)

#        x = x.view(-1, 125)
#        print(x.shape)
#        x = self.fc3(x)
#        print(x.shape)

#        return log_softmax(x)
        return x
    
class Net2(Module):
    def __init__(self):
        super(Net2, self).__init__()

    def forward(self, x):
        x = sconv(x, n_kernels=5, kernel_size=3, kernels=kernels1, biases=biases1)
        x = spool(x)
        x = srelu(x)
        
        x = sconv(x, n_kernels=5, kernel_size=3, kernels=kernels2, biases=biases2)
        x = spool(x)
        x = srelu(x)
        
        x = x.view(-1, 125)
        x = fc(x, weights=fc_weights, biases=biases3)

#        return log_softmax(x)
        return x


network1 = Net1()
network2 = Net2()

network1.conv1.weight.data = trained_model['conv1.weight']
network1.conv1.bias.data = trained_model['conv1.bias']
network1.conv2.weight.data = trained_model['conv2.weight']
network1.conv2.bias.data = trained_model['conv2.bias']
network1.fc3.weight.data = trained_model['fc3.weight']
network1.fc3.bias.data = trained_model['fc3.bias']

kernels1 = trained_model['conv1.weight']
biases1 = trained_model['conv1.bias']
kernels2 = trained_model['conv2.weight']
biases2 = trained_model['conv2.bias']
fc_weights = trained_model['fc3.weight']
biases3 = trained_model['fc3.bias']

# Load mnist_example
#mnist_example = torch.load('mnist_example.pt')
mnist_example = example_data[2][0].unsqueeze(0).unsqueeze(0)

# Run test
result1 = network1(mnist_example)
result2 = network2(mnist_example)

i = 1
diff = result1 - result2
non_zeros = [np.count_nonzero(diff.detach().numpy()[0][i]) for i in range(len((diff.detach().numpy()[0])))]
non_zero = np.count_nonzero(diff.detach().numpy()[0][i])

print("""Results:
    Net1: {}
    Net2: {}
    Non_zeros: {}
    Diff: {}""".format(result1.shape, result2.shape, non_zeros, diff.sum()))
print(result1)
print(result2)

print(np.argmax(result1.detach().numpy()) - np.argmax(result2.detach().numpy()))

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

def show(data):
    for row in data:
        for value in row:
            print(round(float(value),2),end=' ')
        print("")
