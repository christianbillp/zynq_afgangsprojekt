#%%
import numpy as np
import torch

data = np.ones([1,1,28,28])
data = np.round(np.random.rand(1,1,28,28), 0)

CO1_B = 1
CO1_F = 1
CO1_R = 28
CO1_C = 28
CO1_K_R = 3
CO1_K_C = 3

CO1_OB = 1
CO1_OF = 5
CO1_OR = 26
CO1_OC = 26

MR1_OB = 1
MR1_OF = 5
MR1_OR = 13
MR1_OC = 13

CO2_B = MR1_OB
CO2_F = MR1_OF
CO2_R = MR1_OR
CO2_C = MR1_OC
CO2_K_R = 3
CO2_K_C = 3

CO2_OB = 1
CO2_OF = 5
CO2_OR = 11
CO2_OC = 11

MR2_OB = 1
MR2_OF = 5
MR2_OR = 5
MR2_OC = 5

CO1_OUTPUT = np.zeros([CO1_OB, CO1_OF, CO1_OR, CO1_OC])
MR1_OUTPUT = np.ones([MR1_OB, MR1_OF, MR1_OR, MR1_OC])
CO2_OUTPUT = np.zeros([CO2_OB, CO2_OF, CO2_OR, CO2_OC])
MR2_OUTPUT = np.ones([MR2_OB, MR2_OF, MR2_OR, MR2_OC])

trained_model = torch.load('mnist_model.pth')
mnist_example = example_data[2][0].unsqueeze(0).unsqueeze(0)
data = mnist_example.numpy()

#CO1_KERNEL = np.ones([5,1,3,3])
#CO1_BIAS = np.ones(5)
#CO2_KERNEL = np.ones([5,1,3,3])
#CO2_BIAS = np.ones(5)
CO1_KERNEL = trained_model['conv1.weight'].numpy() # (5, 1, 3, 3)
CO2_KERNEL = trained_model['conv2.weight'].numpy() # (5, 5, 3, 3)
CO1_BIAS = trained_model['conv1.bias'].numpy() # (5,)
CO2_BIAS = trained_model['conv2.bias'].numpy()


def CO1():
    """
    Function performs convolution on 4 dimensional data
    Output row and cols are 26
    
    Data does not have multiple features
    Kernels have dimension (5, 1, 3, 3) thus each kernel has weights at first index kernel[i][0]
    """
    global data, CO1_OUTPUT, CO1_KERNEL

    for b in range(CO1_OB):
        for f in range(CO1_OF):
            for r in range(CO1_OR):
                for c in range(CO1_OC):
                    convoluted = 0
                    for k_r in range(CO1_K_R):
                        for k_c in range(CO1_K_C):
                            convoluted += data[b][0][r+k_r][c+k_c] * CO1_KERNEL[f][0][k_r][k_c]

                    CO1_OUTPUT[b][f][r][c] = convoluted + CO1_BIAS[f]

def MR1():
    global CO1_OUTPUT, MR1_OUTPUT
    
    for b in range(MR1_OB):
        for f in range(MR1_OF):
            for r in range(MR1_OR):
                for c in range(MR1_OC):
                    frame = np.zeros(4)
                    frame[0] = CO1_OUTPUT[b][f][r*2][c*2]
                    frame[1] = CO1_OUTPUT[b][f][r*2+1][c*2]
                    frame[2] = CO1_OUTPUT[b][f][r*2][c*2+1]
                    frame[3] = CO1_OUTPUT[b][f][r*2+1][c*2+1]
            
                    maximum = 0
                    for i in range(4):
                        if frame[i] > maximum:
                            maximum = frame[i]

                    MR1_OUTPUT[b][f][r][c] = maximum

CO2_TEMP = torch.zeros(5, 5, 11, 11)

def CO2():
    """
    Function performs convolution on 4 dimensional data
    Output format (1, 5, 11, 11)
    """
    global MR1_OUTPUT, CO2_OUTPUT, CO2_KERNEL, CO2_TEMP
    
    temp = torch.zeros(5, 11, 11)

    for b in range(CO2_OB):
        for f in range(CO2_OF):
            for k_n in range(5):                    
                for r in range(CO2_OR):
                    for c in range(CO2_OC):
                        convoluted = 0
                        for k_r in range(CO2_K_R):
                            for k_c in range(CO2_K_C):
                                convoluted += MR1_OUTPUT[b][f][r+k_r][c+k_c] * CO2_KERNEL[f][k_n][k_r][k_c]# + CO2_BIAS[k_n]
                                temp[k_n][r][c] = convoluted
            CO2_TEMP[f] = temp
#            print(temp)
            
                        
                        
#                CO2_OUTPUT[b][f][r][c] = 
                    
                    
#                    for 
#                    CO2_OUTPUT[b][f][r][c] = convoluted + CO2_BIAS[f]

#def MR2():
#    global CO2_OUTPUT, MR2_OUTPUT
#    
#    for b in range(MR2_OB):
#        for f in range(MR2_OF):
#            for r in range(MR2_OR):
#                for c in range(MR2_OC):
#                    frame = np.zeros(4)
#                    frame[0] = CO2_OUTPUT[b][f][r*2][c*2]
#                    frame[1] = CO2_OUTPUT[b][f][r*2+1][c*2]
#                    frame[2] = CO2_OUTPUT[b][f][r*2][c*2+1]
#                    frame[3] = CO2_OUTPUT[b][f][r*2+1][c*2+1]
#            
#                    maximum = 0
#                    for i in range(4):
#                        if frame[i] > maximum:
#                            maximum = frame[i]
#
#                    MR2_OUTPUT[b][f][r][c] = maximum

CO1()
a = CO1_OUTPUT[0][0]
MR1()
b = MR1_OUTPUT[0][0]

CO2()
c = CO2_TEMP
#c=CO2_OUTPUT[0][0]
#cMR2()
#d=MR2_OUTPUT[0][0]
#print(a, b, c, d)
#CO1_OUTPUT
#MR1_OUTPUT
#CO2_OUTPUT
#MR2_OUTPUT

#%%
def combine(data):
    output = torch.zeros(1,5,11,11)
    for i in range(5):
        output[0][i] = sum(data[i])
    
    return output
d=combine(c)[0][0]
result1 = network1(mnist_example)
r = result1[0][0].detach().numpy()

#%%
from matplotlib import pyplot as plt
heatmap = plt.pcolor(c[0][0])

#%%

#%%
import torch

a = result1 # torch.Size([1, 5, 13, 13])
b = torch.tensor(MR1_OUTPUT)
c = torch.tensor(CO2_OUTPUT)
m = a[0][0]
n = CO2_TEMP[0][0] + CO2_TEMP[0][1]  + CO2_TEMP[0][2] + CO2_TEMP[0][3]  + CO2_TEMP[0][4]






