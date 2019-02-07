#%%
import numpy as np
w = trained_model['fc3.weight']
b = trained_model['fc3.bias']
r = result2.detach().numpy()



#%%
x = r

def fc(x):
    print(x.shape)

    output_len = 10
    output = torch.ones(output_len)

    # Iteration for every output neuron
    for i in range(output_len):
        total = torch.ones(1)
        total = np.dot(x[0], w[i])
#        for j in range(125):
#            total += x[0][j] * w[i][j]
        output[i] = total + b[i]
    print("w: {}".format(w.shape))
    print(output)
    
    return output.unsqueeze(0)
        
        
   
    
