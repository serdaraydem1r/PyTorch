#%%
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
#%% manually in numpy
# the list of numbers
z = [1,2,3]
# compute the softmax result
num = np.exp(z)
den = np.sum(np.exp(z))
sigma = num / den
print(sigma)
print(np.sum(sigma))

#%% random integers
z_random = np.random.randint(-10,20, 6400)
num_random = np.exp(z_random)
den_random = np.sum(np.exp(z_random))
sigma_random = num_random / den_random
print(sigma_random)
print(np.sum(sigma_random))

#%%
plt.figure(figsize=(20,20))
plt.plot(z_random, sigma,'ko')
plt.xlabel('Original Number (Z)')
plt.ylabel('Softmax')
plt.show()
#%% pytorch
softfun = nn.Softmax(dim=0)
sigmaT = softfun(torch.tensor(z))
print(sigmaT)
