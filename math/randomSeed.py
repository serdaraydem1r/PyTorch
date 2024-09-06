#%%
import numpy as np
import torch
#%%
randseed1 = np.random.RandomState(17)
randseed2 = np.random.RandomState(8)

print(randseed1.randn(5))
print(randseed2.randn(5))
print(randseed1.randn(5))
print(randseed2.randn(5))
print(np.random.randn(5))

#%%
torch.manual_seed(17)
print(torch.randn(5))
print(np.random.randn(5))