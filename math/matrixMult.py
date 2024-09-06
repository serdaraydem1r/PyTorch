#%%
import numpy as np
import torch
#%% numpy
# create random matrices
a = np.random.randn(3,4)
b = np.random.randn(4,5)
c = np.random.randn(3,7)

# try some multiplications
print(np.round(a@b ,2)), print(' ')
#print(np.round(a@c ,2)), print(' ')
#print(np.round(b@c ,2)), print(' ')
print(np.round(c.T@a,2)), print(' ')

#%% torch

# create random matrices
ta = torch.rand(3,4)
tb = torch.rand(4,5)
tc = torch.rand(1,5)

# try some multiplications
print(torch.round(ta@tb))
print(torch.round(tb@tc.T))

print(np.round(ta@tb))
print(np.round(tb@tc.T))