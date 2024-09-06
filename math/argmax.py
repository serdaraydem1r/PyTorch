#%%
import numpy as np
import torch
import torch.nn as nn

#%% numpy

# create a vector
V = np.array([1,40,2,-3])

# find and report the maximum and minimum values
minval = np.min(V)
maxval = np.max(V)
print(f'Minval: {minval}\nMaxval: {maxval}') # -3, 40

# find and report the argmin and argmax index
minidx = np.argmin(V)
maxidx = np.argmax(V)
print(f'Minidx: {minidx}\nMaxidx: {maxidx}') # 3, 1

# create matrix
M = np.array([[0,1,10],
              [20,8,5]])
print(f'M: {M}'), print(' ')

# various minima in this matrix
minvals1 = np.min(M) # minimum from entıre matrix
minvals2 = np.min(M,axis=0) # minimum of each column
minvals3 = np.min(M,axis=1) # minimum of each row

print(f'minvals1: {minvals1}')
print(f'minvals2: {minvals2}')
print(f'minvals3: {minvals3}')

#%% pytorch
v = torch.tensor([1,40,2,-3])
minval = torch.min(v)
maxval = torch.max(v)
print(f'minval: {minval}')
print(f'maxval: {maxval}')

minidx = torch.argmin(v)
maxidx = torch.argmax(v)
print(f'minidx: {minidx}')
print(f'maxidx: {maxidx}')

# matrix

M = torch.tensor([[0,1,10],[20,8,5]])
print(f'M: {M}'),print(' ')

min1 = torch.min(M)
min2 = torch.min(M, axis=0)
min3 = torch.min(M, axis=1)
print(f'min1: {min1}')
print(f'min2: {min2}')
print(f'min3: {min3}')
print(min2.values)
print(min2.indices)