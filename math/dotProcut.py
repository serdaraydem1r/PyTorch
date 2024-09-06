#%%
import numpy as np
import torch
#%% numpy
# create vector
nv1 = np.array([1,2,3,4])
nv2 = np.array([5,6,7,-8])
print(nv1), print(' ')
print(nv2), print(' ')
# dot product
print(np.dot(nv1,nv2))

# create matrix
nm1 = np.array([[1,2,3,4],[5,6,7,-8],[9,-2,4,7]])
nm2 = np.array([[1,2,-3,0],[5,6,0,-3],[0,-5,0,1]])
print(nm1), print(' ')
print(nm2), print(' ')
# dot product
print(np.dot(nm1,nm2))
#%% torch
# create vector
tv1 = torch.tensor([1,2,0,-9])
tv2 = torch.tensor([5,6,-7,-8])
print(tv1), print(' ')
print(tv2), print(' ')
print(torch.dot(tv1,tv2))


